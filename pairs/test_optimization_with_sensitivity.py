"""
卡爾曼配對交易 - 參數優化測試

功能:
1. 指定配對進行參數優化
2. 支持槓桿和止損
3. 多進程並行評估
4. 魯棒性過濾和敏感性分析

使用方式:
python test_optimization_with_sensitivity.py
"""

import sys
import os
import logging
import json
import pickle
import tempfile
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from factors.kalman_pairs.factor import KalmanPairsFactor
from factors.kalman_pairs.position_strategy import KalmanPairsPositionStrategy
from factors.kalman_pairs.cointegration_utils import test_cointegration
from shared.cross_sectional_backtest import CrossSectionalBacktester
from shared.data_loader import load_local_data

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 配置區 ====================

# 指定交易配對 (手動指定一個配對進行測試)
SPECIFIED_PAIR = ('BTCUSDT', 'ETHUSDT')

# 日期範圍
START_DATE = "2022-11-01"
END_DATE = "2024-12-31"

# 參數網格
PARAM_GRID = {
    "delta": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    "Ve": [0.0001, 0.0005, 0.001, 0.005, 0.01],
    "entry_threshold": [1.0, 1.5, 2.0, 2.5, 3.0],
    "exit_threshold": [0.0, 0.5, 1.0],
    "leverage": [1, 2, 3],
    "rebalance_freq_days": [7, 14, 21, 30],
    "holding_days": [7, 14, 21, 30]
}

# 優化設定
N_TRIALS = 500  # 隨機搜索次數
SHARPE_THRESHOLD = 1.25
TURNOVER_MIN = 0.01
TURNOVER_MAX = 0.7

# 手續費設定
COMMISSION_BPS = 20.0   # 0.02%
STOP_LOSS_COMMISSION_BPS = 55.0  # 0.055%

# ==================== 多進程輔助函數 ====================

worker_cache = {}

def init_worker(shared_data: dict):
    """初始化工作進程"""
    global worker_cache
    os.environ["TQDM_DISABLE"] = "1"
    worker_cache.update(shared_data)

    # 從磁盤加載大對象
    if 'cache_file_path' in shared_data:
        cache_path = Path(shared_data['cache_file_path'])
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                worker_cache['price_data_cache'] = cached_data.get('price_data_cache', {})
                worker_cache['factor_cache'] = cached_data.get('factor_cache', {})


def evaluate_single_params(params: dict) -> dict:
    """
    評估單個參數組合

    Args:
        params: 參數字典

    Returns:
        包含參數和性能指標的字典
    """
    global worker_cache

    try:
        # 從 cache 獲取預計算的因子
        factor_cache_key = (params['delta'], params['Ve'])
        factor_df = worker_cache['factor_cache'].get(factor_cache_key)

        if factor_df is None:
            logger.error(f"因子未找到: delta={params['delta']}, Ve={params['Ve']}")
            return {**params, "sharpe_ratio": -999, "turnover": 0}

        # 計算止損閾值
        stop_loss_pct = -1.0 / params['leverage']

        # 創建回測器
        backtester = CrossSectionalBacktester(
            factor_calculator=None,  # 因子已預計算
            position_strategy=KalmanPairsPositionStrategy(),
            start_date=worker_cache['start_date'],
            end_date=worker_cache['end_date'],
            timeframe='1d',
            rebalance_freq=f"{params['rebalance_freq_days']}D",
            commission_bps=worker_cache['commission_bps'],
            leverage=params['leverage'],
            stop_loss_pct=stop_loss_pct,
            stop_loss_commission_bps=worker_cache['stop_loss_commission_bps'],
            debug_mode=False
        )

        # 設置預載入數據
        backtester.all_data = worker_cache['price_data_cache']

        # 運行回測
        results = backtester.run(
            factor_df=factor_df,
            entry_threshold=params['entry_threshold'],
            exit_threshold=params['exit_threshold'],
            holding_days=params['holding_days'],
            rebalance_freq_days=params['rebalance_freq_days'],
            leverage=params['leverage']
        )

        # 提取性能指標
        perf = results.get('performance_summary', {})

        return {
            **params,
            "sharpe_ratio": perf.get('sharpe_ratio', -999),
            "total_return": perf.get('total_return', 0),
            "max_drawdown": perf.get('max_drawdown', 0),
            "turnover": perf.get('turnover', 0)
        }

    except Exception as e:
        logger.error(f"參數評估失敗: {params}, 錯誤: {e}")
        return {**params, "sharpe_ratio": -999, "turnover": 0}


# ==================== 主流程 ====================

def main():
    logger.info("=== 卡爾曼配對交易 - 參數優化開始 ===")
    logger.info(f"配對: {SPECIFIED_PAIR[0]} - {SPECIFIED_PAIR[1]}")
    logger.info(f"時間範圍: {START_DATE} 到 {END_DATE}")

    # 1. 載入數據
    logger.info("步驟 1/6: 載入價格數據...")
    price_data_cache = {}

    for asset in SPECIFIED_PAIR:
        try:
            df = load_local_data(
                symbol=asset,
                data_source="1d",
                start_date=START_DATE,
                end_date=END_DATE
            )
            if df is not None and len(df) > 100:
                price_data_cache[asset] = df
                logger.info(f"載入 {asset}: {len(df)} 筆數據")
            else:
                raise ValueError(f"{asset} 數據不足")
        except Exception as e:
            logger.error(f"載入 {asset} 失敗: {e}")
            return

    # 2. 檢查協整性
    logger.info("步驟 2/6: 檢查配對協整性...")
    price1 = price_data_cache[SPECIFIED_PAIR[0]]['close']
    price2 = price_data_cache[SPECIFIED_PAIR[1]]['close']

    common_idx = price1.index.intersection(price2.index)
    is_coint, pvalue, _ = test_cointegration(
        price1.loc[common_idx],
        price2.loc[common_idx]
    )

    if is_coint:
        logger.info(f"✓ 配對協整 (p-value = {pvalue:.4f})")
    else:
        logger.warning(f"⚠ 配對不協整 (p-value = {pvalue:.4f}), 但繼續測試...")

    # 3. 預計算因子
    logger.info("步驟 3/6: 預計算卡爾曼濾波器因子...")
    factor_cache = {}
    kalman_factor = KalmanPairsFactor()

    for delta in PARAM_GRID['delta']:
        for Ve in PARAM_GRID['Ve']:
            factor_df = kalman_factor.calculate(
                price_data_cache,
                pair=SPECIFIED_PAIR,
                delta=delta,
                Ve=Ve
            )
            factor_cache[(delta, Ve)] = factor_df
            logger.debug(f"因子計算完成: delta={delta}, Ve={Ve}")

    logger.info(f"因子緩存大小: {len(factor_cache)}")

    # 4. 生成參數組合
    logger.info("步驟 4/6: 生成參數組合...")
    import itertools

    param_combinations = list(itertools.product(
        PARAM_GRID['delta'],
        PARAM_GRID['Ve'],
        PARAM_GRID['entry_threshold'],
        PARAM_GRID['exit_threshold'],
        PARAM_GRID['leverage'],
        PARAM_GRID['rebalance_freq_days'],
        PARAM_GRID['holding_days']
    ))

    total_combinations = len(param_combinations)
    logger.info(f"總組合數: {total_combinations}")

    # 決定使用網格搜索還是隨機搜索
    if total_combinations <= N_TRIALS:
        logger.info(f"使用網格搜索 (所有組合)")
        param_list = [
            {
                'delta': c[0], 'Ve': c[1],
                'entry_threshold': c[2], 'exit_threshold': c[3],
                'leverage': c[4],
                'rebalance_freq_days': c[5], 'holding_days': c[6]
            }
            for c in param_combinations
        ]
    else:
        logger.info(f"使用隨機搜索 ({N_TRIALS} 次)")
        import random
        sampled = random.sample(param_combinations, N_TRIALS)
        param_list = [
            {
                'delta': c[0], 'Ve': c[1],
                'entry_threshold': c[2], 'exit_threshold': c[3],
                'leverage': c[4],
                'rebalance_freq_days': c[5], 'holding_days': c[6]
            }
            for c in sampled
        ]

    # 5. 準備共享數據並保存到磁盤
    logger.info("步驟 5/6: 準備多進程環境...")
    cache_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    cache_file.close()

    with open(cache_file.name, 'wb') as f:
        pickle.dump({
            'price_data_cache': price_data_cache,
            'factor_cache': factor_cache
        }, f)

    shared_data = {
        'start_date': START_DATE,
        'end_date': END_DATE,
        'commission_bps': COMMISSION_BPS,
        'stop_loss_commission_bps': STOP_LOSS_COMMISSION_BPS,
        'cache_file_path': cache_file.name
    }

    # 6. 多進程評估
    logger.info(f"步驟 6/6: 多進程評估 ({len(param_list)} 個參數組合)...")
    results_list = []

    with ProcessPoolExecutor(
        max_workers=os.cpu_count(),
        initializer=init_worker,
        initargs=(shared_data,)
    ) as executor:
        futures = [
            executor.submit(evaluate_single_params, params)
            for params in param_list
        ]

        from tqdm import tqdm
        for future in tqdm(futures, desc="評估進度"):
            try:
                result = future.result(timeout=60)
                results_list.append(result)
            except Exception as e:
                logger.error(f"Future 失敗: {e}")

    # 清理臨時文件
    try:
        os.unlink(cache_file.name)
    except:
        pass

    # 7. 分析結果
    logger.info("分析優化結果...")
    results_df = pd.DataFrame(results_list)

    # 過濾有效結果
    valid_results = results_df[results_df['sharpe_ratio'] > -900].copy()
    logger.info(f"有效結果: {len(valid_results)} / {len(results_df)}")

    if valid_results.empty:
        logger.error("沒有有效結果,優化失敗")
        return

    # 魯棒性過濾
    logger.info("應用魯棒性過濾...")
    filtered = valid_results[
        (valid_results['sharpe_ratio'] >= SHARPE_THRESHOLD) &
        (valid_results['turnover'] >= TURNOVER_MIN) &
        (valid_results['turnover'] < TURNOVER_MAX)
    ].copy()

    logger.info(f"通過篩選: {len(filtered)} 個參數組合")

    if filtered.empty:
        logger.warning("無參數通過篩選,使用 Sharpe 最高的")
        best_params = valid_results.loc[valid_results['sharpe_ratio'].idxmax()].to_dict()
    else:
        # 按 Sharpe 排序,選擇最佳
        filtered = filtered.sort_values('sharpe_ratio', ascending=False)
        best_params = filtered.iloc[0].to_dict()

    logger.info(f"\n最佳參數:")
    logger.info(f"  Delta: {best_params['delta']}")
    logger.info(f"  Ve: {best_params['Ve']}")
    logger.info(f"  Entry Threshold: {best_params['entry_threshold']}")
    logger.info(f"  Exit Threshold: {best_params['exit_threshold']}")
    logger.info(f"  Leverage: {best_params['leverage']}")
    logger.info(f"  Sharpe: {best_params['sharpe_ratio']:.2f}")
    logger.info(f"  Return: {best_params['total_return']*100:.2f}%")
    logger.info(f"  Max DD: {best_params['max_drawdown']*100:.2f}%")

    # 8. 保存結果
    logger.info("保存優化結果...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"factors/kalman_pairs/results/Optimization_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 保存最佳參數
    with open(os.path.join(output_dir, "best_params.json"), 'w') as f:
        json.dump(best_params, f, indent=2)

    # 保存所有結果
    valid_results.to_csv(os.path.join(output_dir, "optimization_results.csv"), index=False)

    # 保存通過篩選的結果
    if not filtered.empty:
        filtered.to_csv(os.path.join(output_dir, "filtered_candidates.csv"), index=False)

    logger.info(f"結果已保存到: {output_dir}")
    logger.info("=== 優化完成 ===")


if __name__ == "__main__":
    main()
