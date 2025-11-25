"""
卡爾曼配對交易 - Walk-Forward 測試

功能:
1. 動態配對發現 (每個訓練窗口自動找出協整率最高的配對)
2. 參數優化 (針對發現的配對)
3. OOS 測試 (測試期開始時重新檢驗協整性)
4. 跳過無效窗口 (無協整配對或無符合條件的參數)

使用方式:
python test_walkforward_with_sensitivity.py
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

# 添加項目根目錄
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from factors.kalman_pairs.factor import KalmanPairsFactor
from factors.kalman_pairs.position_strategy import KalmanPairsPositionStrategy
from factors.kalman_pairs.cointegration_utils import (
    find_best_cointegrated_pair,
    recheck_cointegration,
    get_universe_symbols_for_date
)
from shared.cross_sectional_backtest import CrossSectionalBacktester
from shared.data_loader import load_local_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 配置 ====================
TRAIN_DAYS = 780
TEST_DAYS = 150
STEP_DAYS = 150

START_DATE = "2022-11-01"
END_DATE = "2024-12-31"

PARAM_GRID = {
    "delta": [1e-5, 1e-4, 1e-3],
    "Ve": [0.001, 0.01],
    "entry_threshold": [1.5, 2.0, 2.5],
    "exit_threshold": [0.0, 0.5],
    "leverage": [1, 2],
    "rebalance_freq_days": [14, 21],
    "holding_days": [14, 21]
}

N_TRIALS = 100
SHARPE_THRESHOLD = 1.25
TURNOVER_MIN = 0.01
TURNOVER_MAX = 0.7

COMMISSION_BPS = 20.0
STOP_LOSS_COMMISSION_BPS = 55.0

UNIVERSE_CSV_PATH = "data/40.csv"
MIN_CORRELATION = 0.6
MAX_PVALUE = 0.05
MIN_HALFLIFE = 5
MAX_HALFLIFE = 30

# ==================== 工作進程 ====================
worker_cache = {}

def init_worker(shared_data: dict):
    global worker_cache
    os.environ["TQDM_DISABLE"] = "1"
    worker_cache.update(shared_data)
    if 'cache_file_path' in shared_data:
        cache_path = Path(shared_data['cache_file_path'])
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                worker_cache['price_data_cache'] = cached_data.get('price_data_cache', {})
                worker_cache['factor_cache'] = cached_data.get('factor_cache', {})

def evaluate_single_params(params: dict) -> dict:
    global worker_cache
    try:
        factor_cache_key = (params['delta'], params['Ve'])
        factor_df = worker_cache['factor_cache'].get(factor_cache_key)
        if factor_df is None:
            return {**params, "sharpe_ratio": -999, "turnover": 0}

        stop_loss_pct = -1.0 / params['leverage']
        backtester = CrossSectionalBacktester(
            factor_calculator=None,
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
        backtester.all_data = worker_cache['price_data_cache']
        results = backtester.run(
            factor_df=factor_df,
            entry_threshold=params['entry_threshold'],
            exit_threshold=params['exit_threshold'],
            holding_days=params['holding_days'],
            rebalance_freq_days=params['rebalance_freq_days'],
            leverage=params['leverage']
        )
        perf = results.get('performance_summary', {})
        return {
            **params,
            "sharpe_ratio": perf.get('sharpe_ratio', -999),
            "total_return": perf.get('total_return', 0),
            "max_drawdown": perf.get('max_drawdown', 0),
            "turnover": perf.get('turnover', 0)
        }
    except Exception as e:
        logger.error(f"評估失敗: {e}")
        return {**params, "sharpe_ratio": -999, "turnover": 0}

# ==================== 窗口優化 ====================
def optimize_window(window_id, train_start, train_end, price_data_cache, universe_df):
    """訓練窗口: 配對發現 + 參數優化"""
    logger.info(f"\n[窗口 {window_id}] 訓練期: {train_start} 到 {train_end}")

    # 階段1: 配對發現
    logger.info(f"[窗口 {window_id}] 階段1: 配對發現")
    best_pair, pair_metrics = find_best_cointegrated_pair(
        price_data_cache, train_start, train_end,
        MIN_CORRELATION, MAX_PVALUE, MIN_HALFLIFE, MAX_HALFLIFE
    )

    if best_pair is None:
        logger.warning(f"[窗口 {window_id}] 未找到協整配對")
        return None, None, {'status': 'skipped', 'reason': 'no_cointegrated_pairs'}

    logger.info(f"[窗口 {window_id}] 找到配對: {best_pair}, p-value={pair_metrics['pvalue']:.4f}")

    # 階段2: 預計算因子
    logger.info(f"[窗口 {window_id}] 階段2: 計算因子")
    factor_cache = {}
    kalman_factor = KalmanPairsFactor()
    for delta in PARAM_GRID['delta']:
        for Ve in PARAM_GRID['Ve']:
            factor_df = kalman_factor.calculate(price_data_cache, pair=best_pair, delta=delta, Ve=Ve)
            factor_train = factor_df.loc[train_start:train_end]
            factor_cache[(delta, Ve)] = factor_train

    # 階段3: 參數優化
    logger.info(f"[窗口 {window_id}] 階段3: 參數優化")
    import itertools, random
    param_combinations = list(itertools.product(
        PARAM_GRID['delta'], PARAM_GRID['Ve'], PARAM_GRID['entry_threshold'],
        PARAM_GRID['exit_threshold'], PARAM_GRID['leverage'],
        PARAM_GRID['rebalance_freq_days'], PARAM_GRID['holding_days']
    ))

    if len(param_combinations) > N_TRIALS:
        sampled = random.sample(param_combinations, N_TRIALS)
    else:
        sampled = param_combinations

    param_list = [
        {'delta': c[0], 'Ve': c[1], 'entry_threshold': c[2], 'exit_threshold': c[3],
         'leverage': c[4], 'rebalance_freq_days': c[5], 'holding_days': c[6]}
        for c in sampled
    ]

    # 多進程優化
    cache_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    cache_file.close()
    with open(cache_file.name, 'wb') as f:
        pickle.dump({'price_data_cache': {best_pair[0]: price_data_cache[best_pair[0]],
                                          best_pair[1]: price_data_cache[best_pair[1]]},
                     'factor_cache': factor_cache}, f)

    shared_data = {
        'start_date': train_start, 'end_date': train_end,
        'commission_bps': COMMISSION_BPS,
        'stop_loss_commission_bps': STOP_LOSS_COMMISSION_BPS,
        'cache_file_path': cache_file.name
    }

    results_list = []
    with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=init_worker, initargs=(shared_data,)) as executor:
        futures = [executor.submit(evaluate_single_params, p) for p in param_list]
        for future in futures:
            try:
                results_list.append(future.result(timeout=60))
            except:
                pass

    try:
        os.unlink(cache_file.name)
    except:
        pass

    # 階段4: 選擇最佳參數
    results_df = pd.DataFrame(results_list)
    valid = results_df[results_df['sharpe_ratio'] > -900]

    if valid.empty:
        logger.warning(f"[窗口 {window_id}] 無有效結果")
        return best_pair, None, {'status': 'skipped', 'reason': 'no_valid_results', 'pair': best_pair}

    filtered = valid[
        (valid['sharpe_ratio'] >= SHARPE_THRESHOLD) &
        (valid['turnover'] >= TURNOVER_MIN) &
        (valid['turnover'] < TURNOVER_MAX)
    ]

    if filtered.empty:
        logger.warning(f"[窗口 {window_id}] 無參數滿足條件")
        return best_pair, None, {'status': 'skipped', 'reason': 'no_valid_params', 'pair': best_pair}

    best_params = filtered.sort_values('sharpe_ratio', ascending=False).iloc[0].to_dict()
    logger.info(f"[窗口 {window_id}] IS Sharpe={best_params['sharpe_ratio']:.2f}")

    return best_pair, best_params, {'status': 'success', 'pair': best_pair, 'is_sharpe': best_params['sharpe_ratio']}

# ==================== OOS測試 ====================
def run_oos_test(window_id, test_start, test_end, train_pair, best_params, price_data_cache):
    """測試窗口: 重新檢驗協整性 + OOS 回測"""
    logger.info(f"[窗口 {window_id}] OOS測試: {test_start} 到 {test_end}")

    # 重新檢驗協整性
    asset1_data = price_data_cache[train_pair[0]].loc[:test_start]
    asset2_data = price_data_cache[train_pair[1]].loc[:test_start]
    is_coint, pvalue = recheck_cointegration(asset1_data, asset2_data, lookback_days=252)

    if not is_coint:
        logger.warning(f"[窗口 {window_id}] 配對失去協整性 (p={pvalue:.4f}), 跳過")
        flat_dates = pd.date_range(test_start, test_end, freq='D')
        return pd.Series([1.0] * len(flat_dates), index=flat_dates), {'status': 'skipped', 'reason': 'lost_cointegration'}

    logger.info(f"[窗口 {window_id}] 配對仍協整 (p={pvalue:.4f})")

    # OOS 回測
    kalman_factor = KalmanPairsFactor()
    factor_df = kalman_factor.calculate(price_data_cache, pair=train_pair, delta=best_params['delta'], Ve=best_params['Ve'])
    factor_test = factor_df.loc[test_start:test_end]

    stop_loss_pct = -1.0 / best_params['leverage']
    backtester = CrossSectionalBacktester(
        factor_calculator=None, position_strategy=KalmanPairsPositionStrategy(),
        start_date=test_start, end_date=test_end, timeframe='1d',
        rebalance_freq=f"{best_params['rebalance_freq_days']}D",
        commission_bps=COMMISSION_BPS, leverage=best_params['leverage'],
        stop_loss_pct=stop_loss_pct, stop_loss_commission_bps=STOP_LOSS_COMMISSION_BPS,
        debug_mode=False
    )
    backtester.all_data = {train_pair[0]: price_data_cache[train_pair[0]], train_pair[1]: price_data_cache[train_pair[1]]}

    results = backtester.run(factor_df=factor_test, entry_threshold=best_params['entry_threshold'],
                              exit_threshold=best_params['exit_threshold'], holding_days=best_params['holding_days'],
                              rebalance_freq_days=best_params['rebalance_freq_days'], leverage=best_params['leverage'])

    oos_equity = results['equity_curve']
    oos_metrics = results['performance_summary']

    return oos_equity, {'status': 'success', 'oos_sharpe': oos_metrics['sharpe_ratio'], 'oos_return': oos_metrics['total_return']}

# ==================== 主流程 ====================
def main():
    logger.info("=== 卡爾曼配對交易 Walk-Forward 測試開始 ===")

    # 載入宇宙
    universe_df = pd.read_csv(UNIVERSE_CSV_PATH)
    universe_df['timestamp'] = pd.to_datetime(universe_df['timestamp'], dayfirst=True)

    # 獲取所有資產
    all_symbols = set()
    for _, row in universe_df.iterrows():
        symbols = [s for s in row.iloc[1:] if pd.notna(s) and s != '']
        all_symbols.update(symbols)

    logger.info(f"載入 {len(all_symbols)} 個資產的數據...")
    price_data_cache = {}
    for symbol in all_symbols:
        try:
            df = load_local_data(symbol=f"{symbol}USDT", data_source="1d", start_date=START_DATE, end_date=END_DATE)
            if df is not None and len(df) > 100:
                price_data_cache[f"{symbol}USDT"] = df
        except:
            pass

    logger.info(f"成功載入 {len(price_data_cache)} 個資產")

    # 生成窗口
    windows = []
    current_start = pd.to_datetime(START_DATE)
    end = pd.to_datetime(END_DATE)

    while True:
        train_start = current_start
        train_end = train_start + pd.Timedelta(days=TRAIN_DAYS)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=TEST_DAYS)

        if test_end > end:
            break

        windows.append({
            'train_start': train_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'test_start': test_start.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d')
        })

        current_start += pd.Timedelta(days=STEP_DAYS)

    logger.info(f"生成 {len(windows)} 個窗口")

    # 處理每個窗口
    window_results = []
    all_oos_curves = []

    for i, window in enumerate(windows):
        # 訓練階段
        best_pair, best_params, train_status = optimize_window(
            i, window['train_start'], window['train_end'], price_data_cache, universe_df
        )

        if train_status['status'] == 'skipped':
            logger.info(f"[窗口 {i}] 跳過: {train_status['reason']}")
            flat_dates = pd.date_range(window['test_start'], window['test_end'], freq='D')
            all_oos_curves.append(pd.Series([1.0] * len(flat_dates), index=flat_dates))
            window_results.append({**window, 'window_id': i, 'status': 'skipped', 'skip_reason': train_status['reason']})
            continue

        # 測試階段
        oos_equity, test_status = run_oos_test(i, window['test_start'], window['test_end'], best_pair, best_params, price_data_cache)
        all_oos_curves.append(oos_equity)
        window_results.append({
            **window, 'window_id': i, 'pair': f"{best_pair[0]}-{best_pair[1]}",
            'is_sharpe': train_status.get('is_sharpe'), 'oos_sharpe': test_status.get('oos_sharpe'),
            **best_params, **test_status
        })

    # 合併 OOS 曲線
    logger.info("合併 OOS 權益曲線...")
    combined_oos = all_oos_curves[0].copy()
    for i in range(1, len(all_oos_curves)):
        scale_factor = combined_oos.iloc[-1]
        combined_oos = pd.concat([combined_oos, all_oos_curves[i] * scale_factor])

    # 計算統計
    oos_returns = combined_oos.pct_change().dropna()
    oos_sharpe = np.sqrt(365) * oos_returns.mean() / oos_returns.std() if oos_returns.std() > 0 else 0

    summary = {
        'total_windows': len(windows),
        'successful_windows': sum(1 for r in window_results if r.get('status') != 'skipped'),
        'total_oos_sharpe': oos_sharpe,
        'final_return': combined_oos.iloc[-1] - 1
    }

    logger.info(f"\n=== Walk-Forward 摘要 ===")
    logger.info(f"總窗口數: {summary['total_windows']}")
    logger.info(f"成功窗口: {summary['successful_windows']}")
    logger.info(f"OOS Sharpe: {summary['total_oos_sharpe']:.2f}")
    logger.info(f"最終收益: {summary['final_return']*100:.2f}%")

    # 保存結果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"factors/kalman_pairs/results/WalkForward_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "walkforward_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(window_results).to_csv(os.path.join(output_dir, "window_results.csv"), index=False)
    combined_oos.to_csv(os.path.join(output_dir, "oos_equity_curve.csv"))

    logger.info(f"結果已保存到: {output_dir}")
    logger.info("=== Walk-Forward 測試完成 ===")

if __name__ == "__main__":
    main()
