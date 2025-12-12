"""
動量因子市場中性策略 - 參數優化與敏感性分析

功能:
1. 自適應搜索策略(網格搜索 vs 隨機搜索+補充鄰域)
2. 敏感性分析(夏普>1.25 + 穩健性調整後分數)
3. 生成完整報告和圖表(包含因子vs BTC權益+回撤對比)

作者: Claude Code
創建日期: 2025-10-16
"""

import sys
import json
import os
import logging
import pickle
import tempfile
import numpy as np
import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random
import warnings
warnings.filterwarnings('ignore')

# 配置 logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization_errors.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 常量定義
INVALID_SHARPE = -999.0  # 用於標識失敗的參數評估
DEFAULT_FAILED_METRICS = {
    "sharpe_ratio": INVALID_SHARPE,
    "turnover": 0.0,
    "subniverse_sharpe": INVALID_SHARPE,
    "avg_monthly_return": 0.0,
    "std_monthly_return": 0.0,
    "top_n": 0
}

from factors.cross_sectional_momentum.factor import MomentumFactor
from shared.cross_sectional_backtest import CrossSectionalBacktester
from shared.position_strategies import RotatingPositionStrategy
from shared.data_loader import load_local_data
from shared.visualization import plot_walkforward_performance

# ==================== 多進程優化 ====================

# 用於存儲每個工作進程的共享數據的全局變量
worker_cache = {}

def init_worker(shared_data: dict):
    """
    初始化工作進程。

    ⚠️ 注意: 為避免在 Windows 上的大量內存複製，price_data_cache 和 factor_cache
    不通過 shared_data 傳遞。工作進程會從預先保存的磁盤文件中加載這些大對象。

    Args:
        shared_data: 包含配置參數的字典（不包含大型 DataFrame）
            - start_date, end_date, universe_csv_path等小對象
            - cache_file_path: 預先保存的緩存文件路徑
    """
    global worker_cache
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["PYTHONWARNINGS"] = "ignore"
    worker_cache.update(shared_data)

    # 從磁盤加載大對象（避免進程間序列化複製）
    import pickle
    if 'cache_file_path' in shared_data:
        cache_path = Path(shared_data['cache_file_path'])
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                worker_cache['price_data_cache'] = cached_data.get('price_data_cache', {})
                worker_cache['factor_cache'] = cached_data.get('factor_cache', {})


# ==================== 輔助函數 ====================

def percentile_to_n(universe_size: int, percentile: int) -> int:
    """將百分位數轉換為標的數量"""
    return round(universe_size * percentile / 100)


def calculate_stop_loss(leverage: int) -> float:
    """根據槓桿倍數計算止損點位"""
    return -1.0 / leverage


def evaluate_single_params(params: dict) -> dict:
    """
    評估單個參數組合的輔助函數。
    此函數現在從 `worker_cache` 獲取共享數據。
    """
    global worker_cache
    try:
        # 解析參數
        lookback_period = int(params["lookback_period"])
        long_short_percentile = int(params["long_short_percentile"])
        leverage = int(params["leverage"])
        rebalance_freq_days = int(params["rebalance_freq_days"])
        holding_days = int(params.get("holding_days", rebalance_freq_days))

        # 從 worker_cache 獲取配置
        start_date = worker_cache["start_date"]
        end_date = worker_cache["end_date"]
        universe_csv_path = worker_cache["universe_csv_path"]
        universe_top_n = worker_cache["universe_top_n"]
        commission_bps = worker_cache["commission_bps"]
        slippage_bps = worker_cache["slippage_bps"]
        price_data_cache = worker_cache.get("price_data_cache", {})

        # 計算平均宇宙大小和標的數量
        avg_universe_size = worker_cache["avg_universe_size"]
        top_n = percentile_to_n(avg_universe_size, long_short_percentile)

        factor_params = {
            "lookback_period": lookback_period,
            "holding_days": holding_days,
            "rebalance_freq_days": rebalance_freq_days
        }

        # 創建回測器
        backtester = CrossSectionalBacktester(
            factor_calculator=None,  # 因子已預先計算
            position_strategy=RotatingPositionStrategy(),
            start_date=start_date,
            end_date=end_date,
            timeframe="1d",
            top_n=top_n,
            bottom_n=top_n,
            position_weight=None,
            rebalance_freq=f"{rebalance_freq_days}D",
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            universe_csv_path=universe_csv_path,
            universe_top_n=universe_top_n
        )
        backtester.all_data = price_data_cache

        # 從預先計算的因子緩存中獲取
        factor_cache = worker_cache.get('factor_cache', {})
        factor_df = factor_cache.get(lookback_period)

        if factor_df is None:
            raise ValueError(f"因子未預先計算: lookback_period={lookback_period}")

        # 運行回測
        results = backtester.run(factor_df=factor_df, **factor_params)

        # 提取績效指標
        perf_summary = results.get('performance_summary', {})
        sharpe_ratio = perf_summary.get('sharpe_ratio', INVALID_SHARPE)
        turnover = perf_summary.get('turnover', 0)
        subniverse_sharpe = perf_summary.get('sub_universe_sharpe', INVALID_SHARPE)
        daily_returns = perf_summary.get('daily_return', pd.Series(dtype=float))
        
        if not daily_returns.empty:
            monthly_returns = (1 + daily_returns).resample('M').prod() - 1
            avg_monthly_return = monthly_returns.mean()
            std_monthly_return = monthly_returns.std()
        else:
            avg_monthly_return = 0
            std_monthly_return = 0
            
        avg_monthly_return = 0 if pd.isna(avg_monthly_return) else avg_monthly_return
        std_monthly_return = 0 if pd.isna(std_monthly_return) else std_monthly_return

        return {
            **params,
            "sharpe_ratio": float(sharpe_ratio),
            "turnover": float(turnover),
            "subniverse_sharpe": float(subniverse_sharpe),
            "avg_monthly_return": float(avg_monthly_return),
            "std_monthly_return": float(std_monthly_return),
            "top_n": top_n
        }

    except Exception as e:
        logger.error(f"參數評估失敗 {params}: {e}", exc_info=True)
        result = {**params, **DEFAULT_FAILED_METRICS.copy()}
        return result


def generate_all_combinations(param_grid: dict) -> list:
    """生成所有參數組合"""
    import itertools
    keys = param_grid.keys()
    values = [list(v) for v in param_grid.values()]
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return combinations


def preload_price_data(universe_csv_path: str, start_date: str, end_date: str, max_lookback: int) -> tuple:
    """
    預加載所有需要的價格數據到緩存中

    Returns:
        tuple: (price_data_cache, universe_df) - 價格數據緩存和宇宙DataFrame
    """
    print("正在預加載所有價格數據...")
    universe_df = pd.read_csv(universe_csv_path)
    universe_df["timestamp"] = pd.to_datetime(universe_df["timestamp"], dayfirst=True)

    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)

    # 使用 query 進行向量化過濾
    filtered_universe = universe_df.query(
        '@start_date_dt <= timestamp <= @end_date_dt'
    )

    # 使用 stack().dropna() 進行高效的符號提取
    all_symbols = set(
        filtered_universe.iloc[:, 1:].stack().dropna().astype(str).unique()
    ) - {'', 'nan'}

    print(f"從宇宙文件中篩選出 {len(filtered_universe)} 個時間點（{start_date} 至 {end_date}）")
    print(f"共 {len(all_symbols)} 個唯一標的需要加載")

    price_data_cache = {}
    for symbol in tqdm(all_symbols, desc="加載價格數據"):
        symbol_with_suffix = f"{symbol}USDT"
        # 使用 max_lookback 來確保加載足夠的歷史數據
        df = load_local_data(
            symbol=symbol_with_suffix,
            data_source="1d",
            start_date=start_date,
            end_date=end_date,
            lookback_days=max_lookback
        )
        if df is not None and not df.empty:
            price_data_cache[symbol_with_suffix] = df

    print(f"\n成功加載 {len(price_data_cache)} 個標的的數據")
    return price_data_cache, universe_df


def run_optimization(
    param_grid: dict,
    shared_data: dict,
    n_trials: int = 300,
    n_jobs: int = -1,
    executor = None
) -> pd.DataFrame:
    """
    運行參數優化，使用 initializer 來避免數據重複傳輸。
    """
    total_combinations = np.prod([len(list(v)) for v in param_grid.values()])
    print(f"參數空間大小: {total_combinations} 個組合")
    print(f"隨機搜索次數: {n_trials}")

    if total_combinations <= n_trials:
        print("→ 使用完整網格搜索")
        param_combinations = generate_all_combinations(param_grid)
    else:
        print("→ 使用隨機搜索")
        # 預先轉換為 list 以提升性能
        param_grid_lists = {name: list(values) for name, values in param_grid.items()}

        param_combinations = []
        for _ in range(n_trials):
            params = {name: random.choice(values) for name, values in param_grid_lists.items()}
            param_combinations.append(params)

    print(f"實際評估組合數: {len(param_combinations)}\n")

    num_workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    results = []
    print("開始參數優化...")

    # 使用傳入的 executor 或創建新的
    if executor is None:
        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(shared_data,)) as new_executor:
            futures = [new_executor.submit(evaluate_single_params, params) for params in param_combinations]
            for future in tqdm(futures, desc="參數優化進度", ncols=100):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"一個參數評估失敗: {e}")
    else:
        futures = [executor.submit(evaluate_single_params, params) for params in param_combinations]
        for future in tqdm(futures, desc="參數優化進度", ncols=100):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"一個參數評估失敗: {e}")

    return pd.DataFrame(results)


# ==================== 主函數 ====================

def main():
    """主函數"""
    print("=" * 80)
    print("動量因子市場中性策略 - 參數優化與敏感性分析")
    print("=" * 80)
    print()

    # ========== 用戶配置區 ==========
    START_DATE = "2025-01-01"
    END_DATE = "2025-10-31"
    PARAM_GRID = {
        "lookback_period": range(14, 15, 1),
        "long_short_percentile": range(10, 20, 10),
        "leverage": range(1, 2, 1),
        "rebalance_freq_days": range(1, 2, 1),
        "holding_days": range(1, 2, 1)
    }
    N_TRIALS = 1000
    SHARPE_THRESHOLD = 1.25
    TURNOVER_MIN = 0.01  # 最小換手率：避免交易太少，無法有效捕捉因子信號
    TURNOVER_MAX = 0.7   # 最大換手率：避免過度交易，被交易成本侵蝕
    COMMISSION_BPS = 2
    SLIPPAGE_BPS = 5
    UNIVERSE_CSV_PATH = str(project_root / "data" / "40.csv")
    UNIVERSE_TOP_N = 40
    N_JOBS = -1
    # ========== 配置區結束 ==========

    print(f"優化期間: {START_DATE} 至 {END_DATE}")
    print(f"參數空間:")
    for name, p_range in PARAM_GRID.items():
        print(f"  {name}: {list(p_range)[:3]}...{list(p_range)[-3:]} (共 {len(list(p_range))} 個)")
    print()

    max_lookback = max(PARAM_GRID["lookback_period"]) if PARAM_GRID["lookback_period"] else 0

    # Step 0: 預加載所有價格數據
    price_data_cache, universe_df = preload_price_data(UNIVERSE_CSV_PATH, START_DATE, END_DATE, max_lookback)

    if not price_data_cache:
        print("[錯誤] 價格數據加載失敗，程序終止。")
        return

    # 從返回的 universe_df 計算平均宇宙大小（避免重複讀取 CSV）
    avg_universe_size = len([s for s in universe_df.iloc[0, 1:] if pd.notna(s) and s != ""])

    # Step 0.5: 預先計算所有 lookback_period 的因子值（關鍵優化！）
    print("\n【Step 0.5】預先計算所有 lookback_period 的因子值...")
    unique_lookbacks = sorted(set(PARAM_GRID["lookback_period"]))
    print(f"需要計算 {len(unique_lookbacks)} 個不同的 lookback 因子: {unique_lookbacks}")

    momentum_factor = MomentumFactor()
    factor_cache = {}
    for lookback in tqdm(unique_lookbacks, desc="計算因子"):
        factor_cache[lookback] = momentum_factor.calculate(
            price_data_cache, lookback_period=lookback, min_volume=0
        )
    print(f"因子預計算完成！\n")

    # Step 0.6: 保存緩存到臨時文件（避免多進程序列化開銷）
    print("【Step 0.6】保存緩存到臨時文件以優化多進程內存使用...")
    import tempfile
    import pickle
    cache_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    cache_file_path = cache_file.name
    cache_file.close()

    with open(cache_file_path, 'wb') as f:
        pickle.dump({
            'price_data_cache': price_data_cache,
            'factor_cache': factor_cache
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"緩存已保存至: {cache_file_path}\n")

    # Step 1: 準備共享數據並運行優化（不包含大對象）
    print("【Step 1】運行參數優化")
    shared_data = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "universe_csv_path": UNIVERSE_CSV_PATH,
        "universe_top_n": UNIVERSE_TOP_N,
        "commission_bps": COMMISSION_BPS,
        "slippage_bps": SLIPPAGE_BPS,
        "avg_universe_size": avg_universe_size,
        "cache_file_path": cache_file_path,  # 傳遞文件路徑而非大對象
    }

    # 創建持久化的 process pool 以便在多個階段重複使用
    num_workers = os.cpu_count() if N_JOBS == -1 else max(1, N_JOBS)
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(shared_data,)) as executor:
        results_df = run_optimization(
            param_grid=PARAM_GRID,
            shared_data=shared_data,
            n_trials=N_TRIALS,
            n_jobs=N_JOBS,
            executor=executor
        )

        print(f"優化完成,共評估 {len(results_df)} 個參數組合")
        valid_results = results_df[results_df['sharpe_ratio'] > INVALID_SHARPE].copy()
        print(f"有效結果數量: {len(valid_results)}\n")

        if valid_results.empty:
            debug_path = "debug_failed_optimization.csv"
            results_df.to_csv(debug_path)
            raise RuntimeError(
                f"優化失敗: 沒有有效的優化結果（所有參數組合的夏普比率 <= {INVALID_SHARPE}）。"
                f"調試數據已保存至 {debug_path}。請檢查數據質量和參數範圍。"
            )

    # Step 2: 按指標過濾最優參數
    print("【Step 2】按指標過濾最優參數")
    print(f"過濾條件: 夏普比率 >= {SHARPE_THRESHOLD}, 換手率 {TURNOVER_MIN} ~ {TURNOVER_MAX}")

    param_names = list(PARAM_GRID.keys())
    candidates = valid_results[
        (valid_results['sharpe_ratio'] >= SHARPE_THRESHOLD) &
        (valid_results['turnover'] >= TURNOVER_MIN) &
        (valid_results['turnover'] <= TURNOVER_MAX)
    ].copy()

    print(f"符合條件的參數組合數: {len(candidates)}")

    if candidates.empty:
        print("警告: 沒有符合所有條件的參數，放寬換手率限制...")
        print("  → 回退策略: 只保留夏普比率閾值")
        candidates = valid_results[
            valid_results['sharpe_ratio'] >= SHARPE_THRESHOLD
        ].copy()

    if candidates.empty:
        print("警告: 沒有符合夏普比率閾值的參數，使用夏普比率最高的參數")
        print("  → 回退策略: 從所有有效結果中選擇")
        candidates = valid_results.copy()

    # 選擇夏普比率最高的參數
    best_robust_params = candidates.nlargest(1, 'sharpe_ratio').iloc[0][param_names].to_dict()
    best_result = candidates.nlargest(1, 'sharpe_ratio').iloc[0]

    print(f"\n最優參數: {best_robust_params}")
    print(f"夏普比率: {best_result['sharpe_ratio']:.4f}")
    print(f"換手率: {best_result['turnover']:.4f}")
    print()

    # Step 5: 使用最佳穩健參數運行最終回測
    print("【Step 5】使用最佳穩健參數運行最終回測")
    lookback_period = int(best_robust_params["lookback_period"])
    long_short_percentile = int(best_robust_params["long_short_percentile"])
    leverage = int(best_robust_params["leverage"])
    rebalance_freq_days = int(best_robust_params["rebalance_freq_days"])
    holding_days = int(best_robust_params.get("holding_days", rebalance_freq_days))
    
    top_n = percentile_to_n(avg_universe_size, long_short_percentile)
    n_tranches = holding_days / rebalance_freq_days if rebalance_freq_days > 0 else 0

    print(f"最佳參數: lookback={lookback_period}, percentile={long_short_percentile}%, rebalance={rebalance_freq_days}D, holding={holding_days}D")
    print(f"做多/做空標的數量: {top_n}")
    print(f"資金分成批次數: {n_tranches:.1f}")

    momentum_factor = MomentumFactor()
    backtester = CrossSectionalBacktester(
        factor_calculator=momentum_factor,
        position_strategy=RotatingPositionStrategy(),
        start_date=START_DATE,
        end_date=END_DATE,
        timeframe="1d",
        top_n=top_n,
        bottom_n=top_n,
        position_weight=None,
        rebalance_freq=f"{rebalance_freq_days}D",
        commission_bps=COMMISSION_BPS,
        slippage_bps=SLIPPAGE_BPS,
        universe_csv_path=UNIVERSE_CSV_PATH,
        universe_top_n=UNIVERSE_TOP_N
    )
    backtester.all_data = price_data_cache

    print("\n使用預先計算的因子值...")
    factor_df = factor_cache.get(lookback_period)
    if factor_df is None:
        # Fallback: 如果緩存中沒有（不應該發生）
        print(f"警告: 因子緩存中未找到 lookback={lookback_period}，重新計算...")
        factor_df = momentum_factor.calculate(price_data_cache, lookback_period=lookback_period, min_volume=0)

    print("開始運行最終回測...")
    final_results = backtester.run(
        factor_df=factor_df,
        lookback_period=lookback_period,
        holding_days=holding_days,
        rebalance_freq_days=rebalance_freq_days
    )
    
    if not final_results or 'equity_curve' not in final_results or final_results['equity_curve'].empty:
        print("警告: 最終回測未能生成有效的權益曲線!")
        return

    factor_equity = final_results['equity_curve']
    perf_summary = final_results.get('performance_summary', {})
    print("最終回測績效指標:")
    for key, value in perf_summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()

    # Step 6: 加載BTC數據作為基準
    print("【Step 6】加載BTC數據作為基準")
    first_trade_date_str = perf_summary.get('first_trade_date')
    btc_equity = None
    if first_trade_date_str:
        btc_data = load_local_data(
            symbol="BTCUSDT",
            data_source="1d",
            start_date=first_trade_date_str,
            end_date=END_DATE,
            lookback_days=0
        )
        if btc_data is not None and not btc_data.empty:
            first_trade_date = pd.to_datetime(first_trade_date_str)
            factor_equity_aligned = factor_equity[factor_equity.index >= first_trade_date]
            
            common_dates = factor_equity_aligned.index.intersection(btc_data.index)
            if not common_dates.empty:
                btc_returns = btc_data.loc[common_dates, 'close'].pct_change().fillna(0)
                initial_capital = factor_equity_aligned.loc[common_dates[0]]
                btc_equity = (1 + btc_returns).cumprod() * initial_capital
                print(f"成功加載BTC數據 (對齊至首次交易日期: {first_trade_date.date()})\n")
            else:
                print("警告: BTC和因子數據沒有重疊日期。\n")
        else:
            print("警告: 無法加載BTC數據。\n")
    else:
        print("警告: 沒有實際交易發生，跳過BTC基準對比。\n")

    # Step 7: 生成報告和圖表
    print("【Step 7】生成報告和圖表")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "results" / f"Optimization_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "optimization_results.csv", index=False)

    # 保存符合條件的候選參數
    if not candidates.empty:
        candidates.to_csv(output_dir / "filtered_candidates.csv", index=False)

    # 清理 perf_summary 中的非 JSON 可序列化對象（如 pandas Series）
    clean_perf_summary = {}
    for key, value in perf_summary.items():
        if isinstance(value, (pd.Series, pd.DataFrame)):
            # 跳過 Series 和 DataFrame 對象
            continue
        elif isinstance(value, (np.integer, np.floating)):
            clean_perf_summary[key] = float(value)
        elif isinstance(value, (int, float, str, bool, type(None))):
            clean_perf_summary[key] = value
        else:
            # 其他類型嘗試轉換為字符串
            clean_perf_summary[key] = str(value)

    best_params_summary = {
        "best_robust_params": {k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v for k, v in best_robust_params.items()},
        "performance_metrics": clean_perf_summary,
        "configuration": {
            "start_date": START_DATE, "end_date": END_DATE, "sharpe_threshold": SHARPE_THRESHOLD
        }
    }
    with open(output_dir / "best_robust_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params_summary, f, indent=4, ensure_ascii=False)

    if factor_equity is not None and not factor_equity.empty:
        plot_title = f"動量因子策略 vs BTC基準 (從 {first_trade_date_str or START_DATE} 開始)"
        plot_walkforward_performance(
            results_df=pd.DataFrame(),
            combined_equity_curve=factor_equity,
            buy_hold_equity_curve=btc_equity,
            title=plot_title,
            save_path=output_dir / "equity_and_drawdown.png"
        )

    print(f"\n所有結果已保存到: {output_dir}")
    print("=" * 80)

    # 清理臨時緩存文件
    try:
        if os.path.exists(cache_file_path):
            os.unlink(cache_file_path)
            print(f"\n臨時緩存文件已清理: {cache_file_path}")
    except Exception as e:
        logger.warning(f"清理臨時文件失敗: {e}")


if __name__ == "__main__":
    main()
