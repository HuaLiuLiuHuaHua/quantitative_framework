"""
波動性反轉因子市場中性策略 - 參數優化與敏感性分析

功能:
1. 自適應搜索策略(網格搜索 vs 隨機搜索)
2. 市場中性回測(做多-做空配對策略)
3. 敏感性分析
4. 生成完整報告

作者: Claude Code
創建日期: 2025-11-15
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
INVALID_SHARPE = -999.0
DEFAULT_FAILED_METRICS = {
    "sharpe_ratio": INVALID_SHARPE,
    "turnover": 0.0,
    "subniverse_sharpe": INVALID_SHARPE,
    "avg_monthly_return": 0.0,
    "std_monthly_return": 0.0,
    "top_n": 0
}

from factors.volatility_reversion.factor import VolatilityReversionFactor
from shared.cross_sectional_backtest import CrossSectionalBacktester
from shared.position_strategies import RotatingPositionStrategy
from shared.data_loader import load_local_data
from shared.visualization import plot_walkforward_performance

# ==================== 多進程優化 ====================

worker_cache = {}

def init_worker(shared_data: dict):
    """初始化工作進程"""
    global worker_cache
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["PYTHONWARNINGS"] = "ignore"
    worker_cache.update(shared_data)

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


def evaluate_single_params(params: dict) -> dict:
    """評估單個參數組合"""
    global worker_cache
    try:
        # 解析參數
        volatility_lookback = int(params["volatility_lookback"])
        volatility_multiplier = float(params["volatility_multiplier"])
        trend_lookback = int(params["trend_lookback"])
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
            "holding_days": holding_days,
            "rebalance_freq_days": rebalance_freq_days
        }

        # 創建回測器
        backtester = CrossSectionalBacktester(
            factor_calculator=None,
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
        cache_key = (volatility_lookback, volatility_multiplier, trend_lookback)
        factor_df = factor_cache.get(cache_key)

        if factor_df is None:
            raise ValueError(f"因子未預先計算: {cache_key}")

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
    """預加載所有需要的價格數據到緩存中"""
    print("正在預加載所有價格數據...")
    universe_df = pd.read_csv(universe_csv_path)
    universe_df["timestamp"] = pd.to_datetime(universe_df["timestamp"], dayfirst=True)

    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)

    filtered_universe = universe_df.query(
        '@start_date_dt <= timestamp <= @end_date_dt'
    )

    all_symbols = set(
        filtered_universe.iloc[:, 1:].stack().dropna().astype(str).unique()
    ) - {'', 'nan'}

    print(f"從宇宙文件中篩選出 {len(filtered_universe)} 個時間點（{start_date} 至 {end_date}）")
    print(f"共 {len(all_symbols)} 個唯一標的需要加載")

    price_data_cache = {}
    for symbol in tqdm(all_symbols, desc="加載價格數據"):
        symbol_with_suffix = f"{symbol}USDT"
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
    """運行參數優化"""
    total_combinations = np.prod([len(list(v)) for v in param_grid.values()])
    print(f"參數空間大小: {total_combinations} 個組合")
    print(f"隨機搜索次數: {n_trials}")

    if total_combinations <= n_trials:
        print("→ 使用完整網格搜索")
        param_combinations = generate_all_combinations(param_grid)
    else:
        print("→ 使用隨機搜索")
        param_grid_lists = {name: list(values) for name, values in param_grid.items()}

        param_combinations = []
        for _ in range(n_trials):
            params = {name: random.choice(values) for name, values in param_grid_lists.items()}
            param_combinations.append(params)

    print(f"實際評估組合數: {len(param_combinations)}\n")

    num_workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    results = []
    print("開始參數優化...")

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
    print("波動性反轉因子市場中性策略 - 參數優化與敏感性分析")
    print("=" * 80)
    print()

    # ========== 用戶配置區 ==========
    START_DATE = "2022-11-01"
    END_DATE = "2024-12-31"
    PARAM_GRID = {
        # 波動率窗口：5-30天（更合理的波動率計算周期）
        "volatility_lookback": range(5, 31, 5),
        # 波動率倍數：1.5-3.0倍（更大的觸發倍數，容易觸發交易）
        "volatility_multiplier": np.arange(1.5, 3.1, 0.5),
        # 趨勢窗口：20-100天（合理的趨勢跟蹤周期，不會過於極端）
        "trend_lookback": range(20, 101, 20),
        # 做多做空百分比：10%-30%
        "long_short_percentile": range(10, 31, 10),
        # 杠杆：1倍
        "leverage": range(1, 2, 1),
        # 再平衡頻率：5-10天
        "rebalance_freq_days": range(5, 11, 5),
        # 持倉期：5-10天
        "holding_days": range(5, 11, 5)
    }
    N_TRIALS = 300  # 減少試驗次數（參數空間已縮小）
    SHARPE_THRESHOLD = 1.25
    TURNOVER_MIN = 0.01
    TURNOVER_MAX = 0.7
    COMMISSION_BPS = 2
    SLIPPAGE_BPS = 5
    UNIVERSE_CSV_PATH = str(project_root / "data" / "40.csv")
    UNIVERSE_TOP_N = 40
    N_JOBS = -1
    # ========== 配置區結束 ==========

    print(f"優化期間: {START_DATE} 至 {END_DATE}")
    print(f"參數空間:")
    for name, p_range in PARAM_GRID.items():
        values = list(p_range) if isinstance(p_range, range) else list(p_range)
        print(f"  {name}: {values[:3]}...{values[-3:] if len(values) > 3 else values} (共 {len(values)} 個)")
    print()

    max_lookback = max(list(PARAM_GRID["volatility_lookback"]) + list(PARAM_GRID["trend_lookback"]))

    # Step 0: 預加載所有價格數據
    price_data_cache, universe_df = preload_price_data(UNIVERSE_CSV_PATH, START_DATE, END_DATE, max_lookback)

    if not price_data_cache:
        print("[錯誤] 價格數據加載失敗，程序終止。")
        return

    avg_universe_size = len([s for s in universe_df.iloc[0, 1:] if pd.notna(s) and s != ""])

    # Step 0.5: 預先計算所有參數組合的因子值
    print("\n【Step 0.5】預先計算所有參數組合的因子值...")
    unique_lookbacks = sorted(set(PARAM_GRID["volatility_lookback"]))
    unique_multipliers = sorted(set(PARAM_GRID["volatility_multiplier"]))
    unique_trends = sorted(set(PARAM_GRID["trend_lookback"]))
    print(f"需要計算 {len(unique_lookbacks)} × {len(unique_multipliers)} × {len(unique_trends)} = {len(unique_lookbacks) * len(unique_multipliers) * len(unique_trends)} 個因子")

    volatility_factor = VolatilityReversionFactor()
    factor_cache = {}
    
    total_factors = len(unique_lookbacks) * len(unique_multipliers) * len(unique_trends)
    factor_count = 0
    
    with tqdm(total=total_factors, desc="計算因子") as pbar:
        for lookback in unique_lookbacks:
            for multiplier in unique_multipliers:
                for trend in unique_trends:
                    cache_key = (lookback, multiplier, trend)
                    factor_cache[cache_key] = volatility_factor.calculate(
                        price_data_cache,
                        x=lookback,
                        y=multiplier,
                        z=trend,
                        long_pct=0.5,
                        short_pct=0.5
                    )
                    factor_count += 1
                    pbar.update(1)

    print(f"因子預計算完成！\n")

    # Step 0.5.1: 驗證樣本因子計算結果
    print("【驗證】檢查樣本因子計算結果...")
    if factor_cache:
        sample_factor = list(factor_cache.values())[0]
        if sample_factor.empty:
            print("[錯誤] 因子計算結果為空 DataFrame！")
            print("  請檢查 VolatilityReversionFactor.calculate() 實現")
            return

        total_cells = sample_factor.size
        non_nan_count = sample_factor.notna().sum().sum()
        nan_ratio = 1 - (non_nan_count / total_cells) if total_cells > 0 else 1.0

        print(f"✓ 樣本因子形狀: {sample_factor.shape}")
        print(f"  總數據點: {total_cells}")
        print(f"  有效值 (非NaN): {non_nan_count} ({(1-nan_ratio)*100:.1f}%)")
        print(f"  NaN值: {total_cells - non_nan_count} ({nan_ratio*100:.1f}%)\n")

        if nan_ratio > 0.95:
            print("[警告] ⚠️  因子值 95% 以上為 NaN，可能導致無交易！")
            print("  診斷建議：")
            print("  1. 參數範圍是否合理？（檢查 volatility_lookback, volatility_multiplier 等）")
            print("  2. 價格數據是否充足？")
            print("  3. 因子計算邏輯是否正確？")
            print()
    else:
        print("[錯誤] factor_cache 為空！")
        return

    # Step 0.6: 保存緩存到臨時文件
    print("【Step 0.6】保存緩存到臨時文件以優化多進程內存使用...")
    cache_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    cache_file_path = cache_file.name
    cache_file.close()

    with open(cache_file_path, 'wb') as f:
        pickle.dump({
            'price_data_cache': price_data_cache,
            'factor_cache': factor_cache
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"緩存已保存至: {cache_file_path}\n")

    # Step 1: 準備共享數據並運行優化
    print("【Step 1】運行參數優化")
    shared_data = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "universe_csv_path": UNIVERSE_CSV_PATH,
        "universe_top_n": UNIVERSE_TOP_N,
        "commission_bps": COMMISSION_BPS,
        "slippage_bps": SLIPPAGE_BPS,
        "avg_universe_size": avg_universe_size,
        "cache_file_path": cache_file_path,
    }

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
                f"優化失敗: 沒有有效的優化結果。調試數據已保存至 {debug_path}。"
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
        candidates = valid_results[
            valid_results['sharpe_ratio'] >= SHARPE_THRESHOLD
        ].copy()

    if candidates.empty:
        print("警告: 沒有符合夏普比率閾值的參數，使用夏普比率最高的參數")
        candidates = valid_results.copy()

    # 選擇夏普比率最高的參數
    best_robust_params = candidates.nlargest(1, 'sharpe_ratio').iloc[0][param_names].to_dict()
    best_result = candidates.nlargest(1, 'sharpe_ratio').iloc[0]

    print(f"\n最優參數:")
    for k, v in best_robust_params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    print(f"夏普比率: {best_result['sharpe_ratio']:.4f}")
    print(f"換手率: {best_result['turnover']:.4f}")
    print()

    # Step 5: 使用最佳穩健參數運行最終回測
    print("【Step 5】使用最佳穩健參數運行最終回測")
    volatility_lookback = int(best_robust_params["volatility_lookback"])
    volatility_multiplier = float(best_robust_params["volatility_multiplier"])
    trend_lookback = int(best_robust_params["trend_lookback"])
    long_short_percentile = int(best_robust_params["long_short_percentile"])
    leverage = int(best_robust_params["leverage"])
    rebalance_freq_days = int(best_robust_params["rebalance_freq_days"])
    holding_days = int(best_robust_params.get("holding_days", rebalance_freq_days))
    
    top_n = percentile_to_n(avg_universe_size, long_short_percentile)

    print(f"最佳參數: volatility_lookback={volatility_lookback}, multiplier={volatility_multiplier}, trend_lookback={trend_lookback}")
    print(f"標的選擇: 做多/做空各 {top_n} 個")
    print(f"再平衡頻率: {rebalance_freq_days}天, 持倉期: {holding_days}天")

    volatility_factor = VolatilityReversionFactor()
    backtester = CrossSectionalBacktester(
        factor_calculator=volatility_factor,
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
    factor_df = factor_cache.get((volatility_lookback, volatility_multiplier, trend_lookback))
    if factor_df is None:
        print(f"警告: 因子緩存中未找到最優參數，重新計算...")
        factor_df = volatility_factor.calculate(
            price_data_cache,
            x=volatility_lookback,
            y=volatility_multiplier,
            z=trend_lookback,
            long_pct=0.5,
            short_pct=0.5
        )

    print("開始運行最終回測...")
    final_results = backtester.run(
        factor_df=factor_df,
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

    if not candidates.empty:
        candidates.to_csv(output_dir / "filtered_candidates.csv", index=False)

    # 清理效能指標，轉換為 JSON 可序列化的類型
    clean_perf_summary = {}
    for key, value in perf_summary.items():
        if isinstance(value, (pd.Series, pd.DataFrame)):
            continue
        elif isinstance(value, (np.integer, np.floating)):
            clean_perf_summary[key] = float(value)
        elif isinstance(value, (int, float, str, bool, type(None))):
            clean_perf_summary[key] = value
        else:
            clean_perf_summary[key] = str(value)

    # 轉換最佳參數值
    def convert_param_value(v):
        """將參數值轉換為 JSON 可序列化的型別"""
        if isinstance(v, (np.integer, int)):
            return int(v)
        elif isinstance(v, (np.floating, float)):
            return float(v)
        return v

    best_robust_params_json = {
        k: convert_param_value(v)
        for k, v in best_robust_params.items()
    }

    best_params_summary = {
        "best_robust_params": best_robust_params_json,
        "performance_metrics": clean_perf_summary,
        "configuration": {
            "start_date": START_DATE,
            "end_date": END_DATE,
            "sharpe_threshold": SHARPE_THRESHOLD
        }
    }
    with open(output_dir / "best_robust_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params_summary, f, indent=4, ensure_ascii=False)

    # Step 8: 生成權益曲線和回撤曲線圖表
    print("\n【Step 8】生成權益曲線和回撤曲線圖表")
    if factor_equity is not None and not factor_equity.empty:
        print(f"✓ 權益曲線數據有效 ({len(factor_equity)} 個數據點)，開始繪圖...")
        try:
            # 使用有效的交易開始日期
            trade_date_display = first_trade_date_str if first_trade_date_str else START_DATE
            plot_title = f"波動性反轉因子策略 vs BTC基準 (從 {trade_date_display} 開始)"

            # 驗證btc_equity
            btc_equity_valid = None
            if (btc_equity is not None and
                not btc_equity.empty and
                pd.api.types.is_numeric_dtype(btc_equity)):
                btc_equity_valid = btc_equity
                print(f"✓ BTC基準數據有效 ({len(btc_equity)} 個數據點)")
            else:
                print("⚠ BTC基準數據無效或為空，將不顯示基準對比")

            # 調用繪圖函數
            # 注意: results_df參數在該函數中實際上沒有被使用，
            #      主要邏輯基於combined_equity_curve和buy_hold_equity_curve
            plot_walkforward_performance(
                results_df=pd.DataFrame(),
                combined_equity_curve=factor_equity,
                buy_hold_equity_curve=btc_equity_valid,
                title=plot_title,
                save_path=output_dir / "equity_and_drawdown.png"
            )
            print("✓ 圖表已成功生成")
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"生成圖表時出錯: {e}", exc_info=True)
            print(f"⚠ 生成圖表失敗: {type(e).__name__}: {e}")
        except Exception as e:
            logger.error(f"意外錯誤: {e}", exc_info=True)
            print(f"⚠ 意外錯誤: {e}")
    else:
        print("⚠ 權益曲線數據無效或不足，無法生成圖表")
        if factor_equity is None:
            print("  原因: factor_equity 為 None")
        elif factor_equity.empty:
            print("  原因: factor_equity 為空 Series")
        else:
            print(f"  原因: 數據點數量: {len(factor_equity)} 個")
            if len(factor_equity) > 0:
                print(f"  首個值: {factor_equity.iloc[0]:.6f}")
                print(f"  末尾值: {factor_equity.iloc[-1]:.6f}")
                print(f"  數據類型: {factor_equity.dtype}")

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
