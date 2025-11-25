"Cross-Sectional Negative Volume Factor - Walk-Forward Optimization with Sensitivity Analysis"

Purpose:
1. Simulate realistic trading by periodically re-optimizing parameters on rolling windows
2. Each training window: optimize + sensitivity analysis → select best ROBUST parameters
3. Each testing window: apply best robust parameters for OOS testing
4. Assess parameter stability and OOS performance consistency over time

Key Features:
- User-configurable training/testing windows and step size
- Adaptive search strategy (grid vs random) based on parameter space size
- Sensitivity analysis with robustness scoring on each training window
- Only select parameters with Sharpe > threshold AND high robustness
- Compare OOS performance vs BTC benchmark
- Track parameter evolution across windows
""

import sys
import json
import os
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import random
from functools import partial
import glob
import logging
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from factors.negative_volume.factor import NegativeVolumeFactor
from shared.cross_sectional_backtest import CrossSectionalBacktester
from shared.sensitivity import (
    get_parameter_neighborhood,
    calculate_custom_robustness_score,
    filter_robust_params
)
from shared.visualization import plot_walkforward_performance

# =========================
# Helper Functions
# =========================

def calculate_num_workers(n_jobs: int = None) -> int:
    """
    Calculate number of parallel workers based on n_jobs parameter.

    Args:
        n_jobs: Number of jobs. None/-1 uses all CPUs, positive int uses that many,
                negative int uses (cpu_count + n_jobs)

    Returns:
        Number of workers to use
    """
    cpu_count = os.cpu_count() or 1
    if n_jobs is None or n_jobs == -1:
        return cpu_count
    elif n_jobs < 0:
        return max(1, cpu_count + n_jobs)
    else:
        return max(1, n_jobs)

def percentile_to_n(universe_size: int, percentile: int) -> int:
    """Convert percentile to actual number of stocks to hold."""
    return round(universe_size * percentile / 100)

def calculate_stop_loss(leverage: int) -> float:
    """Calculate stop loss based on leverage multiplier."""
    return -1.0 / leverage

def generate_all_combinations(param_grid: dict) -> List[dict]:
    """Generate all possible parameter combinations for grid search."""
    import itertools
    param_names = list(param_grid.keys())
    param_values = [list(v) for v in param_grid.values()]
    all_combos = list(itertools.product(*param_values))
    return [dict(zip(param_names, combo)) for combo in all_combos]

def random_sample_combinations(param_grid: dict, n_samples: int) -> List[dict]:
    """Randomly sample n parameter combinations."""
    combinations = []
    for _ in range(n_samples):
        combo = {name: random.choice(list(values)) for name, values in param_grid.items()}
        combinations.append(combo)
    return combinations

def evaluate_single_params(params, **kwargs):
    """
    Evaluate a single parameter combination.
    Returns metrics including sharpe_ratio, turnover, subniverse_sharpe.
    """
    # Suppress warnings and progress bars in child processes
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["PYTHONWARNINGS"] = "ignore"
    try:
        import logging
        logging.getLogger().setLevel(logging.WARNING)
    except Exception:
        pass

    try:
        # Extract parameters
        window = int(params["window"])
        long_short_percentile = int(params["long_short_percentile"])
        leverage = int(params["leverage"])
        rebalance_freq_days = int(params["rebalance_freq_days"])

        stop_loss = calculate_stop_loss(leverage)

        # Get universe data
        universe_csv_path = kwargs["universe_csv_path"]
        universe_df = pd.read_csv(universe_csv_path)
        universe_df["timestamp"] = pd.to_datetime(universe_df["timestamp"], dayfirst=True)

        # Filter universe to training period
        train_start = pd.to_datetime(kwargs["train_start"])
        train_end = pd.to_datetime(kwargs["train_end"])
        universe_df_train = universe_df[
            (universe_df["timestamp"] >= train_start) &
            (universe_df["timestamp"] <= train_end)
        ]

        # Calculate average universe size
        universe_sizes = []
        for _, row in universe_df_train.iterrows():
            symbols = [s for s in row[1:] if pd.notna(s) and s != ""]
            universe_sizes.append(len(symbols))
        avg_universe_size = int(np.mean(universe_sizes)) if universe_sizes else 40

        top_n = percentile_to_n(avg_universe_size, long_short_percentile)
        bottom_n = top_n  # Market-neutral strategy

        # Get price data cache
        price_data_cache = kwargs.get("price_data_cache", {})

        # Initialize negative volume factor
        negative_volume_factor = NegativeVolumeFactor()

        # Create backtester
        backtester = CrossSectionalBacktester(
            factor_calculator=negative_volume_factor,
            start_date=kwargs["train_start"],
            end_date=kwargs["train_end"],
            timeframe="1d",
            top_n=top_n,
            bottom_n=bottom_n,
            position_weight=kwargs.get("position_weight"),
            rebalance_freq=f"{rebalance_freq_days}D",
            commission_bps=kwargs.get("commission_bps", 20),
            slippage_bps=kwargs.get("slippage_bps", 5),
            universe_csv_path=universe_csv_path,
            universe_top_n=kwargs.get("universe_top_n", 40)
        )

        # Use pre-loaded data
        backtester.all_data = price_data_cache

        # Calculate factor ONCE using the full cache
        factor_df = negative_volume_factor.calculate(price_data_cache, window=window, min_volume=0)

        if factor_df is None or factor_df.empty:
             raise ValueError("Factor calculation returned empty or None dataframe.")

        # Run backtest
        results = backtester.run(factor_df=factor_df, stop_loss=stop_loss)

        # Extract performance metrics
        perf_summary = results.get('performance_summary', {})
        sharpe_ratio = perf_summary.get('sharpe_ratio', -99)
        turnover = perf_summary.get('turnover', 0)
        subniverse_sharpe = perf_summary.get('sub_universe_sharpe', -99)
        total_return = perf_summary.get('total_return', 0)
        max_drawdown = perf_summary.get('max_drawdown', 0)

        return {
            **params,
            "sharpe_ratio": float(sharpe_ratio),
            "turnover": float(turnover),
            "subniverse_sharpe": float(subniverse_sharpe),
            "total_return": float(total_return),
            "max_drawdown": float(max_drawdown),
            "top_n": top_n
        }

    except Exception:
        # Silently return failed result - logging in main process only
        return {
            **params,
            "sharpe_ratio": np.nan,
            "turnover": np.nan,
            "subniverse_sharpe": np.nan,
            "total_return": np.nan,
            "max_drawdown": np.nan,
            "top_n": 0
        }

def supplement_neighborhood_for_best(best_params, param_grid, results_df, **kwargs):
    """
    Supplement neighborhood points for the best parameter found.
    Evaluates all ±2 step neighbors that weren't already tested.
    """
    logger.info("补充最佳参数的邻域点评估")

    # Get all neighborhood points
    all_neighbors = []
    param_names = list(param_grid.keys())

    for param_name in param_names:
        neighbors = get_parameter_neighborhood(
            param_name=param_name,
            param_value=best_params[param_name],
            param_grid=param_grid,
            step_size=2
        )
        for neighbor_val in neighbors:
            neighbor_params = best_params.copy()
            neighbor_params[param_name] = neighbor_val
            all_neighbors.append(neighbor_params)

    # Remove duplicates
    unique_neighbors = []
    seen = set()
    for params in all_neighbors:
        param_tuple = tuple(params[k] for k in param_names)
        if param_tuple not in seen:
            seen.add(param_tuple)
            unique_neighbors.append(params)

    # Filter out already tested combinations
    tested_set = set()
    for _, row in results_df.iterrows():
        param_tuple = tuple(int(row[k]) for k in param_names)
        tested_set.add(param_tuple)

    to_evaluate = []
    for params in unique_neighbors:
        param_tuple = tuple(params[k] for k in param_names)
        if param_tuple not in tested_set:
            to_evaluate.append(params)

    if not to_evaluate:
        logger.info("所有邻域点已在随机搜索中评估过")
        return results_df

    logger.info(f"需要补充评估 {len(to_evaluate)} 个邻域点")

    # Evaluate neighborhood points in parallel
    num_workers = calculate_num_workers(kwargs.get("n_jobs"))
    worker = partial(evaluate_single_params, **kwargs)

    new_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, params) for params in to_evaluate]
        for future in tqdm(futures, desc="邻域点评估", ascii=True, ncols=100):
            try:
                new_results.append(future.result())
            except Exception as e:
                logger.warning(f"邻域点评估失败: {e}")

    # Merge with existing results
    new_results_df = pd.DataFrame(new_results)
    combined_df = pd.concat([results_df, new_results_df], ignore_index=True)

    logger.info(f"已补充 {len(new_results)} 个邻域点")
    return combined_df

def optimize_window_with_sensitivity(
    window_id: int,
    train_start: str,
    train_end: str,
    param_grid: dict,
    n_trials: int,
    sharpe_threshold: float,
    price_data_cache: dict,
    **kwargs
) -> Tuple[dict, pd.DataFrame, dict]:
    """
    Optimize a single training window with sensitivity analysis.

    Returns:
        best_robust_params: Best robust parameter combination
        results_df: Full optimization results
        sensitivity_report: Sensitivity analysis report
    """
    print(f"\n{'='*60}")
    print(f"优化窗口 {window_id}: {train_start} 至 {train_end}")
    print(f"{ '='*60}")

    # Determine search strategy
    total_combinations = np.prod([len(list(v)) for v in param_grid.values()])
    use_grid_search = total_combinations <= n_trials

    if use_grid_search:
        logger.info(f"使用网格搜索 (总组合数 {total_combinations} <= {n_trials})")
        param_combinations = generate_all_combinations(param_grid)
    else:
        logger.info(f"使用随机搜索 (总组合数 {total_combinations} > {n_trials})")
        param_combinations = random_sample_combinations(param_grid, n_trials)

    # Prepare kwargs for evaluation
    eval_kwargs = {
        **kwargs,
        "train_start": train_start,
        "train_end": train_end,
        "price_data_cache": price_data_cache
    }

    # Run optimization
    logger.info(f"开始参数优化 ({len(param_combinations)} 个组合)")
    num_workers = calculate_num_workers(kwargs.get("n_jobs"))
    worker = partial(evaluate_single_params, **eval_kwargs)

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, params) for params in param_combinations]
        for future in tqdm(futures, desc=f"窗口{window_id}优化", ascii=True, ncols=100):
            try:
                results.append(future.result())
            except Exception as e:
                logger.warning(f"参数评估失败: {e}")

    results_df = pd.DataFrame(results)

    # Clean results
    if results_df.empty or "sharpe_ratio" not in results_df.columns:
        logger.error("优化失败: 无有效结果")
        return {},
        pd.DataFrame(),
        {})

    results_df["sharpe_ratio"] = pd.to_numeric(results_df["sharpe_ratio"], errors="coerce")
    results_df = results_df.dropna(subset=["sharpe_ratio"])

    if results_df.empty:
        logger.error("优化失败: 所有Sharpe比率为NaN")
        return {},
        pd.DataFrame(),
        {})

    # Find initial best parameters
    best_idx = results_df["sharpe_ratio"].idxmax()
    best_params = {}
    for k in param_grid.keys():
        try:
            best_params[k] = int(float(results_df.loc[best_idx, k]))
        except Exception:
            best_params[k] = results_df.loc[best_idx, k]

    logger.info(f"初始最佳参数: {best_params}, Sharpe: {results_df.loc[best_idx, 'sharpe_ratio']:.4f}")

    # Supplement neighborhood if using random search
    if not use_grid_search:
        results_df = supplement_neighborhood_for_best(
            best_params=best_params,
            param_grid=param_grid,
            results_df=results_df,
            **eval_kwargs
        )

    # Run sensitivity analysis
    logger.info("运行敏感性分析")
    robust_params_df, best_robust_params = filter_robust_params(
        results_df=results_df,
        param_grid=param_grid,
        metric="sharpe_ratio",
        sharpe_threshold=sharpe_threshold,
        top_n=10
    )

    if robust_params_df.empty:
        logger.warning(f"[窗口{window_id}] 没有参数组合满足 Sharpe > {sharpe_threshold} 的条件，使用fallback策略")
        print(f"⚠️  警告: 没有参数组合满足 Sharpe > {sharpe_threshold} 的条件")
        print("回退到初始最佳夏普比率参数")
        # Fallback to best sharpe parameter
        best_robust_params = best_params
        sensitivity_report = {
            "warning": "No parameters met robustness criteria - FALLBACK USED",
            "robustness_score": 0,
            "fallback_used": True,
            "fallback_sharpe": float(results_df.loc[best_idx, 'sharpe_ratio'])
        }
    else:
        # The best robust params are already returned by the function
        best_robust_row = robust_params_df.iloc[0]
        sensitivity_report = {
            "best_robust_sharpe": float(best_robust_row["center_sharpe"]),
            "robustness_score": float(best_robust_row["robustness_score"]),
            "degradation_factor": float(best_robust_row.get("degradation_factor", 0)),
            "volatility_factor": float(best_robust_row.get("volatility_factor", 0)),
            "n_robust_candidates": len(robust_params_df),
            "fallback_used": False
        }

        logger.info(f"最佳稳健参数: {best_robust_params}, 稳健性得分: {sensitivity_report['robustness_score']:.4f}, Sharpe: {sensitivity_report['best_robust_sharpe']:.4f}")

    return best_robust_params, results_df, sensitivity_report

def run_oos_test(
    window_id: int,
    test_start: str,
    test_end: str,
    best_params: dict,
    param_grid: dict,
    price_data_cache: dict,
    **kwargs
) -> Tuple[pd.Series, dict]:
    """
    Run out-of-sample test with best robust parameters.

    Returns:
        equity_curve: OOS equity curve
        oos_metrics: OOS performance metrics
    """
    logger.info(f"运行OOS测试 (窗口{window_id}): {test_start} 至 {test_end}")
    test_start_dt = pd.to_datetime(test_start)

    try:
        # Extract parameters
        window = int(best_params["window"])
        long_short_percentile = int(best_params["long_short_percentile"])
        leverage = int(best_params["leverage"])
        rebalance_freq_days = int(best_params["rebalance_freq_days"])

        stop_loss = calculate_stop_loss(leverage)

        # Get universe data
        universe_csv_path = kwargs["universe_csv_path"]
        universe_df = pd.read_csv(universe_csv_path)
        universe_df["timestamp"] = pd.to_datetime(universe_df["timestamp"], dayfirst=True)

        # Calculate average universe size for test period
        universe_df_test = universe_df[
            (universe_df["timestamp"] >= pd.to_datetime(test_start)) &
            (universe_df["timestamp"] <= pd.to_datetime(test_end))
        ]
        universe_sizes = []
        for _, row in universe_df_test.iterrows():
            symbols = [s for s in row[1:] if pd.notna(s) and s != ""]
            universe_sizes.append(len(symbols))
        avg_universe_size = int(np.mean(universe_sizes)) if universe_sizes else 40

        top_n = percentile_to_n(avg_universe_size, long_short_percentile)
        bottom_n = top_n

        # Calculate negative volume factor
        negative_volume_factor = NegativeVolumeFactor()

        # Create backtester
        backtester = CrossSectionalBacktester(
            factor_calculator=negative_volume_factor,
            start_date=test_start,
            end_date=test_end,
            timeframe="1d",
            top_n=top_n,
            bottom_n=bottom_n,
            position_weight=kwargs.get("position_weight"),
            rebalance_freq=f"{rebalance_freq_days}D",
            commission_bps=kwargs.get("commission_bps", 20),
            slippage_bps=kwargs.get("slippage_bps", 5),
            universe_csv_path=universe_csv_path,
            universe_top_n=kwargs.get("universe_top_n", 40)
        )

        # Use pre-loaded data
        backtester.all_data = price_data_cache

        # Calculate factor ONCE using the full cache
        factor_df = negative_volume_factor.calculate(price_data_cache, window=window, min_volume=0)

        if factor_df is None or factor_df.empty:
            raise ValueError("Factor calculation returned empty or None dataframe for OOS test.")

        logger.debug(f"[窗口{window_id}] Factor shape: {factor_df.shape}, columns: {factor_df.columns.tolist()[:5]}...")
        logger.debug(f"[窗口{window_id}] Factor date range: {factor_df.index.min()} to {factor_df.index.max()}")

        # Run backtest
        results = backtester.run(factor_df=factor_df, stop_loss=stop_loss)
        equity_curve = results["equity_curve"]

        logger.debug(f"[窗口{window_id}] Equity curve length: {len(equity_curve)}, final value: {equity_curve.iloc[-1]:.4f}")

        # Extract performance metrics
        perf_summary = results.get('performance_summary', {})
        oos_metrics = {
            "sharpe_ratio": float(perf_summary.get('sharpe_ratio', 0)),
            "total_return": float(perf_summary.get('total_return', 0)),
            "max_drawdown": float(perf_summary.get('max_drawdown', 0)),
            "turnover": float(perf_summary.get('turnover', 0)),
            "subniverse_sharpe": float(perf_summary.get('sub_universe_sharpe', 0))
        }

        logger.info(f"OOS结果 - Sharpe: {oos_metrics['sharpe_ratio']:.4f}, 总回报: {oos_metrics['total_return']*100:.2f}%")

    except Exception as e:
        logger.error(f"OOS测试失败 (窗口 {window_id}): {e}")
        equity_curve = pd.Series([1.0], index=[test_start_dt])
        oos_metrics = {
            "sharpe_ratio": 0,
            "total_return": 0,
            "max_drawdown": 0,
            "turnover": 0,
            "subniverse_sharpe": 0
        }

    return equity_curve, oos_metrics

def load_btc_benchmark(start_date: str, end_date: str, data_dir: Path) -> pd.Series:
    """Load BTC price data as benchmark and calculate equity curve."""
    logger.info("加载BTC基准数据")

    pattern = str(data_dir / "BTCUSDT_1d_*.csv")
    files = glob.glob(pattern)

    if not files:
        logger.warning("未找到BTC数据文件")
        return pd.Series([1.0], index=[pd.to_datetime(start_date)])

    try:
        df = pd.read_csv(files[0])
        for col in ["date", "timestamp", "datetime", "Date", "DATE"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break

        # Filter to date range
        df = df.loc[start_date:end_date]

        # Calculate returns and equity curve
        returns = df["close"].pct_change().fillna(0)
        equity_curve = (1 + returns).cumprod()

        logger.info(f"成功加载BTC数据 ({len(equity_curve)} 个数据点)")
        return equity_curve

    except Exception as e:
        logger.error(f"加载BTC数据失败: {e}")
        return pd.Series([1.0], index=[pd.to_datetime(start_date)])

# =========================
# Main Walk-Forward Function
# =========================

def run_walkforward_with_sensitivity(
    param_grid: dict,
    start_date: str,
    end_date: str,
    train_days: int,
    test_days: int,
    step_days: int,
    universe_csv_path: str,
    universe_top_n: int,
    position_weight: float,
    commission_bps: float,
    slippage_bps: float,
    sharpe_threshold: float = 1.25,
    n_trials: int = 200,
    n_jobs: int = -1
):
    """
    Run walk-forward optimization with sensitivity analysis.

    Args:
        param_grid: Parameter grid for optimization
        start_date: Overall start date
        end_date: Overall end date
        train_days: Training window size (days)
        test_days: Testing window size (days)
        step_days: Step size for rolling windows (days)
        universe_csv_path: Path to universe CSV file
        universe_top_n: Number of assets in universe
        position_weight: Position weighting (None for equal weight)
        commission_bps: Commission in basis points
        slippage_bps: Slippage in basis points
        sharpe_threshold: Minimum Sharpe ratio for robustness filtering
        n_trials: Number of random trials for optimization
        n_jobs: Number of parallel jobs (-1 for all CPUs)
    """
    print("="*80)
    print("Cross-Sectional Walk-Forward Optimization with Sensitivity Analysis for Negative Volume Factor")
    print("="*80)
    print(f"\n配置:")
    print(f"  时间范围: {start_date} 至 {end_date}")
    print(f"  训练窗口: {train_days} 天")
    print(f"  测试窗口: {test_days} 天")
    print(f"  步长: {step_days} 天")
    print(f"  Sharpe阈值: {sharpe_threshold}")
    print(f"  随机试验数: {n_trials}")
    print(f"  并行任务数: {n_jobs}")

    # Pre-load all price data
    print("\n正在预加载价格数据...")
    universe_df = pd.read_csv(universe_csv_path)
    universe_df["timestamp"] = pd.to_datetime(universe_df["timestamp"], dayfirst=True)

    all_symbols = set()
    for _, row in universe_df.iterrows():
        all_symbols.update([s for s in row[1:] if pd.notna(s) and s != ""])

    data_dir = Path(universe_csv_path).parent
    price_data_cache = {}

    for symbol in tqdm(all_symbols, desc="加载价格数据"):
        pattern = str(data_dir / f"{symbol}USDT_1d_*.csv")
        files = glob.glob(pattern)
        if files:
            try:
                df = pd.read_csv(files[0])
                for col in ["date", "timestamp", "datetime", "Date", "DATE"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        df = df.set_index(col).sort_index()
                        break
                # Store with USDT suffix to match universe format
                price_data_cache[f"{symbol}USDT"] = df
            except Exception as e:
                logger.warning(f"读取 {files[0]} 失败: {e}")

    print(f"成功加载 {len(price_data_cache)} 个标的价格数据")

    # Validate data completeness
    # Note: price_data_cache keys have USDT suffix, but all_symbols don't
    expected_keys = {f"{s}USDT" for s in all_symbols}
    missing_symbols = expected_keys - set(price_data_cache.keys())
    if missing_symbols:
        logger.warning(f"[数据完整性检查] 缺失 {len(missing_symbols)} 个标的数据: {missing_symbols}")
        print(f"\n[警告] Universe中有 {len(missing_symbols)} 个标的缺少价格数据")
        print(f"缺失的标的: {sorted(list(missing_symbols))[:10]}{'...' if len(missing_symbols) > 10 else ''}")
    else:
        print(f"\n[数据完整性检查] ✓ 所有 {len(all_symbols)} 个标的数据已完整加载")

    # Generate walk-forward windows
    print("\n生成Walk-Forward窗口...")
    windows = []
    current_date = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)

    while current_date + timedelta(days=train_days + test_days) <= end_date_dt:
        train_start = current_date
        train_end = train_start + timedelta(days=train_days - 1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_days - 1)

        windows.append({
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d")
        })

        current_date += timedelta(days=step_days)

    # Add final partial window if space remains
    if current_date + timedelta(days=train_days) < end_date_dt:
        train_start = current_date
        train_end = train_start + timedelta(days=train_days - 1)
        test_start = train_end + timedelta(days=1)

        if test_start <= end_date_dt:
            windows.append({
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": end_date
            })

    print(f"生成 {len(windows)} 个Walk-Forward窗口")

    # Prepare kwargs for optimization/testing
    kwargs = {
        "universe_csv_path": universe_csv_path,
        "universe_top_n": universe_top_n,
        "position_weight": position_weight,
        "commission_bps": commission_bps,
        "slippage_bps": slippage_bps,
        "n_jobs": n_jobs
    }

    # Run walk-forward analysis
    all_oos_curves = []
    window_results = []
    param_evolution = []

    for i, window in enumerate(windows):
        # Optimize training window with sensitivity analysis
        best_robust_params, opt_results_df, sensitivity_report = optimize_window_with_sensitivity(
            window_id=i,
            train_start=window["train_start"],
            train_end=window["train_end"],
            param_grid=param_grid,
            n_trials=n_trials,
            sharpe_threshold=sharpe_threshold,
            price_data_cache=price_data_cache,
            **kwargs
        )

        if not best_robust_params:
            logger.warning(f"窗口 {i} 优化失败，跳过")
            continue

        # Log OOS test configuration
        logger.debug(f"窗口 {i} OOS测试配置 - 期间: {window['test_start']} 至 {window['test_end']}, 参数: {best_robust_params}")

        # Run OOS test
        oos_equity, oos_metrics = run_oos_test(
            window_id=i,
            test_start=window["test_start"],
            test_end=window["test_end"],
            best_params=best_robust_params,
            param_grid=param_grid,
            price_data_cache=price_data_cache,
            **kwargs
        )

        all_oos_curves.append(oos_equity)

        # Record results
        window_results.append({
            "window_id": i,
            "train_period": f"{window['train_start']} to {window['train_end']}",
            "test_period": f"{window['test_start']} to {window['test_end']}",
            "is_sharpe": sensitivity_report.get("best_robust_sharpe", sensitivity_report.get("fallback_sharpe", 0)),
            "oos_sharpe": oos_metrics["sharpe_ratio"],
            "oos_return": oos_metrics["total_return"],
            "oos_max_dd": oos_metrics["max_drawdown"],
            "robustness_score": sensitivity_report.get("robustness_score", 0),
            "fallback_used": sensitivity_report.get("fallback_used", False)
        })

        param_evolution.append({
            "window_id": i,
            "test_start": window["test_start"],
            **best_robust_params,
            "is_sharpe": sensitivity_report.get("best_robust_sharpe", 0),
            "oos_sharpe": oos_metrics["sharpe_ratio"]
        })

    # Combine OOS equity curves
    logger.info("合并OOS权益曲线")
    combined_oos_equity = pd.Series(dtype=float)

    if all_oos_curves:
        for curve in all_oos_curves:
            if combined_oos_equity.empty:
                combined_oos_equity = curve
            else:
                # Scale next curve to continue from last point
                scaling_factor = combined_oos_equity.iloc[-1] / curve.iloc[0]
                scaled_curve = curve * scaling_factor
                # Concatenate by appending the new scaled curve without its first point
                # to avoid duplicating the junction point. This is robust.
                to_append = scaled_curve.iloc[1:]
                combined_oos_equity = pd.concat([combined_oos_equity, to_append])
    
    # Prepend the initial capital of 1.0 to the beginning of the equity curve
    if not combined_oos_equity.empty:
        first_day = combined_oos_equity.index[0]
        # Insert the initial capital point on the day before the first trade
        initial_point_date = first_day - pd.Timedelta(days=1)
        initial_point = pd.Series([1.0], index=[initial_point_date])
        combined_oos_equity = pd.concat([initial_point, combined_oos_equity])

    # Load BTC benchmark - start from first OOS window
    if windows:
        first_oos_start = windows[0]["test_start"]
        logger.info(f"BTC基准数据从第一个OOS窗口开始: {first_oos_start}")
    else:
        first_oos_start = start_date
        logger.warning(f"未生成Walk-Forward窗口，BTC基准使用完整起始日期: {start_date}")

    # Align to actual strategy equity date range if available
    if not combined_oos_equity.empty:
        actual_start = combined_oos_equity.index[0].strftime("%Y-%m-%d")
        actual_end = combined_oos_equity.index[-1].strftime("%Y-%m-%d")
        logger.info(f"BTC数据对齐到策略实际范围: {actual_start} ~ {actual_end}")
        btc_equity = load_btc_benchmark(actual_start, actual_end, data_dir)
    else:
        logger.warning("OOS权益曲线为空，使用预期日期范围")
        btc_equity = load_btc_benchmark(first_oos_start, end_date, data_dir)

    # Calculate summary statistics
    results_df = pd.DataFrame(window_results)
    param_evolution_df = pd.DataFrame(param_evolution)

    # Calculate Sharpe Ratio for the entire OOS period
    total_oos_sharpe = 0
    if not combined_oos_equity.empty:
        oos_daily_returns = combined_oos_equity.pct_change().fillna(0)
        if oos_daily_returns.std() != 0:
            # Annualize Sharpe Ratio (assuming 252 trading days)
            total_oos_sharpe = np.sqrt(252) * oos_daily_returns.mean() / oos_daily_returns.std()

    if not results_df.empty:
        fallback_count = int(results_df["fallback_used"].sum())
        summary = {
            "total_windows": len(windows),
            "successful_windows": len(results_df),
            "fallback_windows": fallback_count,
            "fallback_pct": float(fallback_count / len(results_df) * 100) if len(results_df) > 0 else 0,
            "total_oos_sharpe": float(total_oos_sharpe),
            "avg_oos_sharpe": float(results_df["oos_sharpe"].mean()),
            "median_oos_sharpe": float(results_df["oos_sharpe"].median()),
            "oos_sharpe_std": float(results_df["oos_sharpe"].std()),
            "avg_is_sharpe": float(results_df["is_sharpe"].mean()),
            "is_oos_ratio": float(results_df["oos_sharpe"].mean() / results_df["is_sharpe"].mean()) if results_df["is_sharpe"].mean() != 0 else 0,
            "consistency_pct": float((results_df["oos_sharpe"] > 0).sum() / len(results_df) * 100),
            "avg_robustness_score": float(results_df["robustness_score"].mean()),
            "final_oos_return": float(combined_oos_equity.iloc[-1] - 1) if not combined_oos_equity.empty else 0
        }
    else:
        summary = {"error": "No successful windows"}

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "results" / f"WalkForward_Sensitivity_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    with open(output_dir / "walkforward_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    # Save window results
    results_df.to_csv(output_dir / "window_results.csv", index=False)

    # Save parameter evolution
    param_evolution_df.to_csv(output_dir / "window_params_evolution.csv", index=False)

    # Save OOS equity curve
    combined_oos_equity.to_csv(output_dir / "oos_equity_curve.csv", header=["equity"])

    # Generate charts
    logger.info("生成图表")

    # Chart 1: OOS Equity + Drawdown vs BTC
    try:
        plot_walkforward_performance(
            results_df=results_df,
            title="Walk-Forward OOS Performance vs BTC Benchmark (Negative Volume Factor)",
            save_path=output_dir / "oos_equity_and_drawdown.png",
            figsize=(14, 10),
            combined_equity_curve=combined_oos_equity,
            buy_hold_equity_curve=btc_equity
        )
        logger.info("已保存: oos_equity_and_drawdown.png")
    except Exception as e:
        logger.error(f"生成OOS图表失败: {e}")

    # Chart 2: IS vs OOS Sharpe comparison
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        x = results_df["window_id"]
        ax.plot(x, results_df["is_sharpe"], marker="o", label="In-Sample Sharpe", linewidth=2)
        ax.plot(x, results_df["oos_sharpe"], marker="s", label="Out-of-Sample Sharpe", linewidth=2)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Window ID", fontsize=12)
        ax.set_ylabel("Sharpe Ratio", fontsize=12)
        ax.set_title("In-Sample vs Out-of-Sample Sharpe Ratio (Negative Volume Factor)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "is_oos_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("已保存: is_oos_comparison.png")
    except Exception as e:
        logger.error(f"生成IS/OOS对比图失败: {e}")

    # Print summary
    print("\n" + "="*80)
    print("Walk-Forward 分析总结")
    print("="*80)
    print(f"总窗口数: {summary.get('total_windows', 0)}")
    print(f"成功窗口数: {summary.get('successful_windows', 0)}")
    print(f"Fallback窗口数: {summary.get('fallback_windows', 0)} ({summary.get('fallback_pct', 0):.1f}%)")
    if summary.get('fallback_windows', 0) > 0:
        print(f"  ⚠️ 警告: {summary.get('fallback_windows', 0)} 个窗口使用了fallback参数（未满足稳健性标准）")
    print(f"平均OOS Sharpe: {summary.get('avg_oos_sharpe', 0):.4f}")
    print(f"中位数OOS Sharpe: {summary.get('median_oos_sharpe', 0):.4f}")
    print(f"OOS Sharpe标准差: {summary.get('oos_sharpe_std', 0):.4f}")
    print(f"平均IS Sharpe: {summary.get('avg_is_sharpe', 0):.4f}")
    print(f"IS/OOS比率: {summary.get('is_oos_ratio', 0):.4f}")
    print(f"一致性 (OOS>0): {summary.get('consistency_pct', 0):.2f}%")
    print(f"平均稳健性得分: {summary.get('avg_robustness_score', 0):.4f}")
    print(f"最终OOS总回报: {summary.get('final_oos_return', 0)*100:.2f}%")
    print(f"整体OOS Sharpe比率: {summary.get('total_oos_sharpe', 0):.4f}")
    print(f"\n结果保存至: {output_dir}")
    print("="*80)

    return {
        "summary": summary,
        "results_df": results_df,
        "param_evolution_df": param_evolution_df,
        "combined_oos_equity": combined_oos_equity,
        "output_dir": output_dir
    }

# =========================
# User Configuration Section
# =========================

if __name__ == "__main__":
    # --- Parameter Grid (all use range()) ---
    PARAM_GRID = {
        "window": range(77, 106, 1),        # Volume lookback window
        "long_short_percentile": range(20, 30, 10),     # Top/bottom percentile: 10%-40%, step 5%
        "leverage": range(1, 2, 1),                   # Leverage: 1x
        "rebalance_freq_days": range(1, 31, 1)         # Rebalance frequency: 7-42 days, step 7 (ensures 2-13 rebalances in 90-day test window)
    }

    # --- Walk-Forward Configuration ---
    TRAIN_DAYS = 780          # Training window: 780 days (~2 years)
    TEST_DAYS = 30           # Testing window: 30 days (~1 month)
    STEP_DAYS = 30            # Step size: 30 days (rolling forward)

    # --- Time Period ---
    START_DATE = "2022-11-01"
    END_DATE = "2025-10-31"

    # --- Universe Configuration ---
    UNIVERSE_CSV_PATH = str(project_root / "data" / "40.csv")
    UNIVERSE_TOP_N = 40

    # --- Backtest Configuration ---
    POSITION_WEIGHT = None    # Equal weighting
    COMMISSION_BPS = 20       # 0.2% commission
    SLIPPAGE_BPS = 5          # 0.05% slippage

    # --- Optimization Configuration ---
    SHARPE_THRESHOLD = 1.25   # Minimum Sharpe for robustness filtering
    N_TRIALS = 1000            # Random search trials per window
    N_JOBS = 3               # Use all CPU cores

    # --- Run Walk-Forward Analysis ---
    run_walkforward_with_sensitivity(
        param_grid=PARAM_GRID,
        start_date=START_DATE,
        end_date=END_DATE,
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
        step_days=STEP_DAYS,
        universe_csv_path=UNIVERSE_CSV_PATH,
        universe_top_n=UNIVERSE_TOP_N,
        position_weight=POSITION_WEIGHT,
        commission_bps=COMMISSION_BPS,
        slippage_bps=SLIPPAGE_BPS,
        sharpe_threshold=SHARPE_THRESHOLD,
        n_trials=N_TRIALS,
        n_jobs=N_JOBS
    )