"""
MACD-ATR Strategy - Walk-Forward Analysis
MACD-ATR策略 - Walk-Forward分析
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np
from datetime import datetime

# 導入策略和工具
from strategies.macd_atr.strategy import MACDATRStrategy
from shared.walkforward import WalkForwardAnalyzer
from shared.visualization import plot_walkforward_performance

def macd_atr_strategy_func(data: pd.DataFrame, macd_fast: int, macd_slow: int, macd_signal: int, atr_window: int, atr_multiplier: float):
    """MACD-ATR策略包裝函數"""
    strategy = MACDATRStrategy()
    return strategy.generate_signals(
        data,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        atr_window=atr_window,
        atr_multiplier=atr_multiplier
    )

def load_data(data_config: str = "BTCUSDT_1d", start_date: str = "2022-10-01", end_date: str = "2025-09-30", verbose: bool = True):
    """
    載入數據 (優先使用本地數據)
    """
    from shared.data_loader import load_data_with_fallback
    
    try:
        symbol, timeframe = data_config.split('_')
    except ValueError:
        raise ValueError(f"data_config格式不正確,應為 'SYMBOL_TIMEFRAME', 例如 'BTCUSDT_1h', 但收到 '{data_config}'")

    fetcher_func = None
    if symbol == "BTCUSDT":
        if timeframe == "1h":
            from data_fetchers.bybit_btc_1h_fetcher import fetch_bybit_btc_1h_data
            fetcher_func = fetch_bybit_btc_1h_data
        elif timeframe == "1d":
            from data_fetchers.bybit_btc_1d_fetcher import fetch_bybit_btc_1d_data
            fetcher_func = fetch_bybit_btc_1d_data
    elif symbol == "SOLUSDT":
        if timeframe == "1h":
            from data_fetchers.bybit_sol_1h_fetcher import fetch_bybit_sol_1h_data
            fetcher_func = fetch_bybit_sol_1h_data

    if fetcher_func is None:
        raise NotImplementedError(f"未實現對 {data_config} 的數據加載")

    return load_data_with_fallback(
        data_source=timeframe,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        fetcher_func=fetcher_func,
        verbose=verbose
    )

def run_walkforward(
    data_config: str = "BTCUSDT_1d",
    macd_fast_range: tuple = (8, 16, 2),
    macd_slow_range: tuple = (20, 30, 2),
    macd_signal_range: tuple = (7, 11, 2),
    atr_window_range: tuple = (10, 20, 2),
    atr_multiplier_range: tuple = (1.5, 3.0, 0.5),
    start_date: str = "2022-10-01",
    end_date: str = "2025-09-30",
    train_window: int = 730,
    test_window: int = 30,
    step: int = 30,
    objective: str = 'profit_factor',
    transaction_cost: float = 0.002,
    slippage: float = 0.0005,
    n_jobs: int = 1,
    verbose: bool = True
):
    """
    執行Walk-Forward分析
    """
    if verbose:
        print("\n" + "=" * 70)
        print("MACD-ATR策略Walk-Forward分析")
        print("=" * 70)

    df = load_data(data_config, start_date, end_date, verbose)

    param_grid = {
        'macd_fast': list(range(macd_fast_range[0], macd_fast_range[1] + 1, macd_fast_range[2])),
        'macd_slow': list(range(macd_slow_range[0], macd_slow_range[1] + 1, macd_slow_range[2])),
        'macd_signal': list(range(macd_signal_range[0], macd_signal_range[1] + 1, macd_signal_range[2])),
        'atr_window': list(range(atr_window_range[0], atr_window_range[1] + 1, atr_window_range[2])),
        'atr_multiplier': list(np.arange(atr_multiplier_range[0], atr_multiplier_range[1] + atr_multiplier_range[2] * 0.001, atr_multiplier_range[2]))
    }

    timeframe = data_config.split('_')[1]
    periods_per_year = 365 if timeframe == "1d" else 365 * 24

    analyzer = WalkForwardAnalyzer(
        strategy_func=macd_atr_strategy_func,
        data=df,
        param_grid=param_grid,
        train_window=train_window,
        test_window=test_window,
        step=step,
        objective=objective,
        transaction_cost=transaction_cost,
        slippage=slippage,
        n_jobs=n_jobs,
        periods_per_year=periods_per_year
    )

    results_df = analyzer.run(verbose=verbose)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_source = f"WalkForward_MACD_ATR_{data_config}_{date_str}"

    output_dir = Path(__file__).parent / "results"
    save_path = analyzer.save_results(data_source=data_source, output_dir=output_dir)

    if verbose:
        print("\n繪製Walk-Forward分析圖表...")

    plot_save_path = save_path / "walkforward_chart.png"
    plot_walkforward_performance(
        results_df=results_df,
        title="MACD-ATR策略Walk-Forward分析",
        save_path=plot_save_path,
        combined_equity_curve=analyzer.combined_equity_curve,
        buy_hold_equity_curve=analyzer.buy_hold_equity_curve
    )

    if verbose:
        print("\n" + "=" * 70)
        print("樣本外(OOS)績效摘要")
        print("=" * 70)
        print(f"總窗口數:         {len(results_df)}")
        print(f"平均OOS夏普:      {results_df['oos_sharpe_ratio'].mean():.3f}")

    return {
        'analyzer': analyzer,
        'results_df': results_df,
        'save_path': save_path
    }


if __name__ == "__main__":
    DATA_CONFIG = "SOLUSDT_1h"
    MACD_FAST_RANGE = (5, 20, 3)
    MACD_SLOW_RANGE = (25, 45, 5)
    MACD_SIGNAL_RANGE = (7, 15, 2)
    ATR_WINDOW_RANGE = (10, 30, 5)
    ATR_MULTIPLIER_RANGE = (1.0, 2.5, 0.5)
    START_DATE = "2022-10-01"
    END_DATE = "2025-09-30"
    TRAIN_WINDOW = 8760
    TEST_WINDOW = 720
    STEP = 720
    OBJECTIVE = 'profit_factor'
    TRANSACTION_COST = 0.002
    SLIPPAGE = 0.0005
    N_JOBS = 3

    run_walkforward(
        data_config=DATA_CONFIG,
        macd_fast_range=MACD_FAST_RANGE,
        macd_slow_range=MACD_SLOW_RANGE,
        macd_signal_range=MACD_SIGNAL_RANGE,
        atr_window_range=ATR_WINDOW_RANGE,
        atr_multiplier_range=ATR_MULTIPLIER_RANGE,
        start_date=START_DATE,
        end_date=END_DATE,
        train_window=TRAIN_WINDOW,
        test_window=TEST_WINDOW,
        step=STEP,
        objective=OBJECTIVE,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE,
        n_jobs=N_JOBS,
        verbose=True
    )