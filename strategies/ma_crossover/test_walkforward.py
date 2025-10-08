"""
MA Crossover Strategy - Walk-Forward Analysis
移動平均線交叉策略 - Walk-Forward分析
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# 導入策略和工具
from strategies.ma_crossover.strategy import MACrossoverStrategy
from shared.walkforward import WalkForwardAnalyzer
from shared.visualization import plot_walkforward_performance

def ma_crossover_strategy_func(data: pd.DataFrame, short_period: int, long_period: int):
    """MA Crossover策略包裝函數"""
    # 確保 short_period < long_period，否則跳過此次回測
    if short_period >= long_period:
        return None
    strategy = MACrossoverStrategy()
    return strategy.generate_signals(data, short_period=short_period, long_period=long_period)

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
    short_period_range: tuple = (5, 30, 5),
    long_period_range: tuple = (20, 100, 10),
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
        print("MA Crossover策略Walk-Forward分析")
        print("=" * 70)

    df = load_data(data_config, start_date, end_date, verbose)

    param_grid = {
        'short_period': list(range(short_period_range[0], short_period_range[1] + 1, short_period_range[2])),
        'long_period': list(range(long_period_range[0], long_period_range[1] + 1, long_period_range[2]))
    }

    timeframe = data_config.split('_')[1]
    periods_per_year = 365 if timeframe == "1d" else 365 * 24

    analyzer = WalkForwardAnalyzer(
        strategy_func=ma_crossover_strategy_func,
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
    data_source = f"WalkForward_MACrossover_{data_config}_{date_str}"

    output_dir = Path(__file__).parent / "results"
    save_path = analyzer.save_results(data_source=data_source, output_dir=output_dir)

    if verbose:
        print("\n繪製Walk-Forward分析圖表...")

    plot_save_path = save_path / "walkforward_chart.png"
    plot_walkforward_performance(
        results_df=results_df,
        title="MA Crossover策略Walk-Forward分析",
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
    DATA_CONFIG = "BTCUSDT_1h"
    SHORT_PERIOD_RANGE = (10, 50, 5)
    LONG_PERIOD_RANGE = (60, 150, 10)
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
        short_period_range=SHORT_PERIOD_RANGE,
        long_period_range=LONG_PERIOD_RANGE,
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
