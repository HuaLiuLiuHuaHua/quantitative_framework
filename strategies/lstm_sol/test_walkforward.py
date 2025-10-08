"""
LSTM SOL (Simplified) Strategy - Walk-Forward Analysis
LSTM SOL (簡化版) 策略 - Walk-Forward分析
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np
from datetime import datetime

# 導入策略和工具
from strategies.lstm_sol.strategy import LSTMSOLStrategy
from shared.walkforward import WalkForwardAnalyzer
from shared.visualization import plot_walkforward_performance


def lstm_sol_strategy_func(data: pd.DataFrame, rsi_window: int, sma_period: int, rsi_long_threshold: int, rsi_short_threshold: int):
    """LSTM SOL (Simplified) 策略包裝函數"""
    strategy = LSTMSOLStrategy()
    # 傳遞選擇優化的參數，其他使用默認值
    return strategy.generate_signals(
        data, 
        symbol="SOL/USD",
        rsi_window=rsi_window,
        sma_period=sma_period,
        rsi_long_threshold=rsi_long_threshold,
        rsi_short_threshold=rsi_short_threshold
    )

def load_data(data_config: str = "SOLUSDT_1h", start_date: str = "2022-10-01", end_date: str = "2025-09-30", verbose: bool = True):
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
    data_config: str = "SOLUSDT_1h",
    rsi_window_range: tuple = (10, 30, 5),
    sma_period_range: tuple = (15, 40, 5),
    rsi_long_threshold_range: tuple = (55, 70, 5),
    rsi_short_threshold_range: tuple = (30, 45, 5),
    start_date: str = "2022-10-01",
    end_date: str = "2025-09-30",
    train_window: int = 8760,
    test_window: int = 720,
    step: int = 720,
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
        print("LSTM SOL (Simplified) 策略Walk-Forward分析")
        print("=" * 70)

    df = load_data(data_config, start_date, end_date, verbose)

    param_grid = {
        'rsi_window': list(range(rsi_window_range[0], rsi_window_range[1] + 1, rsi_window_range[2])),
        'sma_period': list(range(sma_period_range[0], sma_period_range[1] + 1, sma_period_range[2])),
        'rsi_long_threshold': list(range(rsi_long_threshold_range[0], rsi_long_threshold_range[1] + 1, rsi_long_threshold_range[2])),
        'rsi_short_threshold': list(range(rsi_short_threshold_range[0], rsi_short_threshold_range[1] + 1, rsi_short_threshold_range[2]))
    }

    timeframe = data_config.split('_')[1]
    periods_per_year = 365 if timeframe == "1d" else 365 * 24

    analyzer = WalkForwardAnalyzer(
        strategy_func=lstm_sol_strategy_func,
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
    data_source = f"WalkForward_LSTMSOL_{data_config}_{date_str}"

    output_dir = Path(__file__).parent / "results"
    save_path = analyzer.save_results(data_source=data_source, output_dir=output_dir)

    if verbose:
        print("\n繪製Walk-Forward分析圖表...")

    plot_save_path = save_path / "walkforward_chart.png"
    plot_walkforward_performance(
        results_df=results_df,
        title="LSTM SOL (Simplified) 策略Walk-Forward分析",
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
        print(f"平均OOS收益:      {results_df['oos_total_return'].mean()*100:.2f}%")
        print(f"平均OOS回撤:      {results_df['oos_max_drawdown'].mean()*100:.2f}%")
        print(f"平均OOS獲利因子:  {results_df['oos_profit_factor'].mean():.3f}")
        print(f"平均OOS勝率:      {results_df['oos_win_rate'].mean()*100:.2f}%")

    return {
        'analyzer': analyzer,
        'results_df': results_df,
        'save_path': save_path
    }


if __name__ == "__main__":
    DATA_CONFIG = "SOLUSDT_1h"
    RSI_WINDOW_RANGE = (10, 30, 5)
    SMA_PERIOD_RANGE = (15, 40, 5)
    RSI_LONG_THRESHOLD_RANGE = (55, 70, 5)
    RSI_SHORT_THRESHOLD_RANGE = (30, 45, 5)
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
        rsi_window_range=RSI_WINDOW_RANGE,
        sma_period_range=SMA_PERIOD_RANGE,
        rsi_long_threshold_range=RSI_LONG_THRESHOLD_RANGE,
        rsi_short_threshold_range=RSI_SHORT_THRESHOLD_RANGE,
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