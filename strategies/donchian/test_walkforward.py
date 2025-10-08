"""
Donchian Strategy - Walk-Forward Analysis
Donchian策略 - Walk-Forward分析

測試目的：
1. 模擬實盤的滾動優化過程
2. 評估樣本外(OOS)績效
3. 檢驗參數穩定性
4. 避免前視偏差
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# 導入策略和工具
from strategies.donchian.strategy import DonchianStrategy
from shared.walkforward import WalkForwardAnalyzer
from shared.visualization import plot_walkforward_performance

# 將策略函數定義在模組的最上層，以支持多進程
def donchian_strategy_func(data: pd.DataFrame, lookback: int):
    """Donchian策略包裝函數"""
    strategy = DonchianStrategy()
    return strategy.generate_signals(data, lookback=lookback)

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
    lookback_range: tuple = (10, 100, 5),
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
        print("Donchian策略Walk-Forward分析")
        print("=" * 70)
        print(f"參數設置:")
        print(f"  數據配置:         {data_config}")
        print(f"  lookback範圍:     {lookback_range}")
        print(f"  訓練窗口:         {train_window} K棒")
        print(f"  測試窗口:         {test_window} K棒")
        print(f"  滾動步長:         {step} K棒")
        print(f"  優化目標:         {objective}")
        print(f"  交易成本:         {transaction_cost*100:.2f}%")
        print(f"  滑價:             {slippage*100:.3f}%")
        print(f"  並行核心:         {n_jobs}")
        print("=" * 70)

    df = load_data(data_config, start_date, end_date, verbose)

    param_grid = {
        'lookback': list(range(lookback_range[0], lookback_range[1] + 1, lookback_range[2]))
    }

    timeframe = data_config.split('_')[1]
    periods_per_year = 365 if timeframe == "1d" else 365 * 24

    analyzer = WalkForwardAnalyzer(
        strategy_func=donchian_strategy_func,
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
    data_source = f"WalkForward_Donchian_{data_config}_{date_str}"

    output_dir = Path(__file__).parent / "results"
    save_path = analyzer.save_results(data_source=data_source, output_dir=output_dir)

    if verbose:
        print("\n繪製Walk-Forward分析圖表...")

    plot_save_path = save_path / "walkforward_chart.png"
    plot_walkforward_performance(
        results_df=results_df,
        title="Donchian策略Walk-Forward分析",
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
        print(f"正收益窗口:       {(results_df['oos_total_return'] > 0).sum()} / {len(results_df)}")

        is_oos_ratio = (results_df['oos_sharpe_ratio'].mean() /
                       results_df['is_sharpe_ratio'].mean()
                       if results_df['is_sharpe_ratio'].mean() != 0 else 0)
        print(f"OOS/IS夏普比率:   {is_oos_ratio:.3f}")
        print("  (接近1表示沒有過度擬合)")
        print("=" * 70)

    return {
        'analyzer': analyzer,
        'results_df': results_df,
        'save_path': save_path
    }


if __name__ == "__main__":
    DATA_CONFIG = "BTCUSDT_1h"
    LOOKBACK_RANGE = (20, 200, 10)
    START_DATE = "2022-10-01"
    END_DATE = "2025-09-30"
    TRAIN_WINDOW = 8760
    TEST_WINDOW = 720
    STEP = 720
    OBJECTIVE = 'profit_factor'
    TRANSACTION_COST = 0.002
    SLIPPAGE = 0.0005
    N_JOBS = 3

    print("\n" + "=" * 70)
    print("Donchian策略 - Walk-Forward分析")
    print("=" * 70)

    wf_results = run_walkforward(
        data_config=DATA_CONFIG,
        lookback_range=LOOKBACK_RANGE,
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

    print("\n測試完成！")
    print(f"結果已保存到: {wf_results['save_path']}")
    print(f"平均樣本外夏普比率: {wf_results['results_df']['oos_sharpe_ratio'].mean():.3f}")