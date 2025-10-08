"""
RSI-ATR Strategy - Walk-Forward Analysis
RSI-ATR策略 - Walk-Forward分析

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
from strategies.rsi_atr.strategy import RSIATRStrategy
from shared.walkforward import WalkForwardAnalyzer
from shared.visualization import plot_walkforward_performance

# 將策略函數定義在模組的最上層，以支持多進程
def rsi_atr_strategy_func(data: pd.DataFrame, rsi_window: int, rsi_long_threshold: int,
                          rsi_short_threshold: int, atr_window: int, atr_multiplier: float):
    """RSI-ATR策略包裝函數"""
    # 增加參數驗證，避免優化器傳入無效組合
    if rsi_long_threshold <= rsi_short_threshold:
        return None  # 返回 None，優化器會跳過此組合
    
    strategy = RSIATRStrategy()
    return strategy.generate_signals(
        data,
        rsi_window=rsi_window,
        rsi_long_threshold=rsi_long_threshold,
        rsi_short_threshold=rsi_short_threshold,
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
    rsi_window_range: tuple = (10, 20, 2),
    rsi_long_threshold_range: tuple = (55, 70, 5),
    rsi_short_threshold_range: tuple = (30, 45, 5),
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
        print("RSI-ATR策略Walk-Forward分析")
        print("=" * 70)
        print(f"參數設置:")
        print(f"  數據配置:         {data_config}")
        print(f"  rsi_window範圍:   {rsi_window_range}")
        print(f"  rsi_long_threshold範圍: {rsi_long_threshold_range}")
        print(f"  rsi_short_threshold範圍: {rsi_short_threshold_range}")
        print(f"  atr_window範圍:   {atr_window_range}")
        print(f"  atr_multiplier範圍: {atr_multiplier_range}")
        print(f"  訓練窗口:         {train_window} K棒")
        print(f"  測試窗口:         {test_window} K棒")
        print(f"  滾動步長:         {step} K棒")
        print(f"  優化目標:         {objective}")
        print(f"  交易成本:         {transaction_cost*100:.2f}%")
        print(f"  滑價:             {slippage*100:.3f}%")
        print(f"  並行核心:         {n_jobs}")
        print("=" * 70)

    # 步驟1: 載入數據
    df = load_data(data_config, start_date, end_date, verbose)

    # 步驟2: 設置參數網格
    param_grid = {
        'rsi_window': list(range(rsi_window_range[0], rsi_window_range[1] + 1, rsi_window_range[2])),
        'rsi_long_threshold': list(range(rsi_long_threshold_range[0], rsi_long_threshold_range[1] + 1, rsi_long_threshold_range[2])),
        'rsi_short_threshold': list(range(rsi_short_threshold_range[0], rsi_short_threshold_range[1] + 1, rsi_short_threshold_range[2])),
        'atr_window': list(range(atr_window_range[0], atr_window_range[1] + 1, atr_window_range[2])),
        'atr_multiplier': list(np.arange(atr_multiplier_range[0], atr_multiplier_range[1] + atr_multiplier_range[2] * 0.001, atr_multiplier_range[2]))
    }

    # 步驟3: 創建Walk-Forward分析器
    timeframe = data_config.split('_')[1]
    periods_per_year = 365 if timeframe == "1d" else 365 * 24

    analyzer = WalkForwardAnalyzer(
        strategy_func=rsi_atr_strategy_func,
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

    # 步驟4: 執行Walk-Forward分析
    results_df = analyzer.run(verbose=verbose)

    # 步驟5: 保存結果
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_source = f"WalkForward_RSI_ATR_{data_config}_{date_str}"

    output_dir = Path(__file__).parent / "results"
    save_path = analyzer.save_results(data_source=data_source, output_dir=output_dir)

    # 步驟6: 繪製Walk-Forward分析圖表
    if verbose:
        print("\n繪製Walk-Forward分析圖表...")

    plot_save_path = save_path / "walkforward_chart.png"
    plot_walkforward_performance(
        results_df=results_df,
        title="RSI-ATR策略Walk-Forward分析",
        save_path=plot_save_path,
        combined_equity_curve=analyzer.combined_equity_curve,
        buy_hold_equity_curve=analyzer.buy_hold_equity_curve
    )

    # 步驟7: 打印樣本外績效摘要
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
    # ========== 數據配置 ========== 
    DATA_CONFIG = "SOLUSDT_1h"        # 數據配置: BTCUSDT_1d, BTCUSDT_1h, SOLUSDT_1h

    # ========== Walk-Forward參數 ========== 
    # 保持與舊文件一致的參數，但可以根據需要調整
    RSI_WINDOW_RANGE = (1, 25, 1)
    RSI_LONG_THRESHOLD_RANGE = (50, 95, 5)
    RSI_SHORT_THRESHOLD_RANGE = (5, 51, 5)
    ATR_WINDOW_RANGE = (1, 5, 1)
    ATR_MULTIPLIER_RANGE = (0.5, 1.6, 0.5)
    
    START_DATE = "2022-10-01"
    END_DATE = "2025-09-30"
    TRAIN_WINDOW = 17520
    TEST_WINDOW = 720
    STEP = 720
    OBJECTIVE = 'profit_factor'
    TRANSACTION_COST = 0.002
    SLIPPAGE = 0.0005
    N_JOBS = 3 # 修正為3，-1可能在某些系統上有問題

    # 執行Walk-Forward分析
    print("\n" + "=" * 70)
    print("RSI-ATR策略 - Walk-Forward分析")
    print("=" * 70)

    wf_results = run_walkforward(
        data_config=DATA_CONFIG,
        rsi_window_range=RSI_WINDOW_RANGE,
        rsi_long_threshold_range=RSI_LONG_THRESHOLD_RANGE,
        rsi_short_threshold_range=RSI_SHORT_THRESHOLD_RANGE,
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

    print("\n測試完成！")
    print(f"結果已保存到: {wf_results['save_path']}")
    print(f"平均樣本外夏普比率: {wf_results['results_df']['oos_sharpe_ratio'].mean():.3f}")