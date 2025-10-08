"""
BB-ATR Strategy - Parameter Optimization
BB-ATR策略 - 參數優化測試

測試目的：
1. 優化策略參數
2. 尋找最佳參數組合
3. 分析參數敏感性
4. 保存優化結果和圖表

參數範圍：
- bb_window: 15 到 25，步長5
- bb_std: 1.5 到 2.5，步長0.5
- atr_window: 10 到 20，步長2
- atr_multiplier: 1.5 到 3.0，步長0.5
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import numpy as np
import pandas as pd
from datetime import datetime

# 導入策略和工具
from strategies.bb_atr.strategy import BBATRStrategy
from shared.optimizer import ParameterOptimizer
from shared.visualization import plot_optimization_results_medium_style
from shared.data_loader import load_data_with_fallback


def bb_atr_strategy_func(data: pd.DataFrame, bb_window: int, bb_std: float,
                           atr_window: int, atr_multiplier: float):
    """BB-ATR策略包裝函數 (模組級別函數以支援多進程)"""
    strategy = BBATRStrategy()
    return strategy.generate_signals(
        data,
        bb_window=bb_window,
        bb_std=bb_std,
        atr_window=atr_window,
        atr_multiplier=atr_multiplier
    )


def load_data(symbol: str, timeframe: str, start_date: str, end_date: str, verbose: bool = False):
    """
    載入數據 (優先使用本地數據，若無則下載)
    """
    # 根據 symbol 和 timeframe 選擇對應的 fetcher
    if symbol == "BTCUSDT" and timeframe == "1h":
        from data_fetchers.bybit_btc_1h_fetcher import fetch_bybit_btc_1h_data
        fetcher_func = fetch_bybit_btc_1h_data
    elif symbol == "BTCUSDT" and timeframe == "1d":
        from data_fetchers.bybit_btc_1d_fetcher import fetch_bybit_btc_1d_data
        fetcher_func = fetch_bybit_btc_1d_data
    elif symbol == "SOLUSDT" and timeframe == "1h":
        from data_fetchers.bybit_sol_1h_fetcher import fetch_bybit_sol_1h_data
        fetcher_func = fetch_bybit_sol_1h_data
    else:
        raise ValueError(f"Unsupported symbol/timeframe combination: {symbol}/{timeframe}")

    return load_data_with_fallback(
        data_source=timeframe,  # data_source is '1h' or '1d'
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        fetcher_func=fetcher_func,
        verbose=verbose
    )


def run_optimization(
    symbol: str = "BTCUSDT",
    timeframe: str = "1d",
    bb_window_range: tuple = (15, 25, 5),
    bb_std_range: tuple = (1.5, 2.5, 0.5),
    atr_window_range: tuple = (10, 20, 2),
    atr_multiplier_range: tuple = (1.5, 3.0, 0.5),
    start_date: str = "2021-01-01",
    end_date: str = "2023-12-31",
    objective: str = 'profit_factor',
    transaction_cost: float = 0.002,
    slippage: float = 0.0005,
    n_jobs: int = 4,
    verbose: bool = True
):
    """
    執行參數優化
    """
    # 步驟1: 載入數據
    df = load_data(symbol, timeframe, start_date, end_date, verbose=False)

    # 步驟2: 設置參數網格
    param_grid = {
        'bb_window': list(range(bb_window_range[0], bb_window_range[1], bb_window_range[2])),
        'bb_std': list(np.arange(bb_std_range[0], bb_std_range[1], bb_std_range[2])),
        'atr_window': list(range(atr_window_range[0], atr_window_range[1], atr_window_range[2])),
        'atr_multiplier': list(np.arange(atr_multiplier_range[0], atr_multiplier_range[1], atr_multiplier_range[2]))
    }

    # 步驟3: 執行優化
    periods_per_year = 365 if timeframe == "1d" else 365 * 24

    optimizer = ParameterOptimizer(
        strategy_func=bb_atr_strategy_func,
        data=df,
        param_grid=param_grid,
        objective=objective,
        transaction_cost=transaction_cost,
        slippage=slippage,
        n_jobs=n_jobs,
        periods_per_year=periods_per_year
    )

    results_df = optimizer.optimize(verbose=verbose)

    # 步驟4: 保存結果
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_name = f"Optimization_BB_ATR_{symbol}_{timeframe}_{objective}_{date_str}"

    output_dir = Path(__file__).parent / "results"
    save_path = optimizer.save_results(data_source=test_name, output_dir=output_dir)

    # 步驟5: 繪製優化結果圖表
    plot_optimization_results_medium_style(
        results_df=results_df,
        param_names=['bb_window', 'bb_std', 'atr_window', 'atr_multiplier'],
        output_dir=save_path
    )

    # 步驟6: 打印最佳結果
    if verbose:
        best_params = optimizer.get_best_params()
        best_score = optimizer.best_score
        if best_score is not None:
            print(f"Best Parameters: bb_window={best_params['bb_window']}, bb_std={best_params['bb_std']:.2f}, atr_window={best_params['atr_window']}, atr_multiplier={best_params['atr_multiplier']:.2f}, {objective}={best_score:.3f}")
        else:
            print(f"Optimization finished, but no valid score was found.")

    return {
        'optimizer': optimizer,
        'results_df': results_df,
        'best_params': optimizer.get_best_params(),
        'best_score': optimizer.best_score,
        'save_path': save_path
    }


if __name__ == "__main__":
    # ========== 數據配置 ==========
    SYMBOL = "SOLUSDT"
    TIMEFRAME = "1h"

    # ========== 優化參數 ==========
    BB_WINDOW_RANGE = (120, 169, 1)
    BB_STD_RANGE = (2.0, 3.1, 0.5)
    ATR_WINDOW_RANGE = (8, 13, 1)
    ATR_MULTIPLIER_RANGE = (0.5, 1.6, 0.5)
    START_DATE = "2022-10-01"
    END_DATE = "2024-09-30"
    OBJECTIVE = 'profit_factor'
    TRANSACTION_COST = 0.002
    SLIPPAGE = 0.0005
    N_JOBS = -1  # 並行核心數 (-1 表示使用所有可用的CPU核心)

    # 執行優化
    print(f"Running BB-ATR Strategy Parameter Optimization for {SYMBOL} ({TIMEFRAME}) from {START_DATE} to {END_DATE}")

    optimization_results = run_optimization(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        bb_window_range=BB_WINDOW_RANGE,
        bb_std_range=BB_STD_RANGE,
        atr_window_range=ATR_WINDOW_RANGE,
        atr_multiplier_range=ATR_MULTIPLIER_RANGE,
        start_date=START_DATE,
        end_date=END_DATE,
        objective=OBJECTIVE,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE,
        n_jobs=N_JOBS,
        verbose=True
    )

    print(f"Results saved to: {optimization_results['save_path']}")
