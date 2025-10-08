"""
MA Crossover Strategy - Parameter Optimization
MA交叉策略 - 參數優化測試

測試目的：
1. 優化short_period和long_period參數
2. 尋找最佳參數組合
3. 分析參數敏感性
4. 保存優化結果和圖表

參數範圍：
- short_period: 5 到 30，步長5
- long_period: 20 到 100，步長10
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np
from datetime import datetime

# 導入策略和工具
from strategies.ma_crossover.strategy import MACrossoverStrategy
from shared.optimizer import ParameterOptimizer
from shared.visualization import plot_optimization_results_medium_style


def ma_strategy_func(data: pd.DataFrame, short_period: int, long_period: int):
    """MA策略包裝函數 (模組級別函數以支援多進程)"""
    # 檢查參數有效性（short必須小於long）
    if short_period >= long_period:
        # 返回空信號（無效參數組合）
        return pd.Series(0, index=data.index)

    strategy = MACrossoverStrategy()
    return strategy.generate_signals(data, short_period=short_period, long_period=long_period)


def load_data(data_config: str = "BTCUSDT_1d", start_date: str = "2021-01-01", end_date: str = "2024-12-31", verbose: bool = True):
    """
    載入數據 (優先使用本地數據)

    使用統一的data_loader模組,優先從本地載入已下載的數據,
    大幅提升測試速度,避免重複下載。
    """
    from shared.data_loader import load_data_with_fallback
    from data_fetchers.bybit_btc_1d_fetcher import fetch_bybit_btc_1d_data

    # 根據配置選擇數據源和fetcher
    if data_config == "BTCUSDT_1h":
        from data_fetchers.bybit_btc_1h_fetcher import fetch_bybit_btc_1h_data
        data_source = "1h"
        fetcher_func = fetch_bybit_btc_1h_data
    else:  # 默認使用 BTCUSDT_1d
        data_source = "1d"
        fetcher_func = fetch_bybit_btc_1d_data

    return load_data_with_fallback(
        data_source=data_source,
        start_date=start_date,
        end_date=end_date,
        fetcher_func=fetcher_func,
        verbose=verbose
    )


def run_optimization(
    data_config: str = "BTCUSDT_1d",
    short_range: tuple = (5, 30, 5),
    long_range: tuple = (20, 100, 10),
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

    Args:
        data_config: 數據配置 (BTCUSDT_1d 或 BTCUSDT_1h)
        short_range: short_period參數範圍 (start, end, step)
        long_range: long_period參數範圍 (start, end, step)
        start_date: 開始日期
        end_date: 結束日期
        objective: 優化目標（profit_factor, sharpe_ratio等）
        transaction_cost: 交易成本
        slippage: 滑價
        n_jobs: 並行核心數
        verbose: 是否打印詳細信息

    Returns:
        Dict: 優化結果字典
    """
    # 步驟1: 載入數據
    df = load_data(data_config, start_date, end_date, verbose=False)

    # 步驟2: 設置參數網格
    param_grid = {
        'short_period': list(range(short_range[0], short_range[1], short_range[2])),
        'long_period': list(range(long_range[0], long_range[1], long_range[2]))
    }

    # 步驟3: 執行優化
    # 根據數據配置設定年化參數
    periods_per_year = 365 if data_config == "BTCUSDT_1d" else 365 * 24

    optimizer = ParameterOptimizer(
        strategy_func=ma_strategy_func,
        data=df,
        param_grid=param_grid,
        objective=objective,
        transaction_cost=transaction_cost,
        slippage=slippage,
        n_jobs=n_jobs,
        periods_per_year=periods_per_year
    )

    results_df = optimizer.optimize(verbose=verbose)

    # 步驟4: 保存結果（改進命名方式以顯示測試名稱）
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 改進命名：包含測試名稱、策略、數據源、優化目標
    data_source = f"Optimization_MACrossover_{data_config}_{objective}_{date_str}"

    output_dir = Path(__file__).parent / "results"
    save_path = optimizer.save_results(data_source=data_source, output_dir=output_dir)

    # 步驟5: 繪製優化結果圖表 (Medium風格)
    # Minor Bug Fix 9: Create copy of results_df to avoid side effects
    results_df_copy = results_df.copy()

    # 確保 results_df 包含 final_value 欄位
    if 'final_value' not in results_df_copy.columns:
        # Medium Bug Fix 5: Fix total_return calculation - total_return is decimal not percentage
        # total_return is already in decimal format (e.g., 0.15 for 15%), not percentage (15.0)
        initial_capital = 100000
        results_df_copy['final_value'] = initial_capital * (1 + results_df_copy['total_return'])

    plot_optimization_results_medium_style(
        results_df=results_df_copy,
        param_names=['short_period', 'long_period'],
        output_dir=save_path
    )

    # 步驟6: 打印最佳結果 (Medium風格簡潔輸出)
    if verbose:
        best_params = optimizer.get_best_params()
        print(f"Best Parameters: short_period={best_params['short_period']}, long_period={best_params['long_period']}, {objective}={optimizer.best_score:.3f}")

    return {
        'optimizer': optimizer,
        'results_df': results_df,
        'best_params': optimizer.get_best_params(),
        'best_score': optimizer.best_score,
        'save_path': save_path
    }


if __name__ == "__main__":
    # ========== 數據配置 ==========
    DATA_CONFIG = "BTCUSDT_1h"        # 數據配置: BTCUSDT_1d 或 BTCUSDT_1h

    # ========== 優化參數 ==========
    SHORT_RANGE = (18, 121, 1)           # short_period從1到5，步長1
    LONG_RANGE = (510, 720, 1)         # long_period從36到145，步長1
    START_DATE = "2022-10-01"         # 開始日期
    END_DATE = "2024-09-30"           # 結束日期
    OBJECTIVE = 'profit_factor'       # 優化目標（可選: sharpe_ratio, profit_factor, calmar_ratio）
    TRANSACTION_COST = 0.002          # 交易成本 0.2%
    SLIPPAGE = 0.0005                 # 滑價 0.05%
    N_JOBS = 1                        # 並行核心數 (Windows多進程問題，暫時使用1)

    # 執行優化 (Medium 風格簡潔輸出)
    print(f"Parameter Optimization - MA Crossover Strategy")
    print(f"Data: {DATA_CONFIG}, {START_DATE} to {END_DATE}")

    optimization_results = run_optimization(
        data_config=DATA_CONFIG,
        short_range=SHORT_RANGE,
        long_range=LONG_RANGE,
        start_date=START_DATE,
        end_date=END_DATE,
        objective=OBJECTIVE,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE,
        n_jobs=N_JOBS,
        verbose=True
    )

    print(f"Results saved to: {optimization_results['save_path']}")
