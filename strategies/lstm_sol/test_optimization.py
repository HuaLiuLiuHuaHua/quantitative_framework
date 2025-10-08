"""
LSTM SOL Strategy - Parameter Optimization
LSTM SOL策略 - 參數優化測試

測試目的：
1. 優化策略參數
2. 尋找最佳參數組合
3. 分析參數敏感性
4. 保存優化結果和圖表

參數範圍：
- rsi_window: 10 到 20，步長2
- rsi_long_threshold: 50 到 70，步長5
- rsi_short_threshold: 30 到 50，步長5
- sma_period: 10 到 50，步長10
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np
from datetime import datetime

# 導入策略和工具
from strategies.lstm_sol.strategy import LSTMSOLStrategy
from shared.optimizer import ParameterOptimizer
from shared.visualization import plot_optimization_results_medium_style


def lstm_sol_strategy_func(data: pd.DataFrame, **params):
    """LSTM SOL策略包裝函數 (模組級別函數以支援多進程)"""
    strategy = LSTMSOLStrategy()
    return strategy.generate_signals(data, symbol="SOL/USD", **params)


def load_data(data_config: str = "SOLUSDT_1h", start_date: str = "2021-01-01", end_date: str = "2024-12-31", verbose: bool = True):
    """
    載入數據 (優先使用本地數據)

    使用統一的data_loader模組,優先從本地載入已下載的數據,
    大幅提升測試速度,避免重複下載。
    """
    from shared.data_loader import load_data_with_fallback
    from data_fetchers.bybit_sol_1h_fetcher import fetch_bybit_sol_1h_data

    # 根據配置選擇數據源和fetcher
    if data_config == "SOLUSDT_1h":
        data_source = "1h"
        symbol = "SOLUSDT"
        fetcher_func = fetch_bybit_sol_1h_data
    else:
        raise ValueError(f"不支援的數據配置: {data_config}")

    return load_data_with_fallback(
        data_source=data_source,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        fetcher_func=fetcher_func,
        verbose=verbose
    )


def run_optimization(
    data_config: str = "SOLUSDT_1h",
    param_grid: dict = None,
    start_date: str = "2021-01-01",
    end_date: str = "2023-12-31",
    objective: str = 'profit_factor',
    transaction_cost: float = 0.002,
    slippage: float = 0.0005,
    n_jobs: int = 1,
    verbose: bool = True
):
    """
    執行參數優化

    Args:
        data_config: 數據配置 (SOLUSDT_1h)
        param_grid: 參數網格字典
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

    # 步驟3: 執行優化
    # 1小時線，一年 365*24 小時
    periods_per_year = 365 * 24

    optimizer = ParameterOptimizer(
        strategy_func=lstm_sol_strategy_func,
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
    data_source = f"Optimization_LSTMSOL_{data_config}_{objective}_{date_str}"

    output_dir = Path(__file__).parent / "results"
    save_path = optimizer.save_results(data_source=data_source, output_dir=output_dir)

    # 步驟5: 繪製優化結果圖表 (Medium風格)
    results_df_copy = results_df.copy()

    # 確保 results_df 包含 final_value 欄位
    if 'final_value' not in results_df_copy.columns:
        initial_capital = 100000
        results_df_copy['final_value'] = initial_capital * (1 + results_df_copy['total_return'])

    # 獲取實際參數名稱
    param_names = list(param_grid.keys())

    plot_optimization_results_medium_style(
        results_df=results_df_copy,
        param_names=param_names,
        output_dir=save_path
    )

    # 步驟6: 打印最佳結果 (Medium風格簡潔輸出)
    if verbose:
        best_params = optimizer.get_best_params()
        params_str = ", ".join([f"{k}={v}" for k, v in best_params.items()])
        print(f"Best Parameters: {params_str}, {objective}={optimizer.best_score:.3f}")

    return {
        'optimizer': optimizer,
        'results_df': results_df,
        'best_params': optimizer.get_best_params(),
        'best_score': optimizer.best_score,
        'save_path': save_path
    }


if __name__ == "__main__":
    # ========== 數據配置 ==========
    DATA_CONFIG = "SOLUSDT_1h"        # 數據配置: SOLUSDT_1h
    START_DATE = "2022-10-01"         # 開始日期
    END_DATE = "2024-09-30"           # 結束日期
    OBJECTIVE = 'profit_factor'       # 優化目標（可選: sharpe_ratio, profit_factor, calmar_ratio）
    TRANSACTION_COST = 0.002          # 交易成本 0.2%
    SLIPPAGE = 0.0005                 # 滑價 0.05%
    N_JOBS = 1                        # 並行核心數 (Windows: N_JOBS=1 最快)

    # ========== 合理的參數網格 (平衡優化效果與時間) ==========
    # 總組合數: 2*3*3*2*2*2*2*2*2*2 = 1,152 組合 (約 21 分鐘)
    PARAM_GRID = {
        # RSI 參數
        'rsi_window': [14, 18],                        # RSI 窗口期 (2個值)
        'rsi_long_threshold': [55, 60, 65],            # RSI 做多閾值 (3個值)
        'rsi_short_threshold': [35, 40, 45],           # RSI 做空閾值 (3個值)
        'rsi_exit_offset': [5, 7],                     # RSI 出場偏移量 (2個值)

        # SMA 參數
        'sma_period': [20, 30],                        # SMA 週期 (2個值)
        'sma_long_buffer': [0.98, 0.99],               # 多頭出場緩衝 (2個值)
        'sma_short_buffer': [1.01, 1.02],              # 空頭出場緩衝 (2個值)

        # MACD 參數
        'macd_fast': [12, 14],                         # MACD 快線 (2個值)
        'macd_slow': [26, 28],                         # MACD 慢線 (2個值)
        'macd_signal': [9, 10]                         # MACD 信號線 (2個值)
    }

    # 計算總組合數
    import numpy as np
    total_combinations = np.prod([len(v) for v in PARAM_GRID.values()])
    print(f"Parameter Optimization - LSTM SOL Strategy (Enhanced)")
    print(f"Data: {DATA_CONFIG}, {START_DATE} to {END_DATE}")
    print(f"Total combinations: {total_combinations:,}")
    print(f"Estimated time: ~{total_combinations * 1.1 / 60:.1f} minutes\n")

    # 執行優化
    optimization_results = run_optimization(
        data_config=DATA_CONFIG,
        param_grid=PARAM_GRID,
        start_date=START_DATE,
        end_date=END_DATE,
        objective=OBJECTIVE,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE,
        n_jobs=N_JOBS,
        verbose=True
    )

    print(f"\nResults saved to: {optimization_results['save_path']}")
