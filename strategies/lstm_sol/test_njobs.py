"""
測試 N_JOBS 最佳並行數
Test optimal N_JOBS for parallel optimization
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import time
import pandas as pd
from strategies.lstm_sol.strategy import LSTMSOLStrategy
from shared.optimizer import ParameterOptimizer
from shared.data_loader import load_data_with_fallback
from data_fetchers.bybit_sol_1h_fetcher import fetch_bybit_sol_1h_data


def lstm_sol_strategy_func(data: pd.DataFrame, rsi_window: int, rsi_long_threshold: int, rsi_short_threshold: int, sma_period: int):
    """LSTM SOL策略包裝函數"""
    strategy = LSTMSOLStrategy()
    return strategy.generate_signals(
        data,
        symbol="SOL/USD",
        rsi_window=rsi_window,
        rsi_long_threshold=rsi_long_threshold,
        rsi_short_threshold=rsi_short_threshold,
        sma_period=sma_period
    )


if __name__ == "__main__":
    # 載入小量數據進行測試
    print("載入測試數據...")
    df = load_data_with_fallback(
        data_source="1h",
        symbol="SOLUSDT",
        start_date="2024-01-01",
        end_date="2024-06-30",
        fetcher_func=fetch_bybit_sol_1h_data,
        verbose=False
    )

    # 小參數網格 (32 組合)
    param_grid = {
        'rsi_window': [14, 16, 18, 20],
        'rsi_long_threshold': [55, 60],
        'rsi_short_threshold': [40, 45],
        'sma_period': [20, 30]
    }

    # 測試不同的 N_JOBS
    test_n_jobs = [1, 2, 4, 8, 12, 16]
    results = []

    print(f"\n測試 {len(param_grid['rsi_window']) * len(param_grid['rsi_long_threshold']) * len(param_grid['rsi_short_threshold']) * len(param_grid['sma_period'])} 組參數組合")
    print(f"CPU 核心數: 22\n")

    for n_jobs in test_n_jobs:
        print(f"測試 N_JOBS = {n_jobs}...", end=" ", flush=True)

        optimizer = ParameterOptimizer(
            strategy_func=lstm_sol_strategy_func,
            data=df,
            param_grid=param_grid,
            objective='profit_factor',
            transaction_cost=0.002,
            slippage=0.0005,
            n_jobs=n_jobs,
            periods_per_year=365 * 24
        )

        start_time = time.time()
        optimizer.optimize(verbose=False)
        elapsed_time = time.time() - start_time

        results.append({
            'n_jobs': n_jobs,
            'time': elapsed_time,
            'speedup': results[0]['time'] / elapsed_time if results else 1.0
        })

        print(f"耗時: {elapsed_time:.2f}秒, 加速比: {results[-1]['speedup']:.2f}x")

    # 輸出結果
    print("\n" + "="*60)
    print("N_JOBS 性能測試結果:")
    print("="*60)
    print(f"{'N_JOBS':<10} {'耗時(秒)':<15} {'加速比':<10} {'建議':<20}")
    print("-"*60)

    best_idx = min(range(len(results)), key=lambda i: results[i]['time'])

    for i, r in enumerate(results):
        suggestion = "[BEST]" if i == best_idx else ""
        if r['speedup'] / results[best_idx]['speedup'] > 0.95 and r['n_jobs'] < results[best_idx]['n_jobs']:
            suggestion = "[RECOMMENDED]"

        print(f"{r['n_jobs']:<10} {r['time']:<15.2f} {r['speedup']:<10.2f} {suggestion}")

    print("\n建議:")
    print(f"- 最快設置: N_JOBS = {results[best_idx]['n_jobs']}")
    print(f"- 推薦設置: N_JOBS = {min(12, results[best_idx]['n_jobs'])} (兼顧性能與系統負載)")
