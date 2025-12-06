"""
XGBoost 因子組合 - 隨機參數優化

使用隨機搜索尋找最優參數
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import matplotlib
matplotlib.use('Agg')

from shared.data_loader import load_local_data
from shared.optimizer import ParameterOptimizer
from strategies.ml_combined.xgboost_combined.strategy import XGBoostCombinedStrategy


def main():
    print("=" * 70)
    print("XGBoost 因子組合 - 隨機參數優化")
    print("=" * 70)

    # ===== 配置 =====
    SYMBOL = 'BTCUSDT'
    DATA_SOURCE = '1h'
    TRAIN_START = '2022-12-01'
    TRAIN_END = '2023-12-31'
    TEST_START = '2024-01-01'
    TEST_END = '2024-12-31'

    # ===== 加載數據 =====
    print(f"\n加載訓練數據...")
    train_data = load_local_data(
        symbol=SYMBOL,
        data_source=DATA_SOURCE,
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        lookback_days=100
    )

    print(f"加載測試數據...")
    test_data = load_local_data(
        symbol=SYMBOL,
        data_source=DATA_SOURCE,
        start_date=TEST_START,
        end_date=TEST_END,
        lookback_days=100
    )

    if train_data is None or test_data is None:
        print(f"[ERROR] 未找到數據")
        return

    print(f"[OK] 訓練數據: {len(train_data)}")
    print(f"[OK] 測試數據: {len(test_data)}")

    # ===== 初始化策略和優化器 =====
    print(f"\n初始化參數優化器...")
    strategy = XGBoostCombinedStrategy()

    optimizer = ParameterOptimizer(
        strategy=strategy,
        data=train_data,
        test_data=test_data,
        n_iter=50,  # 隨機搜索迭代次數
        n_jobs=-1,  # 使用所有 CPU 核心
        transaction_cost=0.0006,
        slippage=0.0001,
        initial_capital=100000,
        periods_per_year=8760
    )

    # ===== 執行優化 =====
    print(f"\n執行隨機參數優化 (50 次迭代)...")
    results = optimizer.random_search()

    # ===== 打印最優參數 =====
    print("\n" + "=" * 70)
    print("優化結果")
    print("=" * 70)
    print(f"\n最優參數:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")

    print(f"\n最優結果:")
    print(f"  訓練夏普比: {results['best_train_sharpe']:.2f}")
    print(f"  測試夏普比: {results['best_test_sharpe']:.2f}")
    print(f"  OOS/IS 比: {results['best_test_sharpe']/results['best_train_sharpe']:.2f}")

    print(f"\n[OK] 優化完成!")
    print(f"\n下一步:")
    print(f"  1. 進行 Walk-Forward 驗證:")
    print(f"     python strategies/ml_combined/xgboost_combined/test_walkforward_random.py")
    print(f"  2. MCPT 統計檢驗:")
    print(f"     python strategies/ml_combined/xgboost_combined/test_mcpt.py")


if __name__ == '__main__':
    main()
