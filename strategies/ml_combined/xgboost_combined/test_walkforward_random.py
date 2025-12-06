"""
XGBoost 因子組合 - Walk-Forward 驗證

使用滾動窗口進行樣本外驗證
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import matplotlib
matplotlib.use('Agg')

from shared.data_loader import load_local_data
from shared.walkforward import WalkForwardOptimizer
from strategies.ml_combined.xgboost_combined.strategy import XGBoostCombinedStrategy


def main():
    print("=" * 70)
    print("XGBoost 因子組合 - Walk-Forward 驗證")
    print("=" * 70)

    # ===== 配置 =====
    SYMBOL = 'BTCUSDT'
    DATA_SOURCE = '1h'
    START_DATE = '2023-01-01'
    END_DATE = '2024-12-31'
    TRAIN_PERIOD = 180  # 訓練期: 180 天
    TEST_PERIOD = 30    # 測試期: 30 天

    # ===== 加載完整數據 =====
    print(f"\n加載數據: {START_DATE} ~ {END_DATE}")
    data = load_local_data(
        symbol=SYMBOL,
        data_source=DATA_SOURCE,
        start_date=START_DATE,
        end_date=END_DATE,
        lookback_days=100
    )

    if data is None or len(data) == 0:
        print(f"[ERROR] 未找到數據")
        return

    print(f"[OK] 數據長度: {len(data)}")

    # ===== 初始化 Walk-Forward 優化器 =====
    print(f"\n初始化 Walk-Forward 優化器...")
    strategy = XGBoostCombinedStrategy()

    wf_optimizer = WalkForwardOptimizer(
        strategy=strategy,
        data=data,
        train_period=TRAIN_PERIOD,
        test_period=TEST_PERIOD,
        n_iter=20,  # 每個窗口 20 次隨機搜索
        n_jobs=-1,
        transaction_cost=0.0006,
        slippage=0.0001,
        initial_capital=100000,
        periods_per_year=8760
    )

    # ===== 執行 Walk-Forward =====
    print(f"\n執行 Walk-Forward 驗證...")
    print(f"  訓練期: {TRAIN_PERIOD} 天")
    print(f"  測試期: {TEST_PERIOD} 天")

    results = wf_optimizer.run()

    # ===== 打印結果 =====
    print("\n" + "=" * 70)
    print("Walk-Forward 結果")
    print("=" * 70)

    print(f"\n整體統計:")
    print(f"  平均訓練夏普比: {results['avg_train_sharpe']:.2f}")
    print(f"  平均測試夏普比: {results['avg_test_sharpe']:.2f}")
    print(f"  平均 OOS/IS 比: {results['avg_oos_is_ratio']:.2f}")
    print(f"  一致性 (OOS 勝率): {results['consistency']:.1f}%")

    print(f"\n窗口詳情:")
    for i, window_result in enumerate(results['window_results']):
        print(f"  窗口 {i+1}: train_sharpe={window_result['train_sharpe']:.2f}, "
              f"test_sharpe={window_result['test_sharpe']:.2f}, "
              f"ratio={window_result['oos_is_ratio']:.2f}")

    print(f"\n[OK] Walk-Forward 驗證完成!")
    print(f"\n下一步:")
    print(f"  1. MCPT 統計檢驗:")
    print(f"     python strategies/ml_combined/xgboost_combined/test_mcpt.py")


if __name__ == '__main__':
    main()
