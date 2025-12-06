"""
XGBoost 因子組合訓練腳本

訓練 XGBoost 模型來組合多個因子：
- 手動設計的技術指標因子
- LSTM 深度學習因子
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

from shared.data_loader import load_local_data
from strategies.ml_combined.xgboost_combined.factor import XGBoostCombinedFactor


def main():
    print("=" * 70)
    print("XGBoost 因子組合訓練")
    print("=" * 70)

    # ===== 配置 =====
    SYMBOL = 'BTCUSDT'
    DATA_SOURCE = '1h'
    START_DATE = '2022-12-01'
    END_DATE = '2025-01-31'

    # ===== 加載數據 =====
    print(f"\n加載數據: {SYMBOL} {DATA_SOURCE} ({START_DATE} ~ {END_DATE})")
    data = load_local_data(
        symbol=SYMBOL,
        data_source=DATA_SOURCE,
        start_date=START_DATE,
        end_date=END_DATE,
        lookback_days=100
    )

    if data is None or len(data) == 0:
        raise RuntimeError(
            f"未找到數據,請檢查:\n"
            f"  - 是否有 {SYMBOL} 的數據文件\n"
            f"  - 數據目錄: data/\n"
            f"  - 日期範圍: {START_DATE} ~ {END_DATE}"
        )

    print(f"[OK] 加載成功,數據長度: {len(data)}")
    print(f"   日期範圍: {data.index[0]} ~ {data.index[-1]}")

    # ===== 初始化因子 =====
    print("\n初始化 XGBoost 因子組合...")
    factor = XGBoostCombinedFactor(
        max_depth=5,          # 樹深度
        n_estimators=100,     # 樹數量
        learning_rate=0.1     # 學習率
    )

    # ===== 訓練 =====
    print("\n開始訓練...")
    factor.train(
        data=data,
        use_lstm=True,           # 包含 LSTM 因子
        future_return_period=5,  # 預測未來 5 期
        min_return_threshold=0.02  # 定義上漲為 > 2% 收益
    )

    # ===== 完成 =====
    print("\n" + "=" * 70)
    print("[OK] 訓練完成!")
    print("=" * 70)
    print("\n下一步:")
    print("  1. 運行: python strategies/ml_combined/xgboost_combined/test_backtest.py")
    print("     → 驗證組合因子是否有效")
    print("\n優化建議:")
    print("  1. 如果 Sharpe > 1.5, 可進行參數優化:")
    print("     python strategies/ml_combined/xgboost_combined/test_optimization_random.py")
    print("  2. 進行 Walk-Forward 驗證:")
    print("     python strategies/ml_combined/xgboost_combined/test_walkforward_random.py")


if __name__ == '__main__':
    main()
