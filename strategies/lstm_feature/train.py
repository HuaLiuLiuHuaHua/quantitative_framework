"""
LSTM 特徵因子訓練腳本

訓練深度學習因子，支持單/多時間尺度
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

from shared.data_loader import load_local_data
from strategies.lstm_feature.factor import LSTMFeatureFactor


def main():
    print("=" * 70)
    print("LSTM 特徵因子訓練")
    print("=" * 70)

    # ===== 配置 =====
    SYMBOL = 'BTCUSDT'
    DATA_SOURCE = '1h'
    START_DATE = '2020-04-01'
    END_DATE = '2024-04-30'

    # 時間尺度選擇
    # 選項 1: 單時間尺度
    TIME_SCALES = ['1h']

    # 選項 2: 多時間尺度 (解除註釋以使用)
    # TIME_SCALES = ['1h', '4h', '1d']

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
        print(f"[ERROR] 未找到數據,請檢查:")
        print(f"  - 是否有 {SYMBOL} 的數據文件")
        print(f"  - 數據目錄: data/")
        return

    print(f"[OK] 加載成功,數據長度: {len(data)}")
    print(f"   日期範圍: {data.index[0]} ~ {data.index[-1]}")

    # ===== 初始化因子 =====
    print("\n初始化 LSTM 特徵因子...")
    factor = LSTMFeatureFactor(
        sequence_length=60,       # 2.5天（60小時）- 最佳短期預測長度
        hidden_dim=128,           # LSTM隱藏層128維 - 提升模型表達能力
        num_layers=2,             # 2層LSTM - 提供足夠深度學習複雜模式
        output_dim=1,             # 輸出1維（回歸）
        time_scales=TIME_SCALES   # 時間尺度列表
    )

    # ===== 訓練 =====
    print("\n開始訓練...")
    factor.train(
        data=data,
        epochs=150,         # 增加訓練輪次，給模型更多時間學習
        batch_size=64,      # 減小batch size提升訓練穩定性
        val_split=0.2,
        patience=30         # 增加early stopping耐心值
    )

    # ===== 完成 =====
    print("\n" + "=" * 70)
    print("[OK] 訓練完成!")
    print("=" * 70)
    print("\n下一步:")
    print("  1. 運行: python strategies/lstm_feature/test_backtest.py")
    print("     → 驗證因子是否有效")
    print("\n多時間尺度訓練:")
    print("  1. 修改 TIME_SCALES = ['1h', '4h', '1d']")
    print("  2. 重新運行此腳本進行訓練")


if __name__ == '__main__':
    main()
