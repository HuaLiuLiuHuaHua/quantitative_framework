"""
決策樹組合策略訓練腳本

執行此腳本訓練決策樹模型並保存
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

from shared.data_loader import load_local_data
from strategies.tree_combined.factor import TreeCombinedFactor


def main():
    print("=" * 70)
    print("決策樹組合策略訓練")
    print("=" * 70)

    # 配置
    SYMBOL = 'BTCUSDT'
    DATA_SOURCE = '1h'
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'

    # 加載訓練數據
    print(f"\n加載數據: {SYMBOL} {DATA_SOURCE} ({START_DATE} ~ {END_DATE})")
    data = load_local_data(
        symbol=SYMBOL,
        data_source=DATA_SOURCE,
        start_date=START_DATE,
        end_date=END_DATE,
        lookback_days=100
    )

    if data is None or len(data) == 0:
        print(f"❌ 未找到數據,請檢查:")
        print(f"  - 是否有 {SYMBOL} 的數據文件")
        print(f"  - 數據目錄: c:\\Users\\liual\\OneDrive\\桌面\\quantitative_framework\\data\\")
        return

    print(f"✅ 加載成功,數據長度: {len(data)}")

    # 初始化因子
    print("\n初始化決策樹因子...")
    factor = TreeCombinedFactor(max_depth=5)

    # 訓練
    print("\n開始訓練...")
    factor.train(data=data)

    print("\n" + "=" * 70)
    print("✅ 訓練完成!")
    print("=" * 70)
    print("\n下一步:")
    print("  1. 運行: python strategies/tree_combined/test_backtest.py")
    print("  2. 驗證策略是否能正確使用因子")


if __name__ == '__main__':
    main()
