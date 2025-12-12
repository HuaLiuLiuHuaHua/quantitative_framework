"""
MA交叉策略 - 基礎回測

驗證策略邏輯是否正確
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

from shared.data_loader import load_local_data
from shared.backtest import BacktestEngine
from strategies.ma_cross.strategy import MACrossStrategy
import matplotlib
matplotlib.use('Agg')


def main():
    print("=" * 70)
    print("MA交叉策略 - 基礎回測")
    print("=" * 70)

    # 配置
    SYMBOL = 'BTCUSDT'
    DATA_SOURCE = '1h'
    TEST_START_DATE = '2024-01-01'
    TEST_END_DATE = '2024-12-31'

    # 加載測試數據
    print(f"\n加載測試數據: {SYMBOL} {DATA_SOURCE} ({TEST_START_DATE} ~ {TEST_END_DATE})")
    data = load_local_data(
        symbol=SYMBOL,
        data_source=DATA_SOURCE,
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        lookback_days=100
    )

    if data is None or len(data) == 0:
        print(f"❌ 未找到測試數據")
        return

    print(f"✅ 加載成功,數據長度: {len(data)}")

    # 初始化策略
    print("\n初始化策略...")
    strategy = MACrossStrategy()

    # 生成信號
    print("\n生成信號...")
    signals = strategy.generate_signals(data, short_window=20, long_window=50)

    # 統計信號
    n_long = (signals == 1).sum()
    n_short = (signals == -1).sum()
    n_neutral = (signals == 0).sum()

    print(f"信號分佈:")
    print(f"  做多: {n_long} ({n_long/len(signals)*100:.1f}%)")
    print(f"  做空: {n_short} ({n_short/len(signals)*100:.1f}%)")
    print(f"  空倉: {n_neutral} ({n_neutral/len(signals)*100:.1f}%)")

    # 執行回測
    print("\n執行回測...")
    engine = BacktestEngine(
        data=data,
        signals=signals,
        transaction_cost=0.0006,  # 0.06% 交易費
        slippage=0.0001,          # 0.01% 滑點
        initial_capital=100000,   # 初始資金100k
        periods_per_year=8760     # 小時線
    )

    results = engine.run()

    # 打印結果
    print("\n" + "=" * 70)
    print("回測結果")
    print("=" * 70)

    engine.print_summary()

    # 保存結果 (可選)
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)

    print(f"\n✅ 回測完成!")
    print(f"結果保存到: {output_dir}")


if __name__ == '__main__':
    main()
