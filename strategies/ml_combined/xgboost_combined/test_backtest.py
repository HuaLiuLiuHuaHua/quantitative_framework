"""
XGBoost 因子組合 - 基礎回測

驗證訓練好的因子組合是否有預測力
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import matplotlib
matplotlib.use('Agg')

from shared.data_loader import load_local_data
from shared.backtest import BacktestEngine
from strategies.ml_combined.xgboost_combined.strategy import XGBoostCombinedStrategy


def main():
    print("=" * 70)
    print("XGBoost 因子組合 - 基礎回測")
    print("=" * 70)

    # ===== 配置 =====
    SYMBOL = 'BTCUSDT'
    DATA_SOURCE = '1h'
    TEST_START_DATE = '2024-01-01'
    TEST_END_DATE = '2024-12-31'

    # ===== 加載測試數據 =====
    print(f"\n加載測試數據: {SYMBOL} {DATA_SOURCE} ({TEST_START_DATE} ~ {TEST_END_DATE})")
    data = load_local_data(
        symbol=SYMBOL,
        data_source=DATA_SOURCE,
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        lookback_days=100
    )

    if data is None or len(data) == 0:
        print(f"[ERROR] 未找到測試數據")
        return

    print(f"[OK] 加載成功,數據長度: {len(data)}")

    # ===== 初始化策略 =====
    print("\n初始化策略...")
    strategy = XGBoostCombinedStrategy()

    # ===== 生成信號 =====
    print("\n生成信號...")
    signals = strategy.generate_signals(
        data,
        buy_threshold=0.55,
        sell_threshold=0.45
    )

    # 統計信號分佈
    n_long = (signals == 1).sum()
    n_short = (signals == -1).sum()
    n_neutral = (signals == 0).sum()
    n_total = len(signals)

    print(f"信號分佈:")
    print(f"  做多:  {n_long:6d} ({n_long/n_total*100:5.1f}%)")
    print(f"  做空:  {n_short:6d} ({n_short/n_total*100:5.1f}%)")
    print(f"  空倉:  {n_neutral:6d} ({n_neutral/n_total*100:5.1f}%)")

    # 信號質量檢查
    if n_long + n_short < n_total * 0.05:
        print(f"\n[WARNING] 信號太稀疏 ({(n_long+n_short)/n_total*100:.1f}%)")
        print(f"   建議: 降低 buy_threshold/sell_threshold")

    if n_long + n_short > n_total * 0.95:
        print(f"\n[WARNING] 信號過於頻繁 ({(n_long+n_short)/n_total*100:.1f}%)")
        print(f"   建議: 提高 buy_threshold/sell_threshold")

    # ===== 執行回測 =====
    print("\n執行回測...")
    engine = BacktestEngine(
        data=data,
        signals=signals,
        transaction_cost=0.0006,  # 0.06% 交易費用
        slippage=0.0001,          # 0.01% 滑點
        initial_capital=100000,   # 初始資金 100,000
        periods_per_year=8760     # 小時線: 24 * 365
    )

    results = engine.run()

    # ===== 打印結果 =====
    print("\n" + "=" * 70)
    print("回測結果")
    print("=" * 70)
    engine.print_summary()

    # ===== 評估 =====
    print("\n" + "=" * 70)
    print("評估")
    print("=" * 70)

    sharpe = engine.sharpe_ratio
    oos_return = engine.total_return
    max_dd = engine.max_drawdown

    if sharpe > 1.5:
        print(f"[OK] 夏普比率優秀: {sharpe:.2f} (> 1.5)")
    elif sharpe > 1.0:
        print(f"[OK] 夏普比率良好: {sharpe:.2f} (> 1.0)")
    else:
        print(f"[WARNING] 夏普比率偏低: {sharpe:.2f} (< 1.0)")

    if max_dd > -0.30:
        print(f"[OK] 最大回撤可控: {max_dd:.2%} (> -30%)")
    else:
        print(f"[WARNING] 最大回撤過大: {max_dd:.2%} (< -30%)")

    print("\n下一步:")
    print("  1. 如果 Sharpe > 1.5, 可進行參數優化:")
    print("     python strategies/ml_combined/xgboost_combined/test_optimization_random.py")
    print("  2. 進行 Walk-Forward 驗證:")
    print("     python strategies/ml_combined/xgboost_combined/test_walkforward_random.py")
    print("  3. MCPT 統計檢驗:")
    print("     python strategies/ml_combined/xgboost_combined/test_mcpt.py")


if __name__ == '__main__':
    main()
