"""
LSTM 特徵策略 - 基礎回測

驗證訓練好的因子是否有預測力
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import json
import matplotlib
matplotlib.use('Agg')

from shared.data_loader import load_local_data
from shared.backtest import BacktestEngine
from strategies.lstm_feature.strategy import LSTMFeatureStrategy


def main():
    print("=" * 70)
    print("LSTM 特徵策略 - 基礎回測")
    print("=" * 70)

    # ===== 配置 =====
    SYMBOL = 'BTCUSDT'
    DATA_SOURCE = '1h'
    TEST_START_DATE = '2024-05-01'
    TEST_END_DATE = '2025-11-30'

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
    strategy = LSTMFeatureStrategy()

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

    if n_long + n_short > n_total * 0.95:
        print(f"\n[WARNING] 信號過於頻繁 ({(n_long+n_short)/n_total*100:.1f}%)")

    # ===== 執行回測 =====
    print("\n執行回測...")
    engine = BacktestEngine(
        data=data,
        signals=signals,
        transaction_cost=0.0002,  # 0.02% 交易費用
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

    sharpe = engine.metrics['sharpe_ratio']
    oos_return = engine.metrics['total_return']
    max_dd = engine.metrics['max_drawdown']

    print(f"夏普比率: {sharpe:.2f}")
    print(f"總收益率: {oos_return:.2%}")
    print(f"最大回撤: {max_dd:.2%}")

    # ===== 儲存結果 =====
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    backtest_results = {
        'strategy': 'LSTM Feature',
        'symbol': SYMBOL,
        'data_source': DATA_SOURCE,
        'test_period': f"{TEST_START_DATE} ~ {TEST_END_DATE}",
        'metrics': {
            'sharpe_ratio': float(sharpe),
            'total_return': float(oos_return),
            'max_drawdown': float(max_dd),
            'annualized_return': float(engine.metrics['annualized_return']),
            'volatility': float(engine.metrics['volatility']),
            'calmar_ratio': float(engine.metrics['calmar_ratio']),
            'win_rate': float(engine.metrics['win_rate']),
            'profit_factor': float(engine.metrics['profit_factor']),
            'total_trades': int(engine.metrics['total_trades'])
        },
        'signal_distribution': {
            'long_signals': int(n_long),
            'short_signals': int(n_short),
            'neutral_signals': int(n_neutral),
            'long_pct': float(n_long / n_total * 100),
            'short_pct': float(n_short / n_total * 100),
            'neutral_pct': float(n_neutral / n_total * 100)
        }
    }

    result_file = results_dir / f'backtest_results_{SYMBOL}_{DATA_SOURCE}.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(backtest_results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 結果已儲存到: {result_file}")


if __name__ == '__main__':
    main()
