"""
XGBoost 因子組合 - MCPT 統計檢驗

使用 Monte Carlo Permutation Test 驗證策略的統計顯著性
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import matplotlib
matplotlib.use('Agg')

from shared.data_loader import load_local_data
from shared.mcpt import MonteCarloPermutationTest
from strategies.ml_combined.xgboost_combined.strategy import XGBoostCombinedStrategy


def main():
    print("=" * 70)
    print("XGBoost 因子組合 - MCPT 統計檢驗")
    print("=" * 70)

    # ===== 配置 =====
    SYMBOL = 'BTCUSDT'
    DATA_SOURCE = '1h'
    TEST_START = '2024-01-01'
    TEST_END = '2024-12-31'
    N_PERMUTATIONS = 1000  # 隨機排列次數

    # ===== 加載數據 =====
    print(f"\n加載測試數據...")
    data = load_local_data(
        symbol=SYMBOL,
        data_source=DATA_SOURCE,
        start_date=TEST_START,
        end_date=TEST_END,
        lookback_days=100
    )

    if data is None or len(data) == 0:
        print(f"[ERROR] 未找到數據")
        return

    print(f"[OK] 數據長度: {len(data)}")

    # ===== 初始化策略 =====
    print(f"\n初始化策略...")
    strategy = XGBoostCombinedStrategy()

    # ===== 執行 MCPT =====
    print(f"\n執行 MCPT ({N_PERMUTATIONS} 次隨機排列)...")
    mcpt = MonteCarloPermutationTest(
        strategy=strategy,
        data=data,
        n_permutations=N_PERMUTATIONS,
        transaction_cost=0.0006,
        slippage=0.0001,
        initial_capital=100000,
        periods_per_year=8760,
        n_jobs=-1
    )

    results = mcpt.run()

    # ===== 打印結果 =====
    print("\n" + "=" * 70)
    print("MCPT 結果")
    print("=" * 70)

    print(f"\n原始策略性能:")
    print(f"  夏普比率: {results['original_sharpe']:.2f}")
    print(f"  總收益: {results['original_return']:.2%}")
    print(f"  最大回撤: {results['original_max_dd']:.2%}")

    print(f"\n隨機排列統計:")
    print(f"  平均夏普比率: {results['perm_mean_sharpe']:.2f}")
    print(f"  標準差: {results['perm_std_sharpe']:.2f}")
    print(f"  最大夏普比率: {results['perm_max_sharpe']:.2f}")

    print(f"\n統計顯著性 (p-value):")
    print(f"  p-value: {results['p_value']:.4f}")

    if results['p_value'] < 0.05:
        print(f"  [OK] 結果在 95% 信心水平上顯著! (p < 0.05)")
    else:
        print(f"  [WARNING] 結果不具統計顯著性 (p >= 0.05)")

    print(f"\nPercentile:")
    print(f"  策略排名: {results['percentile']:.1f}% (vs. 隨機排列)")

    print(f"\n[OK] MCPT 檢驗完成!")


if __name__ == '__main__':
    main()
