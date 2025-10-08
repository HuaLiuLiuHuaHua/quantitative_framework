"""
BB-ATR Strategy - Walk-Forward MCPT
BB-ATR策略 - Walk-Forward蒙地卡羅排列測試

測試目的：
1. 結合Walk-Forward和MCPT的終極驗證
2. 每個排列都重新優化參數（最嚴格的測試）
3. 評估策略在樣本外的統計顯著性
4. 檢驗是否存在過度擬合

這是最嚴格的回測驗證方法：
- Walk-Forward確保無前視偏差
- MCPT驗證統計顯著性
- 結合兩者提供最可靠的策略評估
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from strategies.bb_atr.strategy import BBATRStrategy


def load_data(start_date: str = "2024-01-01", end_date: str = "2024-12-31", verbose: bool = True):
    from shared.data_loader import load_data_with_fallback
    from data_fetchers.bybit_btc_1h_fetcher import fetch_bybit_btc_1h_data

    return load_data_with_fallback(
        data_source="1h",
        start_date=start_date,
        end_date=end_date,
        fetcher_func=fetch_bybit_btc_1h_data,
        verbose=verbose
    )


def run_walkforward_mcpt(
    bb_window_range: tuple, bb_std_range: tuple, atr_window_range: tuple, atr_multiplier_range: tuple,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    n_permutations: int = 200,
    train_window: int = 365,
    train_step: int = 30,
    metric: str = 'profit_factor',
    transaction_cost: float = 0.002,
    slippage: float = 0.0005,
    verbose: bool = True
):
    if n_permutations <= 0:
        raise ValueError(f"n_permutations must be > 0, got {n_permutations}")
    if train_window <= 0:
        raise ValueError(f"train_window must be > 0, got {train_window}")
    if train_step <= 0:
        raise ValueError(f"train_step must be > 0, got {train_step}")

    print(f"Walk Forward MCPT Analysis")
    print(f"Data: {start_date} to {end_date} | Permutations: {n_permutations}")

    df = load_data(start_date, end_date, verbose=False)

    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Data missing required columns: {missing_columns}")

    if len(df) < train_window:
        raise ValueError(f"Insufficient data: need at least {train_window} rows, got {len(df)}")

    def bb_atr_strategy_func(data: pd.DataFrame, bb_window: int, bb_std: float, atr_window: int, atr_multiplier: float):
        strategy = BBATRStrategy()
        signals = strategy.generate_signals(
            data,
            bb_window=bb_window,
            bb_std=bb_std,
            atr_window=atr_window,
            atr_multiplier=atr_multiplier
        )
        return signals.values

    param_grid = {
        'bb_window': (15, 25, 5),
        'bb_std': (1.5, 2.5, 0.5),
        'atr_window': (10, 20, 2),
        'atr_multiplier': (1.5, 3.0, 0.5)
    }

    from shared.walkforward_utils import walkforward_optimization

    print("Running Real Strategy...")
    real_signals, real_pf = walkforward_optimization(
        df, bb_atr_strategy_func, param_grid, train_window, train_step
    )

    print(f"Strategy: PF={real_pf:.3f}")

    from shared.advanced_permutation import advanced_permutation

    permuted_pfs = []
    perm_better_count = 0

    print(f"Running {n_permutations} permutations...")
    for perm_i in tqdm(range(n_permutations)):
        try:
            perm_df = advanced_permutation(df, start_index=train_window, seed=perm_i)
            perm_signals, perm_pf = walkforward_optimization(
                perm_df, bb_atr_strategy_func, param_grid, train_window, train_step
            )

            if perm_pf >= real_pf:
                perm_better_count += 1

            permuted_pfs.append(perm_pf)

        except Exception as e:
            logging.warning(f"Permutation {perm_i} failed: {str(e)}")
            continue

    successful_permutations = len(permuted_pfs)
    walkforward_mcpt_pval = (perm_better_count + 1) / (successful_permutations + 1)
    print(f"Walkforward MCPT P-Value: {walkforward_mcpt_pval:.4f} | Successful: {successful_permutations}/{n_permutations}")

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dirname = f"WalkforwardMCPT_BB_ATR_BTC1h_{date_str}"
    output_dir = Path(__file__).parent / "results" / result_dirname
    output_dir.mkdir(exist_ok=True, parents=True)

    try:
        plt.style.use('dark_background')
    except Exception as e:
        logging.warning(f"Failed to set dark_background style: {e}. Using default style.")

    pd.Series(permuted_pfs).hist(color='blue', label='Permutations', bins=40)
    plt.axvline(real_pf, color='red', linewidth=2, label='Real Strategy')
    plt.xlabel("Profit Factor")
    plt.ylabel('Frequency')
    plt.title(f"Walkforward MCPT. P-Value: {walkforward_mcpt_pval:.4f}")
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mcpt_distribution.png", dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

    print(f"Charts saved to: {output_dir}")

    results_df = pd.DataFrame({
        'metric': ['profit_factor', 'p_value', 'successful_permutations', 'requested_permutations'],
        'value': [real_pf, walkforward_mcpt_pval, successful_permutations, n_permutations]
    })
    results_df.to_csv(output_dir / "mcpt_results.csv", index=False)

    return {
        'real_pf': real_pf,
        'permuted_pfs': permuted_pfs,
        'p_value': walkforward_mcpt_pval,
        'perm_better_count': perm_better_count,
        'n_permutations': n_permutations,
        'successful_permutations': successful_permutations,
        'save_path': output_dir
    }


if __name__ == "__main__":
    BB_WINDOW_RANGE = (15, 25, 5)
    BB_STD_RANGE = (1.5, 2.5, 0.5)
    ATR_WINDOW_RANGE = (10, 20, 2)
    ATR_MULTIPLIER_RANGE = (1.5, 3.0, 0.5)
    START_DATE = "2024-01-01"
    END_DATE = "2024-12-31"
    N_PERMUTATIONS = 200
    TRAIN_WINDOW = 365
    TRAIN_STEP = 30
    METRIC = 'profit_factor'
    TRANSACTION_COST = 0.002
    SLIPPAGE = 0.0005

    try:
        results = run_walkforward_mcpt(
            bb_window_range=BB_WINDOW_RANGE, bb_std_range=BB_STD_RANGE, atr_window_range=ATR_WINDOW_RANGE, atr_multiplier_range=ATR_MULTIPLIER_RANGE,
            start_date=START_DATE,
            end_date=END_DATE,
            n_permutations=N_PERMUTATIONS,
            train_window=TRAIN_WINDOW,
            train_step=TRAIN_STEP,
            metric=METRIC,
            transaction_cost=TRANSACTION_COST,
            slippage=SLIPPAGE,
            verbose=True
        )
    except Exception as e:
        print(f"\n[ERROR] 執行失敗: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
