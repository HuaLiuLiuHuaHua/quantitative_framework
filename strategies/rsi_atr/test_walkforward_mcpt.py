"""
RSI-ATR Strategy - Walk-Forward MCPT
RSI-ATR策略 - Walk-Forward蒙地卡羅排列測試

測試目的：
1. 結合Walk-Forward和MCPT的終極驗證
2. 每個排列都重新優化參數（最嚴格的測試）
3. 評估策略在樣本外的統計顯著性
4. 檢驗是否存在過度擬合

這是最嚴格的回測驗證方法：
- Walk-Forward確保無前視偏差
- MCPT驗證統計顯著性
- 結合兩者提供最可靠的策略評估

配置：
- mode='walkforward': Walk-Forward MCPT模式
- n_permutations=200: 排列次數（較少因為每次要優化）
- wf_config: Walk-Forward配置
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 禁用圖片自動顯示
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import logging

# 配置logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 導入策略和工具
from strategies.rsi_atr.strategy import RSIATRStrategy


def load_data(start_date: str = "2024-01-01", end_date: str = "2024-12-31", verbose: bool = True):
    """
    載入數據 (優先使用本地數據)

    使用統一的data_loader模組,優先從本地載入已下載的數據,
    大幅提升測試速度,避免重複下載。
    """
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
    rsi_window_range: tuple = (10, 20, 2),
    rsi_long_threshold_range: tuple = (55, 70, 5),
    rsi_short_threshold_range: tuple = (30, 45, 5),
    atr_window_range: tuple = (10, 20, 2),
    atr_multiplier_range: tuple = (1.5, 3.0, 0.5),
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
    """
    執行Walk-Forward MCPT測試 (Medium風格)

    這是最嚴格的策略驗證方法：
    1. 使用Walk-Forward避免前視偏差
    2. 使用MCPT驗證統計顯著性
    3. 每個排列都重新優化參數

    Args:
        rsi_window_range: rsi_window參數範圍 (start, end, step)
        rsi_long_threshold_range: rsi_long_threshold參數範圍 (start, end, step)
        rsi_short_threshold_range: rsi_short_threshold參數範圍 (start, end, step)
        atr_window_range: atr_window參數範圍 (start, end, step)
        atr_multiplier_range: atr_multiplier參數範圍 (start, end, step)
        start_date: 開始日期
        end_date: 結束日期
        n_permutations: 排列次數（必須 > 0）
        train_window: 訓練窗口大小（K棒數量），同時用作排列的start_index
        train_step: 重新訓練頻率（K棒數量）
        metric: 評估指標
        transaction_cost: 交易成本 (目前未實現，保留供未來擴展)
        slippage: 滑價 (目前未實現，保留供未來擴展)
        verbose: 是否打印詳細信息

    Returns:
        Dict: Walk-Forward MCPT測試結果字典，包含以下鍵值:
            - real_pf: 真實策略的Profit Factor
            - permuted_pfs: 排列測試的Profit Factor列表
            - p_value: MCPT計算的p值
            - perm_better_count: 優於真實策略的排列數量
            - n_permutations: 請求的排列次數
            - successful_permutations: 成功完成的排列次數
            - save_path: 結果保存路徑

    Note:
        交易成本和滑價參數目前未在walk-forward優化中實現。
        這些參數被保留供未來擴展使用。如需考慮交易成本，
        請在策略的generate_signals方法中實現。
    """
    # 驗證參數
    if n_permutations <= 0:
        raise ValueError(f"n_permutations must be > 0, got {n_permutations}")

    if train_window <= 0:
        raise ValueError(f"train_window must be > 0, got {train_window}")

    if train_step <= 0:
        raise ValueError(f"train_step must be > 0, got {train_step}")

    print(f"Walk Forward MCPT Analysis")
    print(f"Data: {start_date} to {end_date} | Permutations: {n_permutations}")

    # 步驟1: 載入數據
    df = load_data(start_date, end_date, verbose=False)

    # 驗證數據
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Data missing required columns: {missing_columns}")

    if len(df) < train_window:
        raise ValueError(f"Insufficient data: need at least {train_window} rows, got {len(df)}")

    if len(df) < train_window + train_step:
        logging.warning(f"Data length ({len(df)}) is very small for walk-forward with train_window={train_window} and train_step={train_step}")

    # 驗證日期格式
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format (expected YYYY-MM-DD): {e}")

    # 步驟2: 定義策略函數
    def rsi_atr_strategy_func(data: pd.DataFrame, rsi_window: int, rsi_long_threshold: int,
                              rsi_short_threshold: int, atr_window: int, atr_multiplier: float):
        """RSI-ATR策略包裝函數"""
        strategy = RSIATRStrategy()
        signals = strategy.generate_signals(
            data,
            rsi_window=rsi_window,
            rsi_long_threshold=rsi_long_threshold,
            rsi_short_threshold=rsi_short_threshold,
            atr_window=atr_window,
            atr_multiplier=atr_multiplier
        )
        return signals.values

    # 步驟3: 設置參數網格
    param_grid = {
        'rsi_window': (rsi_window_range[0], rsi_window_range[1], rsi_window_range[2]),
        'rsi_long_threshold': (rsi_long_threshold_range[0], rsi_long_threshold_range[1], rsi_long_threshold_range[2]),
        'rsi_short_threshold': (rsi_short_threshold_range[0], rsi_short_threshold_range[1], rsi_short_threshold_range[2]),
        'atr_window': (atr_window_range[0], atr_window_range[1], atr_window_range[2]),
        'atr_multiplier': (atr_multiplier_range[0], atr_multiplier_range[1], atr_multiplier_range[2])
    }

    # 步驟4: 執行真實策略的Walk-Forward優化
    from shared.walkforward_utils import walkforward_optimization

    print("Running Real Strategy...")
    real_signals, real_pf = walkforward_optimization(
        df, rsi_atr_strategy_func, param_grid, train_window, train_step
    )

    print(f"Strategy: PF={real_pf:.3f}")

    # 步驟5: 執行排列測試
    from shared.advanced_permutation import advanced_permutation

    permuted_pfs = []
    perm_better_count = 0  # 計數比真實策略更好的排列

    print(f"Running {n_permutations} permutations...")
    for perm_i in tqdm(range(n_permutations)):
        try:
            # 創建排列數據（使用train_window作為start_index確保一致性）
            perm_df = advanced_permutation(df, start_index=train_window, seed=perm_i)

            # 在排列數據上執行Walk-Forward
            perm_signals, perm_pf = walkforward_optimization(
                perm_df, rsi_atr_strategy_func, param_grid, train_window, train_step
            )

            # 計數優於或等於真實策略的排列
            if perm_pf >= real_pf:
                perm_better_count += 1

            permuted_pfs.append(perm_pf)

        except Exception as e:
            logging.warning(f"Permutation {perm_i} failed: {str(e)}")
            continue

    # 步驟6: 計算p值（使用標準MCPT公式）
    successful_permutations = len(permuted_pfs)
    walkforward_mcpt_pval = (perm_better_count + 1) / (successful_permutations + 1)
    print(f"Walkforward MCPT P-Value: {walkforward_mcpt_pval:.4f} | Successful: {successful_permutations}/{n_permutations}")

    # 步驟7: 保存結果並繪圖（改進命名方式以顯示測試名稱）
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 改進命名：包含測試名稱、策略、數據源、時間戳
    result_dirname = f"WalkforwardMCPT_RSI_ATR_BTC1h_{date_str}"
    output_dir = Path(__file__).parent / "results" / result_dirname
    output_dir.mkdir(exist_ok=True, parents=True)

    # 繪製MCPT分佈圖（Medium風格：深色背景）
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

    # 步驟8: 保存結果到CSV
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
    # 配置參數
    RSI_WINDOW_RANGE = (10, 20, 2)           # rsi_window從10到20，步長2
    RSI_LONG_THRESHOLD_RANGE = (55, 70, 5)   # rsi_long_threshold從55到70，步長5
    RSI_SHORT_THRESHOLD_RANGE = (30, 45, 5)  # rsi_short_threshold從30到45，步長5
    ATR_WINDOW_RANGE = (10, 20, 2)           # atr_window從10到20，步長2
    ATR_MULTIPLIER_RANGE = (1.5, 3.0, 0.5)   # atr_multiplier從1.5到3.0，步長0.5
    START_DATE = "2024-01-01"                # 開始日期
    END_DATE = "2024-12-31"                  # 結束日期
    N_PERMUTATIONS = 200                     # 排列次數
    TRAIN_WINDOW = 365                       # 訓練窗口: 365根K棒（同時作為排列的start_index）
    TRAIN_STEP = 30                          # 重新訓練頻率: 30根K棒
    METRIC = 'profit_factor'                 # 評估指標
    TRANSACTION_COST = 0.002                 # 交易成本 0.2% (目前未實現)
    SLIPPAGE = 0.0005                        # 滑價 0.05% (目前未實現)

    # 執行Walk-Forward MCPT測試
    try:
        results = run_walkforward_mcpt(
            rsi_window_range=RSI_WINDOW_RANGE,
            rsi_long_threshold_range=RSI_LONG_THRESHOLD_RANGE,
            rsi_short_threshold_range=RSI_SHORT_THRESHOLD_RANGE,
            atr_window_range=ATR_WINDOW_RANGE,
            atr_multiplier_range=ATR_MULTIPLIER_RANGE,
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
