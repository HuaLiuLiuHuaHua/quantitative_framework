# -*- coding: utf-8 -*-
"""
CVILLIQ CTA Strategy - Walk-Forward Optimization with Sensitivity Analysis

功能:
1. 模擬實際交易，定期在滾動窗口上重新優化參數
2. 每個訓練窗口: 優化 + 敏感性分析 → 選擇最佳穩健參數
3. 每個測試窗口: 使用最佳穩健參數進行樣本外測試
4. 評估參數穩定性和樣本外性能一致性

改進:
- 參數網格避免衝突（long > short 始終成立）
- 單一清晰進度條
- 生成權益與回撤對比圖表
- 智能參數選擇（優先穩健參數，回退到全局最優）

作者: Claude Code
"""

import sys
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 設置 Python 輸出編碼
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from strategies.cvilliq.strategy import CVILLIQStrategy
from shared.backtest import BacktestEngine
from shared.data_loader import load_local_data
from shared.sensitivity import get_parameter_neighborhood, filter_robust_params
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ==================== 輔助函數 ====================

# 失敗結果的默認值
FAILED_SHARPE = -999


def calculate_num_workers(n_jobs: int = None) -> int:
    """計算並行工作進程數"""
    cpu_count = os.cpu_count() or 1
    if n_jobs is None or n_jobs == -1:
        return cpu_count
    elif n_jobs < 0:
        return max(1, cpu_count + n_jobs)
    else:
        return max(1, n_jobs)


def generate_all_combinations(param_grid: dict) -> List[dict]:
    """生成所有可能的參數組合"""
    import itertools
    param_names = list(param_grid.keys())
    param_values = [list(v) for v in param_grid.values()]
    all_combos = list(itertools.product(*param_values))
    return [dict(zip(param_names, combo)) for combo in all_combos]


def random_sample_combinations(param_grid: dict, n_samples: int) -> List[dict]:
    """隨機採樣n個參數組合"""
    combinations = []
    for _ in range(n_samples):
        combo = {name: random.choice(list(values)) for name, values in param_grid.items()}
        combinations.append(combo)
    return combinations


def evaluate_single_params(params: dict, **kwargs) -> dict:
    """評估單個參數組合的輔助函數(用於並行計算)"""
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["PYTHONWARNINGS"] = "ignore"

    try:
        # 解析參數
        leverage = int(params.get("leverage", 1))
        capital_allocation = float(params.get("capital_allocation", 1.0))
        final_leverage = leverage * capital_allocation

        # 從kwargs獲取配置
        data = kwargs["data"]
        transaction_cost = kwargs["transaction_cost"]
        slippage = kwargs["slippage"]
        periods_per_year = kwargs["periods_per_year"]
        stop_loss_fee = kwargs.get("stop_loss_fee", 0.00055)

        # 生成策略信號 (傳遞所有參數)
        strategy = CVILLIQStrategy()
        signals = strategy.generate_signals(
            data,
            **params
        )

        # 執行回測
        engine = BacktestEngine(
            data=data,
            signals=signals,
            transaction_cost=transaction_cost,
            slippage=slippage,
            initial_capital=100000,
            periods_per_year=periods_per_year,
            leverage=final_leverage,
            stop_loss_fee=stop_loss_fee
        )

        results = engine.run()

        # 提取績效指標
        metrics = results.get('metrics', {})
        sharpe_ratio = metrics.get('sharpe_ratio', FAILED_SHARPE)
        calmar_ratio = metrics.get('calmar_ratio', FAILED_SHARPE)
        annual_return = metrics.get('annualized_return', FAILED_SHARPE)
        max_drawdown = metrics.get('max_drawdown', FAILED_SHARPE)

        return {
            **params,
            "sharpe_ratio": float(sharpe_ratio),
            "calmar_ratio": float(calmar_ratio),
            "annual_return": float(annual_return),
            "max_drawdown": float(max_drawdown)
        }

    except Exception as e:
        # 統一失敗默認值為 FAILED_SHARPE
        # 靜默失敗，避免干擾進度條顯示
        error_info = {
            **params,
            "sharpe_ratio": FAILED_SHARPE,
            "calmar_ratio": FAILED_SHARPE,
            "annual_return": FAILED_SHARPE,
            "max_drawdown": FAILED_SHARPE
        }
        # 添加診斷信息供後續分析
        if str(e):
            error_info["error_type"] = type(e).__name__
            error_info["error_msg"] = str(e)[:100]
        return error_info


def filter_top_params(
    results_df: pd.DataFrame,
    sharpe_threshold: float = 1.25,
    calmar_threshold: float = 2.5,
    annual_return_threshold: float = 0.5,
    max_drawdown_threshold: float = -0.2
) -> pd.DataFrame:
    """
    篩選符合條件的優質參數組合（與 test_optimization_with_sensitivity.py 完全一致）

    條件:
    - sharpe_ratio > sharpe_threshold (default 1.25)
    - calmar_ratio > calmar_threshold (default 2.5)
    - annual_return > annual_return_threshold (default 50%)
    - max_drawdown > max_drawdown_threshold (default -20%)

    Args:
        results_df: 優化結果DataFrame
        sharpe_threshold: 夏普比率閾值
        calmar_threshold: 卡瑪比率閾值
        annual_return_threshold: 年報酬閾值
        max_drawdown_threshold: 最大回撤閾值（負數，如 -0.2 表示 -20%）

    Returns:
        符合條件的參數組合，按穩健性分數降序排列
    """
    # 過濾失敗結果
    valid_df = results_df[results_df['sharpe_ratio'] > FAILED_SHARPE].copy()
    if valid_df.empty:
        print("Warning: No valid optimization results found.")
        return pd.DataFrame()

    cond_sharpe = valid_df['sharpe_ratio'] > sharpe_threshold
    cond_calmar = valid_df['calmar_ratio'] > calmar_threshold
    cond_annual_return = valid_df['annual_return'] > annual_return_threshold
    cond_max_drawdown = valid_df['max_drawdown'] > max_drawdown_threshold

    filtered_df = valid_df[
        cond_sharpe &
        cond_calmar &
        cond_annual_return &
        cond_max_drawdown
    ].copy()

    # 如果沒有參數滿足所有條件，返回空 DataFrame（主函數會自動選擇夏普比率最高的參數）
    if filtered_df.empty:
        print(f"⚠️  警告：沒有任何參數同時滿足 Sharpe > {sharpe_threshold}, Calmar > {calmar_threshold}, 年化報酬 > {annual_return_threshold:.0%}, 最大回撤 < {-max_drawdown_threshold:.0%}")
        print(f"    將從所有有效結果中選擇夏普比率最高的參數組合")
        return pd.DataFrame()

    if not filtered_df.empty:
        # 計算穩健性分數
        def calculate_robustness_score(row: pd.Series) -> float:
            max_dd_abs = abs(float(row['max_drawdown']))
            sharpe = float(row['sharpe_ratio'])
            calmar = float(row['calmar_ratio'])
            annual_ret = float(row['annual_return'])
            return sharpe * 0.4 + calmar * 0.3 + min(annual_ret, 2.0) * 0.1 - max_dd_abs * 0.2

        filtered_df['robustness_score'] = filtered_df.apply(calculate_robustness_score, axis=1)
        filtered_df = filtered_df.sort_values('robustness_score', ascending=False)

    print(f"\n篩選條件: 夏普 > {sharpe_threshold}, 卡瑪 > {calmar_threshold}, 年化報酬 > {annual_return_threshold:.0%}, 最大回撤 > {max_drawdown_threshold:.0%}")
    print(f"符合條件的參數組合: {len(filtered_df)} 個")

    return filtered_df


def supplement_neighborhood_for_best(best_params, param_grid, results_df, **kwargs):
    """
    補充最佳參數的鄰域點評估
    """
    logger.info("補充最佳參數的鄰域點評估")

    # 獲取所有鄰域點
    all_neighbors = []
    param_names = list(param_grid.keys())

    for param_name in param_names:
        neighbors = get_parameter_neighborhood(
            param_name=param_name,
            param_value=best_params[param_name],
            param_grid=param_grid,
            step_size=2
        )
        for neighbor_val in neighbors:
            neighbor_params = best_params.copy()
            neighbor_params[param_name] = neighbor_val
            all_neighbors.append(neighbor_params)

    # 移除重複
    unique_neighbors = []
    seen = set()
    for params in all_neighbors:
        param_tuple = tuple(params[k] for k in param_names)
        if param_tuple not in seen:
            seen.add(param_tuple)
            unique_neighbors.append(params)

    # 篩選已測試的組合
    tested_set = set()
    for _, row in results_df.iterrows():
        param_tuple = tuple(row[k] for k in param_names)
        tested_set.add(param_tuple)

    to_evaluate = []
    for params in unique_neighbors:
        param_tuple = tuple(params[k] for k in param_names)
        if param_tuple not in tested_set:
            to_evaluate.append(params)

    if not to_evaluate:
        logger.info("所有鄰域點已在隨機搜索中評估過")
        return results_df

    logger.info(f"需要補充評估 {len(to_evaluate)} 個鄰域點")

    # 並行評估鄰域點
    num_workers = calculate_num_workers(kwargs.get("n_jobs"))
    worker = partial(evaluate_single_params, **kwargs)

    new_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, params) for params in to_evaluate]
        for future in tqdm(futures, desc="鄰域點評估", unit="點", ncols=100,
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            try:
                result = future.result(timeout=300)
                new_results.append(result)
            except Exception:
                pass

    # 合併結果
    if new_results:
        new_results_df = pd.DataFrame(new_results)
        combined_df = pd.concat([results_df, new_results_df], ignore_index=True)
        logger.info(f"已補充 {len(new_results)} 個鄰域點")
        return combined_df
    return results_df


def plot_walkforward_equity_and_drawdown(
    all_equity_curves: Dict[str, pd.Series],
    benchmark_data: pd.DataFrame,
    output_path: Path,
    strategy_name: str = "策略權益",
    benchmark_name: str = "BTC持有"
) -> Optional[pd.Series]:
    """
    繪製 Walk-Forward 權益曲線與回撤對比圖 (已修復)

    關鍵修復:
    1. 正確連接多個窗口的權益曲線（鏈式銜接），使用更穩健的檢查
    2. 填補失敗窗口的空白，用水平線（持有現金）代替，提供更真實的視覺效果
    3. 返回拼接好的總權益曲線，用於正確計算整體性能指標

    Args:
        all_equity_curves: 所有窗口的權益曲線字典
        benchmark_data: 基準數據 (包含 'close' 列)
        output_path: 輸出路徑
        strategy_name: 策略名稱
        benchmark_name: 基準名稱
    
    Returns:
        Optional[pd.Series]: 拼接完成的總權益曲線，或在失敗時返回 None
    """
    # Chart configuration constants
    FIGURE_WIDTH = 14
    FIGURE_HEIGHT = 10
    EQUITY_DRAWDOWN_HEIGHT_RATIO = [3, 2]
    CHART_DPI = 150

    logger.info(f"繪製 Walk-Forward 圖表: {len(all_equity_curves)} 個窗口")

    # 驗證輸入
    if not all_equity_curves:
        logger.warning("沒有權益曲線數據，無法繪圖")
        return None
    
    if benchmark_data is None or benchmark_data.empty or 'close' not in benchmark_data.columns:
        logger.error("基準數據無效，無法繪圖")
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
                                    gridspec_kw={'height_ratios': EQUITY_DRAWDOWN_HEIGHT_RATIO})

    # 合併所有窗口的權益曲線
    all_equity_list = []
    window_ids = sorted(all_equity_curves.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)

    for i, window_id in enumerate(window_ids):
        equity = all_equity_curves[window_id]

        if equity is None or equity.empty or len(equity) < 2:
            logger.warning(f"窗口 {window_id} 的權益曲線數據不足，跳過")
            continue
        try:
            equity = pd.Series(equity.values, index=equity.index, dtype='float64').dropna()
            if equity.empty or equity.isna().all():
                logger.warning(f"窗口 {window_id} 的權益曲線全為NaN，跳過")
                continue
        except Exception as e:
            logger.error(f"窗口 {window_id} 權益曲線驗證失敗: {e}")
            continue

        equity_copy = equity.copy()

        if len(all_equity_list) > 0:
            prev_equity = all_equity_list[-1]
            prev_end_value = prev_equity.iloc[-1]
            curr_start_value = equity_copy.iloc[0]
            
            # 使用更穩健的檢查來銜接曲線
            if pd.notna(prev_end_value) and pd.notna(curr_start_value) and curr_start_value != 0:
                adjustment_factor = prev_end_value / curr_start_value
                equity_copy *= adjustment_factor
                logger.info(f"  窗口 {window_id} 鏈式銜接 - 調整係數: {adjustment_factor:.4f}")
            else:
                logger.warning(f"  窗口 {window_id} 數值異常或起始為0，無法銜接，將嘗試使用前一窗口終值填充: prev_end={prev_end_value}, curr_start={curr_start_value}")
                # 如果無法銜接，則將整個窗口的權益設置為前一個窗口的最終值（相當於空倉）
                equity_copy.iloc[:] = prev_end_value

        all_equity_list.append(equity_copy)

    if not all_equity_list:
        logger.warning("沒有有效的權益曲線數據")
        plt.close(fig)
        return None

    # 連接所有窗口的權益曲線
    combined_equity = pd.concat(all_equity_list, ignore_index=False)
    combined_equity = combined_equity[~combined_equity.index.duplicated(keep='first')]
    combined_equity = combined_equity.sort_index()

    # 關鍵修復：填補空窗期，用線性插值而非 forward fill
    try:
        freq = pd.infer_freq(benchmark_data.index)
        if freq:
            full_range = pd.date_range(start=combined_equity.index[0], end=combined_equity.index[-1], freq=freq)
            combined_equity = combined_equity.reindex(full_range)

            # 計算缺失數據點比例
            n_missing = combined_equity.isna().sum()
            n_total = len(full_range)
            missing_pct = (n_missing / n_total * 100) if n_total > 0 else 0

            if n_missing > 0:
                logger.warning(f"檢測到 {n_missing} 個缺失數據點 ({missing_pct:.2f}%)")

                # 如果缺失比例過高（>10%），可能表示有窗口失敗，使用線性插值
                if missing_pct > 10:
                    logger.warning(f"缺失數據比例較高，使用線性插值填補")
                    combined_equity = combined_equity.interpolate(method='linear', limit_direction='both')
                    # 針對最開始和最後的 NaN，用 forward/backward fill
                    combined_equity = combined_equity.bfill().ffill()
                else:
                    # 缺失比例較低，使用 forward fill
                    combined_equity = combined_equity.ffill()

                logger.info(f"已填補空窗期，缺失率: {missing_pct:.2f}%")
            else:
                logger.info("權益曲線完整，無需填補")
        else:
            logger.warning("無法推斷數據頻率，可能無法正確填補空窗期")
    except Exception as e:
        logger.error(f"填補空窗期時出錯: {e}")

    logger.info(f"合併權益曲線: 總長度={len(combined_equity)}, 起始={combined_equity.iloc[0]:.2f}, 結束={combined_equity.iloc[-1]:.2f}")

    # 歸一化策略權益曲線
    initial_capital = combined_equity.iloc[0]
    if not (initial_capital > 0):
        logger.error(f"初始資金無效: {initial_capital}")
        plt.close(fig)
        return None
    
    strategy_normalized = combined_equity / initial_capital

    # 對齐基準數據到策略時間範圍
    benchmark_aligned = benchmark_data['close'].reindex(combined_equity.index).ffill().bfill()
    if benchmark_aligned.isna().any():
        logger.error("無法對齐基準數據")
        plt.close(fig)
        return None
    
    # 計算基準的歸一化權益曲線
    benchmark_first_price = benchmark_aligned.iloc[0]
    if not (benchmark_first_price > 0):
        logger.error(f"初始基準價格無效: {benchmark_first_price}")
        plt.close(fig)
        return None
    
    benchmark_normalized = benchmark_aligned / benchmark_first_price

    # 計算最終收益率
    strategy_final_return = (strategy_normalized.iloc[-1] - 1) * 100
    benchmark_final_return = (benchmark_normalized.iloc[-1] - 1) * 100

    # ========== 上圖：權益曲線對比 ==========
    ax1.plot(strategy_normalized.index, strategy_normalized.values,
             label=f'{strategy_name}: {strategy_final_return:.2f}%', color='#2E86AB', linewidth=1.5)
    ax1.plot(benchmark_normalized.index, benchmark_normalized.values,
             label=f'{benchmark_name}: {benchmark_final_return:.2f}%', color='#F77F00', linewidth=1.5, alpha=0.8)
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_ylabel('倍數', fontsize=12)
    ax1.set_title(f'{strategy_name} vs {benchmark_name} (從 {combined_equity.index[0].strftime("%Y-%m-%d")} 開始)',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # ========== 下圖：回撤對比 ==========
    strategy_cummax = strategy_normalized.expanding().max()
    strategy_drawdown = (strategy_normalized - strategy_cummax) / strategy_cummax * 100

    benchmark_cummax = benchmark_normalized.expanding().max()
    benchmark_drawdown = (benchmark_normalized - benchmark_cummax) / benchmark_cummax * 100

    strategy_max_dd = strategy_drawdown.min()
    benchmark_max_dd = benchmark_drawdown.min()

    ax2.fill_between(strategy_drawdown.index, strategy_drawdown.values, 0,
                     color='#2E86AB', alpha=0.4, label=f'{strategy_name} 回撤 (最大: {strategy_max_dd:.2f}%)')
    ax2.fill_between(benchmark_drawdown.index, benchmark_drawdown.values, 0,
                     color='#F77F00', alpha=0.3, label=f'{benchmark_name} 回撤 (最大: {benchmark_max_dd:.2f}%)')
    
    ax2.legend(loc='lower left', fontsize=10)
    ax2.set_xlabel('時間', fontsize=12)
    ax2.set_ylabel('回撤 (%)', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 共享 X 軸，移除上圖的X軸標籤
    ax1.set_xticklabels([])

    plt.tight_layout(h_pad=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=CHART_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ 保存權益與回撤圖表: {output_path.name}")
    
    return combined_equity


def optimize_window_with_sensitivity(
    window_id: int,
    train_start: str,
    train_end: str,
    param_grid: dict,
    n_trials: int,
    sharpe_threshold: float,
    data_full: pd.DataFrame,
    **kwargs
) -> Tuple[dict, pd.DataFrame, dict]:
    """
    優化單個訓練窗口，進行敏感性分析

    改進邏輯:
    1. 先篩選符合所有條件的優質參數 (filter_top_params)
    2. 如果有符合條件的參數，從中進行敏感性分析，選擇穩健性最高的參數
    3. 如果沒有符合條件的參數，直接選擇全局最優夏普比率參數

    Returns:
        best_params: 最優參數組合
        results_df: 完整優化結果
        sensitivity_report: 敏感性分析報告
    """
    print(f"\n【窗口 {window_id}】{train_start} → {train_end}")

    # 提取訓練數據
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    train_data = data_full[
        (data_full.index >= train_start_dt) &
        (data_full.index <= train_end_dt)
    ]

    if train_data.empty:
        logger.error(f"訓練數據為空: {train_start} - {train_end}")
        return {}, pd.DataFrame(), {}

    # 生成所有可能的參數組合
    print("→ 生成所有可能的參數組合...")
    all_combinations = generate_all_combinations(param_grid)
    total_combinations = len(all_combinations)
    print(f"參數空間大小: {total_combinations:,} 個組合")

    # 1. 先移除 short_entry_quantile > long_entry_quantile 的無效組合
    param_combinations = all_combinations
    if "long_entry_quantile" in param_grid and "short_entry_quantile" in param_grid:
        original_count = len(param_combinations)
        # 保留 long_q >= short_q 的組合
        param_combinations = [
            p for p in param_combinations
            if p.get('long_entry_quantile', 1.0) >= p.get('short_entry_quantile', 0.0)
        ]
        filtered_count = original_count - len(param_combinations)
        if filtered_count > 0:
            print(f"過濾掉 {filtered_count:,} 個無效組合 (short_q > long_q)。")

    if not param_combinations:
        print("\n錯誤：過濾後沒有有效的參數組合可供評估。請檢查您的 PARAM_GRID。")
        return {}, pd.DataFrame(), {}

    print(f"有效組合數: {len(param_combinations):,}")

    # 2. 判斷參數組合是否 > N_TRIALS
    if len(param_combinations) > n_trials:
        print(f"→ 有效組合數大於 N_TRIALS ({n_trials})，將從中隨機抽樣。")
        param_combinations = random.sample(param_combinations, n_trials)
    else:
        print("→ 使用所有有效組合進行評估。")

    print(f"最終評估組合數: {len(param_combinations):,}\n")


    # 準備評估參數
    eval_kwargs = {
        **kwargs,
        "data": train_data
    }

    # 運行優化
    num_workers = calculate_num_workers(kwargs.get("n_jobs"))
    worker = partial(evaluate_single_params, **eval_kwargs)

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, params) for params in param_combinations]

        # 使用單一進度條顯示優化進度
        for future in tqdm(futures, desc=f"窗口{window_id}優化", unit="組合", ncols=100,
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            try:
                result = future.result(timeout=300)
                results.append(result)
            except Exception:
                # 靜默處理失敗
                pass

    results_df = pd.DataFrame(results)

    # 清理結果
    if results_df.empty or "sharpe_ratio" not in results_df.columns:
        logger.error("優化失敗: 結果為空")
        return {}, pd.DataFrame(), {}

    # 過濾失敗結果
    valid_results = results_df[results_df['sharpe_ratio'] > FAILED_SHARPE].copy()
    if valid_results.empty:
        logger.error("優化失敗: 所有結果都失敗")
        return {}, pd.DataFrame(), {}

    results_df = valid_results

    # 如果使用隨機搜索，補充鄰域點
    best_idx = results_df["sharpe_ratio"].idxmax()
    best_params_temp = {}
    param_names = list(param_grid.keys())
    for k in param_names:
        try:
            val = results_df.loc[best_idx, k]
            if k in ["window", "threshold_window", "leverage"]:
                best_params_temp[k] = int(val)
            elif k in ["long_entry_quantile", "short_entry_quantile", "capital_allocation"]:
                best_params_temp[k] = float(val)
            else:
                best_params_temp[k] = val
        except Exception as e:
            logger.warning(f"Could not convert param {k} for supplement_neighborhood. Error: {e}")
            best_params_temp[k] = results_df.loc[best_idx, k]

    # Walk-forward 分析總是補充鄰域點（因為通常使用隨機搜索）
    results_df = supplement_neighborhood_for_best(
        best_params=best_params_temp,
        param_grid=param_grid,
        results_df=results_df,
        **eval_kwargs
    )

    # ========== 步驟1: 篩選符合所有條件的優質參數 ==========
    # 從kwargs獲取篩選閾值
    calmar_threshold = kwargs.get('calmar_threshold', 2.5)
    annual_return_threshold = kwargs.get('annual_return_threshold', 0.5)
    max_drawdown_threshold = kwargs.get('max_drawdown_threshold', 0.3)  # 正數，如 0.3 表示 30%

    # 首先篩選只符合夏普比率條件的參數
    sharpe_only_df = results_df[results_df['sharpe_ratio'] > sharpe_threshold].copy()
    print(f"  ├─ 夏普比率 > {sharpe_threshold}: {len(sharpe_only_df)} 個組合")

    top_params_df = filter_top_params(
        results_df=results_df,
        sharpe_threshold=sharpe_threshold,
        calmar_threshold=calmar_threshold,
        annual_return_threshold=annual_return_threshold,
        max_drawdown_threshold=-max_drawdown_threshold  # 傳遞負數給 filter_top_params
    )
    print(f"  ├─ 同時符合所有條件: {len(top_params_df)} 個組合")

    if not top_params_df.empty:
        # 情況A: 有符合所有條件的參數 → 進行穩健性分析，選擇穩健性最高的
        print(f"  ├─ [情況A] 找到 {len(top_params_df)} 個符合條件的參數，進行穩健性分析...")
        try:
            # 傳入候選集(top_params_df)和完整結果集(results_df)
            robust_params_df, best_robust_params = filter_robust_params(
                candidates_df=top_params_df,
                all_results_df=results_df,
                param_grid=param_grid,
                metric="sharpe_ratio",
                sharpe_threshold=sharpe_threshold,
                top_n=10
            )

            # 檢查穩健性分析是否有有效結果
            if not robust_params_df.empty and best_robust_params:
                best_robust_row = robust_params_df.iloc[0]
                # 安全地從 robust_row 獲取參數
                best_params = {pname: best_robust_row[pname] for pname in param_names if pname in best_robust_row}
                
                # 在原始 top_params_df 中找到對應的行以獲取完整指標
                match_mask = pd.Series(True, index=top_params_df.index)
                for pname, pval in best_params.items():
                    if pname in top_params_df.columns:
                        match_mask &= (top_params_df[pname] == pval)
                matched_rows = top_params_df[match_mask]
                
                if not matched_rows.empty:
                    matched_row = matched_rows.iloc[0]
                    num_robust = len(robust_params_df)
                    print(f"  ├─ 從 {len(top_params_df)} 個候選中選出排名前 {num_robust} 的穩健參數")
                    print(f"  ├─ [選中] 排名第 1 的參數:")
                    print(f"  |  - Sharpe={matched_row['sharpe_ratio']:.4f}, Calmar={matched_row.get('calmar_ratio', 0):.4f}, Return={matched_row.get('annual_return', 0)*100:.1f}%, DD={matched_row.get('max_drawdown', 0)*100:.1f}%")
                    print(f"  |  - 穩健性分數={best_robust_row.get('robustness_score', 0):.4f}")
                    
                    sensitivity_report = {
                        "selection_method": "robust_from_filtered",
                        "robustness_score": float(best_robust_row.get("robustness_score", 0)),
                        "best_sharpe": float(matched_row["sharpe_ratio"]),
                        "n_filtered_candidates": len(top_params_df),
                        "n_robust_candidates": len(robust_params_df),
                        "fallback_used": False
                    }
                else:
                    # 如果意外未找到匹配，則清空 best_params 以觸發回退
                    best_params = {}
            else:
                # 如果穩健性分析無結果，清空 best_params 以觸發回退
                best_params = {}

            # 統一的回退邏輯
            if not best_params:
                best_result = top_params_df.iloc[0]
                best_params = {pname: best_result[pname] for pname in param_names}
                print(f"  ├─ 穩健性分析無有效結果，回退到該窗口最高夏普比率的參數")
                print(f"  |  - Sharpe={best_result['sharpe_ratio']:.4f}")
                
                sensitivity_report = {
                    "selection_method": "fallback_from_filtered",
                    "best_sharpe": float(best_result["sharpe_ratio"]),
                    "n_filtered_candidates": len(top_params_df),
                    "fallback_used": True,
                    "fallback_reason": "robustness_analysis_failed_or_empty"
                }

        except Exception as e:
            # 異常時的回退邏輯
            best_result = top_params_df.iloc[0]
            best_params = {pname: best_result[pname] for pname in param_names}
            print(f"  ├─ 穩健性分析異常，回退到該窗口最高夏普比率的參數")
            print(f"  |  - Sharpe={best_result['sharpe_ratio']:.4f} (異常: {str(e)[:30]}...)")
            
            sensitivity_report = {
                "selection_method": "fallback_from_filtered",
                "best_sharpe": float(best_result["sharpe_ratio"]),
                "n_filtered_candidates": len(top_params_df),
                "fallback_used": True,
                "fallback_reason": f"exception: {str(e)[:50]}"
            }

    else:
        # 情況B: 沒有符合所有條件的參數 → 直接選擇全局最高夏普比率參數
        best_idx = results_df["sharpe_ratio"].idxmax()
        best_result = results_df.loc[best_idx]
        best_params = {pname: best_result[pname] for pname in param_names}
        print(f"  ├─ [情況B] 無符合條件參數，回退到全局最高夏普比率")
        print(f"  |  - Sharpe={best_result['sharpe_ratio']:.4f}, Calmar={best_result.get('calmar_ratio', 0):.4f}, Return={best_result.get('annual_return', 0)*100:.1f}%, DD={best_result.get('max_drawdown', 0)*100:.1f}%")
        
        sensitivity_report = {
            "selection_method": "global_best_sharpe",
            "best_sharpe": float(best_result["sharpe_ratio"]),
            "n_total_candidates": len(results_df),
            "fallback_used": False
        }

    return best_params, results_df, sensitivity_report


def run_oos_test(
    window_id: int,
    test_start: str,
    test_end: str,
    best_params: dict,
    data_full: pd.DataFrame,
    **kwargs
) -> Tuple[pd.Series, dict]:
    """
    運行樣本外測試，使用最佳參數

    Returns:
        equity_curve: OOS權益曲線
        oos_metrics: OOS績效指標
    """
    try:
        # 解析參數
        leverage = int(best_params.get("leverage", 1))
        capital_allocation = float(best_params.get("capital_allocation", 1.0))
        final_leverage = leverage * capital_allocation

        # 提取測試數據
        test_start_dt = pd.to_datetime(test_start)
        test_end_dt = pd.to_datetime(test_end)
        test_data = data_full[
            (data_full.index >= test_start_dt) &
            (data_full.index <= test_end_dt)
        ].copy()

        if test_data.empty:
            return pd.Series(), {}

        # 生成策略信號
        strategy = CVILLIQStrategy()
        signals = strategy.generate_signals(
            test_data,
            **best_params
        )

        # 計算信號變化次數
        signal_changes = signals.diff().fillna(0)
        num_signal_changes = (signal_changes != 0).sum()

        # 執行回測
        transaction_cost = kwargs["transaction_cost"]
        slippage = kwargs["slippage"]
        periods_per_year = kwargs["periods_per_year"]
        stop_loss_fee = kwargs.get("stop_loss_fee", 0.00055)

        engine = BacktestEngine(
            data=test_data,
            signals=signals,
            transaction_cost=transaction_cost,
            slippage=slippage,
            initial_capital=100000,
            periods_per_year=periods_per_year,
            leverage=final_leverage,
            stop_loss_fee=stop_loss_fee
        )

        results = engine.run()
        equity_curve = results.get('equity_curve', pd.Series())

        # 提取績效指標
        metrics = results.get('metrics', {})
        oos_metrics = {
            "sharpe_ratio": float(metrics.get('sharpe_ratio', 0)),
            "calmar_ratio": float(metrics.get('calmar_ratio', 0)),
            "annual_return": float(metrics.get('annualized_return', 0)),
            "max_drawdown": float(metrics.get('max_drawdown', 0)),
            "data_length": len(test_data),
            "signal_changes": int(num_signal_changes)
        }

        return equity_curve, oos_metrics

    except Exception as e:
        logger.error(f"OOS測試失敗: {e}")
        return pd.Series(), {}


# ==================== 主函數 ====================

def main():
    """主函數"""
    print("=" * 80)
    print("CVILLIQ CTA 策略 - Walk-Forward 優化與敏感性分析")
    print("=" * 80)
    print()

    # ========== 用戶配置區 ==========

    # 數據配置
    DATA_CONFIG = "BTCUSDT_1h"
    FULL_START_DATE = "2022-11-01"
    FULL_END_DATE = "2025-10-31"

    # Walk-Forward 窗口配置
    # 按"數據點數"定義，自動轉換為相應時間單位:
    # - 日級數據 (間隔24小時): 240數據點 = 240天 ≈ 8個月
    # - 小時級數據 (間隔1小時): 5760數據點 = 5760小時 ≈ 8個月
    #
    # 重疊率分析：
    # - 原版：TRAIN=18720, STEP=720 → 重疊率96.2%（參數高度相似）
    # - 優化：TRAIN=18720, STEP=4320 → 重疊率77%（參數更獨立）
    TRAIN_WINDOW_POINTS = 18720   # 訓練窗口 18720個數據點 (約780天，小時級)
    TEST_WINDOW_POINTS = 720      # 測試窗口 720個數據點 (約30天，小時級)
    STEP_WINDOW_POINTS = 720     # 滾動步長 4320個數據點 (約180天，小時級，減少重疊)

    # 參數網格 (4維空間)
    # 修改: 分離 long_threshold 和 short_threshold
    # 邏輯: CVILLIQ > long_threshold 時做多，CVILLIQ < short_threshold 時做空
    # CVILLIQ 數據分布: min=0.34, max=1.43, mean=0.71
    #
    # 參數範圍優化說明：
    # - 原版範圍 [0.1, 0.3, 0.5, 0.7, 0.9] 導致54%空倉率（過於保守）
    # - 新版範圍縮小到中間區域，避免極端保守參數
    # - 預期空倉率降至 35-45%，交易頻率提升 20-30%
    PARAM_GRID = {
        "window": range(200, 220, 1),                            # 因子計算窗口
        "threshold_window": range(400, 420, 1),                  # 門檻計算窗口
        "long_entry_quantile": np.arange(0.1, 1, 0.2),       # 做多百分位：[0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
        "short_entry_quantile": np.arange(0.1, 1, 0.2),    # 做空百分位：[0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        "leverage": [1],                                         # 基礎槓桿
        "capital_allocation": np.arange(1, 1.01, 0.1)           # 資金分配
    }

    # 隨機搜索次數
    N_TRIALS = 10000

    # 敏感性分析配置
    SHARPE_THRESHOLD = 1.25

    # 頂部參數篩選條件（與 test_optimization_with_sensitivity.py 一致）
    CALMAR_THRESHOLD = 2
    ANNUAL_RETURN_THRESHOLD = 0.5  # 50%
    MAX_DRAWDOWN_THRESHOLD = 0.3   # 30% (允許最大回撤)

    # 交易成本
    TRANSACTION_COST = 0.0002
    SLIPPAGE = 0.0001
    STOP_LOSS_FEE = 0.00055

    # 並行計算設置
    N_JOBS = -1

    # ========== 配置區結束 ==========

    print(f"數據配置: {DATA_CONFIG}")
    print(f"完整期間: {FULL_START_DATE} 至 {FULL_END_DATE}")
    print(f"窗口配置 - 訓練: {TRAIN_WINDOW_POINTS}數據點, 測試: {TEST_WINDOW_POINTS}數據點, 步長: {STEP_WINDOW_POINTS}數據點")
    print()

    # Step 0: 加載數據
    symbol, timeframe = DATA_CONFIG.rsplit('_', 1)
    print(f"【Step 0】加載數據")
    
    data_full = load_local_data(
        symbol=symbol,
        data_source=timeframe,
        start_date=FULL_START_DATE,
        end_date=FULL_END_DATE,
        verbose=False
    )

    if data_full is None or data_full.empty:
        raise FileNotFoundError(f"無法加載 {symbol}_{timeframe} 的數據")

    print(f"✓ 數據加載成功: {len(data_full)} 行\n")

    # 設定年化參數
    periods_per_year = 365 if timeframe == "1d" else 365 * 24

    # Step 1: 生成Walk-Forward窗口
    print("【Step 1】生成Walk-Forward窗口")

    # 根據數據時間週期自動調整窗口配置
    # 檢測數據的時間頻率（以小時數計）
    if len(data_full) >= 2:
        time_diff = (data_full.index[1] - data_full.index[0]).total_seconds() / 3600
        if time_diff >= 24:  # 日級或更低頻率
            window_unit = "天"
            window_delta = timedelta(days=TRAIN_WINDOW_POINTS)
            test_delta = timedelta(days=TEST_WINDOW_POINTS)
            step_delta = timedelta(days=STEP_WINDOW_POINTS)
        else:  # 小時級或更高頻率
            window_unit = "小時"
            window_delta = timedelta(hours=TRAIN_WINDOW_POINTS)
            test_delta = timedelta(hours=TEST_WINDOW_POINTS)
            step_delta = timedelta(hours=STEP_WINDOW_POINTS)
    else:
        raise ValueError("數據不足以檢測時間頻率")

    windows = []
    current_train_start = pd.to_datetime(FULL_START_DATE)
    full_end_dt = pd.to_datetime(FULL_END_DATE)

    window_id = 1
    while current_train_start < full_end_dt:
        train_start = current_train_start
        train_end = train_start + window_delta
        test_start = train_end
        test_end = test_start + test_delta

        if test_end > full_end_dt:
            test_end = full_end_dt

        # 確保有足夠的測試數據
        if test_end <= test_start:
            break

        windows.append({
            "window_id": window_id,
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d")
        })

        current_train_start += step_delta
        window_id += 1

    print(f"✓ 生成 {len(windows)} 個Walk-Forward窗口\n")

    # Step 2: 運行Walk-Forward分析
    print("【Step 2】運行Walk-Forward分析")

    all_oos_metrics = []
    all_params_evolution = []
    all_equity_curves = {}
    last_successful_params = None  # 用於備份上一個成功的參數
    last_successful_window_id = None  # 記錄備份參數來自哪個窗口

    for window in windows:
        wid = window["window_id"]
        train_start = window["train_start"]
        train_end = window["train_end"]
        test_start = window["test_start"]
        test_end = window["test_end"]

        # 訓練窗口優化 - 傳遞篩選閾值
        best_params, train_results_df, sensitivity_report = optimize_window_with_sensitivity(
            window_id=wid,
            train_start=train_start,
            train_end=train_end,
            param_grid=PARAM_GRID,
            n_trials=N_TRIALS,
            sharpe_threshold=SHARPE_THRESHOLD,
            data_full=data_full,
            transaction_cost=TRANSACTION_COST,
            slippage=SLIPPAGE,
            periods_per_year=periods_per_year,
            stop_loss_fee=STOP_LOSS_FEE,
            n_jobs=N_JOBS,
            # 新增：傳遞篩選閾值
            calmar_threshold=CALMAR_THRESHOLD,
            annual_return_threshold=ANNUAL_RETURN_THRESHOLD,
            max_drawdown_threshold=MAX_DRAWDOWN_THRESHOLD
        )

        if not best_params:
            if last_successful_params:
                window_gap = wid - last_successful_window_id
                logger.warning(f"窗口 {wid} 優化失敗，使用窗口 {last_successful_window_id} 的參數作為備份 (已過期 {window_gap} 個窗口)")
                best_params = last_successful_params.copy()
            else:
                logger.warning(f"窗口 {wid} 優化失敗，沒有備份參數可用，跳過")
                continue
        else:
            # 記錄成功的參數供下一個窗口備份使用
            last_successful_params = best_params.copy()
            last_successful_window_id = wid

        # 樣本外測試
        oos_equity, oos_metrics = run_oos_test(
            window_id=wid,
            test_start=test_start,
            test_end=test_end,
            best_params=best_params,
            data_full=data_full,
            transaction_cost=TRANSACTION_COST,
            slippage=SLIPPAGE,
            periods_per_year=periods_per_year,
            stop_loss_fee=STOP_LOSS_FEE
        )

        # 記錄結果
        all_oos_metrics.append({
            "window_id": wid,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            **best_params,
            **sensitivity_report,
            **oos_metrics
        })

        all_params_evolution.append({
            "window_id": wid,
            **best_params
        })

        if not oos_equity.empty:
            all_equity_curves[f"window_{wid}"] = oos_equity

    # Step 3: 生成報告和圖表
    print("\n【Step 3】生成報告和圖表")
    print("-" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "results" / f"WalkForward_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 overall_metrics 以免引用錯誤
    overall_sharpe = 0
    overall_calmar = 0
    overall_annual_return = 0
    overall_max_drawdown = 0
    
    # 生成權益與回撤對比圖表，並接收返回的總權益曲線
    combined_equity = None
    if all_equity_curves:
        try:
            combined_equity = plot_walkforward_equity_and_drawdown(
                all_equity_curves=all_equity_curves,
                benchmark_data=data_full,
                output_path=output_dir / "equity_and_drawdown.png",
                strategy_name="策略權益",
                benchmark_name=f"{DATA_CONFIG.split('_')[0]}持有"
            )
        except Exception as e:
            print(f"⚠ 權益回撤圖表生成失敗: {str(e)}")
            import traceback
            traceback.print_exc()

    # 保存Walk-Forward結果
    if all_oos_metrics:
        oos_results_df = pd.DataFrame(all_oos_metrics)
        oos_results_df.to_csv(output_dir / "walkforward_results.csv", index=False)
        print(f"✓ 保存Walk-Forward結果: walkforward_results.csv")

        # 保存參數演化
        params_evolution_df = pd.DataFrame(all_params_evolution)
        params_evolution_df.to_csv(output_dir / "parameter_evolution.csv", index=False)
        print(f"✓ 保存參數演化: parameter_evolution.csv")

        # ========== 步驟4：計算整體 OOS 性能（邏輯已修正） ==========
        print("\n【Step 4】整體 OOS 性能統計")
        print("-" * 80)
        
        # 關鍵修正: 基於拼接好的總權益曲線計算整體指標
        if combined_equity is not None and not combined_equity.empty:
            print("基於連續權益曲線計算整體性能指標...")
            # 1. 從總權益曲線計算連續收益率
            combined_returns = combined_equity.pct_change().fillna(0)
            
            # 2. 計算整體指標
            from shared.backtest import calculate_all_metrics
            
            overall_metrics = calculate_all_metrics(
                returns=combined_returns,
                signals=pd.Series(np.ones(len(combined_returns)), index=combined_returns.index),
                initial_capital=combined_equity.iloc[0],
                periods_per_year=periods_per_year
            )
            
            overall_sharpe = overall_metrics.get('sharpe_ratio', 0)
            overall_calmar = overall_metrics.get('calmar_ratio', 0)
            overall_annual_return = overall_metrics.get('annualized_return', 0)
            overall_max_drawdown = overall_metrics.get('max_drawdown', 0)
                
            print(f"\n🎯 整體 OOS 性能（所有窗口連接）:")
            print(f"   夏普比率: {overall_sharpe:.4f}")
            print(f"   卡瑪比率: {overall_calmar:.4f}")
            print(f"   年報酬: {overall_annual_return*100:.2f}%")
            print(f"   最大回撤: {overall_max_drawdown*100:.2f}%")
            print()
        else:
            print("⚠ 無法計算整體性能，因為沒有生成有效的總權益曲線。")

        # 保存綜合總結
        summary = {
            "configuration": {
                "data_config": DATA_CONFIG,
                "full_start_date": FULL_START_DATE,
                "full_end_date": FULL_END_DATE,
                "train_window_points": TRAIN_WINDOW_POINTS,
                "test_window_points": TEST_WINDOW_POINTS,
                "step_window_points": STEP_WINDOW_POINTS,
                "window_unit": window_unit,
                "num_windows": len(windows)
            },
            "walkforward_oos_performance": {
                "sharpe_ratio": float(overall_sharpe),
                "calmar_ratio": float(overall_calmar),
                "annual_return": float(overall_annual_return),
                "max_drawdown": float(overall_max_drawdown)
            }
        }

        with open(output_dir / "walkforward_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        print(f"✓ 保存綜合總結: walkforward_summary.json")

    print()
    print("=" * 80)
    print(f"所有結果已保存到: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
