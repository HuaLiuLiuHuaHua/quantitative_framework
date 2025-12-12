# -*- coding: utf-8 -*-
"""
參數篩選與穩健性分析模塊

從 cvilliq 策略提取的通用功能，用於 Walk-Forward 分析中的參數選擇。
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, List, Dict

logger = logging.getLogger(__name__)


def filter_top_params(
    results_df: pd.DataFrame,
    sharpe_threshold: float = 1.25,
    calmar_threshold: float = 2.5,
    annual_return_threshold: float = 0.5,
    max_drawdown_threshold: float = -0.2
) -> pd.DataFrame:
    """
    篩選符合條件的優質參數組合

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
        符合條件的參數組合 DataFrame
    """
    # 失敗結果的默認值
    FAILED_SHARPE = -999

    # 過濾失敗結果
    valid_df = results_df[results_df['sharpe_ratio'] > FAILED_SHARPE].copy()
    if valid_df.empty:
        logger.warning("警告: 沒有找到有效的優化結果")
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

    # 如果沒有參數滿足所有條件，返回空 DataFrame
    if filtered_df.empty:
        logger.warning(
            f"⚠️  警告：沒有任何參數同時滿足所有條件 "
            f"(Sharpe > {sharpe_threshold}, Calmar > {calmar_threshold}, "
            f"年化報酬 > {annual_return_threshold:.0%}, 最大回撤 < {-max_drawdown_threshold:.0%})"
        )
        return pd.DataFrame()

    logger.info(
        f"篩選條件: 夏普 > {sharpe_threshold}, 卡瑪 > {calmar_threshold}, "
        f"年化報酬 > {annual_return_threshold:.0%}, 最大回撤 > {max_drawdown_threshold:.0%}"
    )
    logger.info(f"符合條件的參數組合: {len(filtered_df)} 個")

    return filtered_df


def filter_robust_params(
    candidates_df: pd.DataFrame,
    all_results_df: pd.DataFrame,
    param_grid: dict,
    metric: str = 'sharpe_ratio',
    sharpe_threshold: float = 1.25,
    top_n: int = 10
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    從候選參數中進行穩健性分析，選擇在鄰域內表現最穩定的參數

    Args:
        candidates_df: 候選參數組合 DataFrame（已篩選）
        all_results_df: 所有優化結果 DataFrame
        param_grid: 參數網格
        metric: 評估指標（default 'sharpe_ratio'）
        sharpe_threshold: 夏普比率閾值，用於篩選鄰域內有效參數
        top_n: 從候選中選出排名前 N 的參數進行穩健性分析

    Returns:
        (robust_params_df, best_robust_params_dict)
        robust_params_df: 按穩健性分數排序的參數
        best_robust_params_dict: 最穩健的參數字典（如果為空則返回None）
    """
    from .sensitivity import get_parameter_neighborhood as get_param_neighbors

    if candidates_df.empty:
        logger.warning("候選參數為空，無法進行穩健性分析")
        return pd.DataFrame(), None

    param_names = list(param_grid.keys())

    # 1. 從候選中選出排名前 top_n 的參數
    top_candidates = candidates_df.nlargest(top_n, metric).copy()

    # 2. 對每個候選參數，檢查其鄰域內的表現
    robust_results = []

    for idx, candidate_row in top_candidates.iterrows():
        candidate_params = {pname: candidate_row[pname] for pname in param_names}

        # 獲取該參數的鄰域
        all_neighbors = []
        for param_name_iter in param_names:
            neighbors = get_param_neighbors(
                param_name=param_name_iter,
                param_value=candidate_params[param_name_iter],
                param_grid=param_grid,
                step_size=2
            )
            for neighbor_val in neighbors:
                neighbor_params = candidate_params.copy()
                neighbor_params[param_name_iter] = neighbor_val
                all_neighbors.append(neighbor_params)

        # 查找鄰域參數在 all_results_df 中的表現
        neighborhood_results = []
        for neighbor_params in all_neighbors:
            # 在 all_results_df 中尋找匹配的結果
            mask = pd.Series(True, index=all_results_df.index)
            for pname, pval in neighbor_params.items():
                if pname in all_results_df.columns:
                    mask &= (all_results_df[pname] == pval)

            matches = all_results_df[mask]
            if not matches.empty:
                neighborhood_results.append(matches.iloc[0][metric])

        # 計算穩健性分數：鄰域內有效參數的比例及其平均表現
        if neighborhood_results:
            valid_neighbors = [r for r in neighborhood_results if pd.notna(r)]
            if valid_neighbors:
                robustness_score = np.mean(valid_neighbors)
                robustness_count = len(valid_neighbors)
            else:
                robustness_score = 0
                robustness_count = 0
        else:
            robustness_score = 0
            robustness_count = 0

        robust_result = {
            **candidate_params,
            'robustness_score': robustness_score,
            'neighborhood_valid_count': robustness_count,
            'original_metric': float(candidate_row[metric])
        }
        robust_results.append(robust_result)

    if not robust_results:
        logger.warning("穩健性分析未找到有效結果")
        return pd.DataFrame(), None

    robust_params_df = pd.DataFrame(robust_results).sort_values(
        'robustness_score', ascending=False
    ).reset_index(drop=True)

    if robust_params_df.empty:
        return pd.DataFrame(), None

    best_robust_params = {
        pname: robust_params_df.iloc[0][pname] for pname in param_names
    }

    return robust_params_df, best_robust_params
