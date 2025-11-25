"""
卡爾曼濾波器配對交易倉位策略
"""

import pandas as pd
from typing import Dict, Tuple, List
import logging

from shared.position_strategies import BasePositionStrategy

logger = logging.getLogger(__name__)


class KalmanPairsPositionStrategy(BasePositionStrategy):
    """
    卡爾曼濾波器配對交易倉位策略

    根據價差 z-score 生成交易信號:
    - z-score > entry_threshold: 做空價差 (賣出 asset1, 買入 asset2)
    - z-score < -entry_threshold: 做多價差 (買入 asset1, 賣出 asset2)
    - |z-score| < exit_threshold: 平倉
    """

    def __init__(self):
        super().__init__(strategy_name="KalmanPairs")

    def calculate_positions(
        self,
        factor_snapshot: pd.Series,
        all_data: Dict[str, pd.DataFrame],
        decision_date: pd.Timestamp,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        leverage: float = 1.0,
        **kwargs
    ) -> Dict[str, Tuple[float, str]]:
        """
        根據 z-score 生成配對交易倉位

        Args:
            factor_snapshot: 當日 z-score (Series with single value for the pair)
            all_data: 所有資產的價格數據字典
            decision_date: 決策日期
            entry_threshold: 入場閾值 (標準差倍數)
            exit_threshold: 出場閾值 (標準差倍數)
            leverage: 槓桿倍數
            **kwargs: 其他參數

        Returns:
            {pair_id: (weight, direction)}
            - weight: 倉位權重 (考慮槓桿)
            - direction: 'long' (做多價差) 或 'short' (做空價差)
        """
        if factor_snapshot.empty:
            return {}

        # 獲取配對 ID 和 z-score
        pair_id = factor_snapshot.index[0]
        zscore = factor_snapshot.iloc[0]

        # 檢查是否為 NaN
        if pd.isna(zscore):
            logger.debug(f"[{decision_date.date()}] {pair_id}: z-score is NaN, 不交易")
            return {}

        # 入場/出場邏輯
        if abs(zscore) < exit_threshold:
            # 價差回歸到均值附近,出場
            logger.debug(
                f"[{decision_date.date()}] {pair_id}: z-score={zscore:.2f} "
                f"< exit_threshold={exit_threshold:.2f}, 出場"
            )
            return {}

        elif zscore > entry_threshold:
            # z-score 過高 → 價差被高估 → 做空價差
            # 實際操作: 賣出 asset1 (價格相對高), 買入 asset2 (價格相對低)
            logger.debug(
                f"[{decision_date.date()}] {pair_id}: z-score={zscore:.2f} "
                f"> entry_threshold={entry_threshold:.2f}, 做空價差"
            )
            return {pair_id: (leverage, 'short')}

        elif zscore < -entry_threshold:
            # z-score 過低 → 價差被低估 → 做多價差
            # 實際操作: 買入 asset1 (價格相對低), 賣出 asset2 (價格相對高)
            logger.debug(
                f"[{decision_date.date()}] {pair_id}: z-score={zscore:.2f} "
                f"< -entry_threshold={-entry_threshold:.2f}, 做多價差"
            )
            return {pair_id: (leverage, 'long')}

        else:
            # 在閾值內但未達到入場條件,不交易
            logger.debug(
                f"[{decision_date.date()}] {pair_id}: z-score={zscore:.2f} "
                f"在閾值內, 不交易"
            )
            return {}

    def get_required_params(self) -> List[str]:
        """
        返回此策略需要的參數列表

        Returns:
            參數名稱列表
        """
        return ['entry_threshold', 'exit_threshold', 'leverage']
