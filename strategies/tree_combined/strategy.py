"""
決策樹組合策略

使用決策樹因子生成交易信號
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np

from shared.base_strategy import BaseStrategy
from strategies.tree_combined.factor import TreeCombinedFactor


class TreeCombinedStrategy(BaseStrategy):
    """決策樹組合策略"""

    def __init__(self):
        super().__init__()
        self.factor = TreeCombinedFactor()
        # 自動加載因子 (如果存在)
        try:
            self.factor.load()
        except FileNotFoundError:
            print("⚠️  未找到訓練好的模型")
            print("   請先運行: python strategies/tree_combined/train.py")

    def get_lookback(self) -> int:
        """返回lookback期"""
        return 60  # 基礎因子需要的數據長度

    def generate_signals(
        self,
        data: pd.DataFrame,
        threshold: float = 0.0,
        **kwargs
    ) -> pd.Series:
        """
        生成交易信號

        Args:
            data: OHLCV DataFrame
            threshold: 因子值閾值 (>0做多, <0做空, 否則空倉)

        Returns:
            信號Series (1=做多, -1=做空, 0=空倉)
        """
        # 計算因子值
        factor_values = self.factor.calculate(data)

        # 生成信號
        signals = pd.Series(0, index=data.index, dtype=int)

        # 因子值 > threshold → 做多
        signals[factor_values > threshold] = 1

        # 因子值 < -threshold → 做空
        signals[factor_values < -threshold] = -1

        # shift(1) 再次確保無 lookahead bias
        signals = signals.shift(1).fillna(0).astype(int)

        return signals

    def get_parameter_grid(self):
        """返回可優化的參數網格"""
        return {
            'threshold': np.arange(-0.3, 0.4, 0.1)
        }

    def get_default_parameters(self):
        """返回默認參數"""
        return {
            'threshold': 0.0
        }
