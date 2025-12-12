"""
MA交叉策略

傳統技術指標策略,不需要訓練
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np

from shared.base_strategy import BaseStrategy


class MACrossStrategy(BaseStrategy):
    """MA交叉策略"""

    def __init__(self):
        super().__init__()

    def get_lookback(self) -> int:
        """返回lookback期"""
        return 60  # 長均線周期+緩衝

    def generate_signals(
        self,
        data: pd.DataFrame,
        short_window: int = 20,
        long_window: int = 50,
        **kwargs
    ) -> pd.Series:
        """
        生成交易信號

        Args:
            data: OHLCV DataFrame
            short_window: 短期均線周期
            long_window: 長期均線周期

        Returns:
            信號Series (1=做多, -1=做空, 0=空倉)
        """
        # 計算均線
        ma_short = data['close'].rolling(short_window).mean()
        ma_long = data['close'].rolling(long_window).mean()

        # MA交叉因子 (短均線 - 長均線) / 長均線
        factor_value = (ma_short - ma_long) / ma_long

        # 生成信號
        signals = pd.Series(0, index=data.index, dtype=int)

        # 因子值 > 0 → 做多
        signals[factor_value > 0] = 1

        # 因子值 < 0 → 做空
        signals[factor_value < 0] = -1

        # shift(1) 防止 lookahead bias
        signals = signals.shift(1).fillna(0).astype(int)

        return signals

    def get_parameter_grid(self):
        """返回可優化的參數網格"""
        return {
            'short_window': list(range(10, 31, 5)),      # 10-30
            'long_window': list(range(40, 101, 10))      # 40-100
        }

    def get_default_parameters(self):
        """返回默認參數"""
        return {
            'short_window': 20,
            'long_window': 50
        }
