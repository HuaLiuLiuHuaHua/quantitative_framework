"""
LSTM 特徵策略 - 從預測收益率生成交易信號

因子輸出預測收益率 (連續值):
- > threshold_buy: 買進 (long)
- < threshold_sell: 賣出 (short)
- 介於之間: 空倉 (neutral)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np

from shared.base_strategy import BaseStrategy
from strategies.lstm_feature.factor import LSTMFeatureFactor


class LSTMFeatureStrategy(BaseStrategy):
    """LSTM 特徵策略 - 用預測概率生成信號"""

    def __init__(self):
        super().__init__()
        self.factor = LSTMFeatureFactor()

        # 嘗試加載已訓練的模型
        try:
            self.factor.load()
        except FileNotFoundError:
            print("[WARNING] 未找到訓練好的模型")
            print("   請先運行: python strategies/lstm_feature/train.py")

    def get_lookback(self) -> int:
        """返回 lookback 期數"""
        # LSTM 需要序列長度 + 安全緩衝
        return self.factor.sequence_length + 50

    def generate_signals(
        self,
        data: pd.DataFrame,
        buy_threshold: float = 0.005,
        sell_threshold: float = -0.005,
        **kwargs
    ) -> pd.Series:
        """
        生成交易信號

        Args:
            data: OHLCV DataFrame
            buy_threshold: 買入閾值 (預測收益率 > 此值則買入) (預設 0.005 = 0.5%)
            sell_threshold: 賣出閾值 (預測收益率 < 此值則賣出) (預設 -0.005 = -0.5%)

        Returns:
            信號 Series (1=買進, -1=賣出, 0=空倉)
        """
        # 計算因子值 (預測收益率)
        returns_pred = self.factor.calculate(data)

        # 生成信號
        signals = pd.Series(0, index=data.index, dtype=int)

        # 買進信號
        signals[returns_pred > buy_threshold] = 1

        # 賣出信號
        signals[returns_pred < sell_threshold] = -1

        # shift(1) 再次防止 lookahead bias
        signals = signals.shift(1).fillna(0).astype(int)

        return signals

    def get_parameter_grid(self):
        """返回可優化的參數網格"""
        return {
            'buy_threshold': np.arange(0.001, 0.010, 0.001),   # 0.1%, 0.2%, ..., 0.9%
            'sell_threshold': np.arange(-0.010, -0.001, 0.001),  # -0.9%, -0.8%, ..., -0.1%
        }

    def get_default_parameters(self):
        """返回默認參數"""
        return {
            'buy_threshold': 0.005,
            'sell_threshold': -0.005,
        }
