"""
XGBoost 因子組合策略 - 從組合因子生成交易信號
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np

from shared.base_strategy import BaseStrategy
from strategies.ml_combined.xgboost_combined.factor import XGBoostCombinedFactor


class XGBoostCombinedStrategy(BaseStrategy):
    """XGBoost 因子組合策略 - 用組合因子生成信號"""

    def __init__(self):
        super().__init__()
        self.factor = XGBoostCombinedFactor()

        # 嘗試加載已訓練的模型
        try:
            self.factor.load()
        except FileNotFoundError:
            print("[WARNING] 未找到訓練好的模型")
            print("   請先運行: python strategies/ml_combined/xgboost_combined/train.py")

    def get_lookback(self) -> int:
        """返回 lookback 期數"""
        # 技術指標需要足夠的歷史數據
        return 100

    def generate_signals(
        self,
        data: pd.DataFrame,
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.45,
        **kwargs
    ) -> pd.Series:
        """
        生成交易信號

        Args:
            data: OHLCV DataFrame
            buy_threshold: 買入閾值 (預測概率 > 此值則買入) (預設 0.55)
            sell_threshold: 賣出閾值 (預測概率 < 此值則賣出) (預設 0.45)

        Returns:
            信號 Series (1=買進, -1=賣出, 0=空倉)
        """
        # 計算因子值 (組合概率)
        prob = self.factor.calculate(data)  # 已經包含 shift(1)

        # 生成信號
        signals = pd.Series(0, index=data.index, dtype=int)

        # 買進信號
        signals[prob > buy_threshold] = 1

        # 賣出信號
        signals[prob < sell_threshold] = -1

        return signals.fillna(0).astype(int)

    def get_parameter_grid(self):
        """返回可優化的參數網格"""
        return {
            'buy_threshold': np.arange(0.50, 0.70, 0.02),
            'sell_threshold': np.arange(0.30, 0.50, 0.02),
        }

    def get_default_parameters(self):
        """返回默認參數"""
        return {
            'buy_threshold': 0.55,
            'sell_threshold': 0.45,
        }
