"""
Momentum Strategy (Single Asset Trend Following)
動量策略 (單一標的趨勢跟隨)

根據價格在過去一段時間的變化率來識別趨勢方向。
進場條件:
- 多頭: N日價格變化率 > 0 (價格處於上升趨勢)
- 空頭: N日價格變化率 < 0 (價格處於下降趨勢)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np
from typing import Dict, Any

from shared.base_strategy import BaseStrategy


def validate_params(params: dict, df_len: int) -> None:
    """
    驗證策略參數

    Args:
        params: 參數字典
        df_len: 數據長度

    Raises:
        ValueError: 如果參數無效
    """
    errors = []

    lookback_period = int(params.get('lookback_period', 20))
    if lookback_period < 2:
        errors.append("lookback_period 必須 >= 2")

    leverage = int(params.get('leverage', 1))
    if leverage < 1:
        errors.append(f"leverage 必須 >= 1, 得到 {leverage}")

    momentum_threshold = float(params.get('momentum_threshold', 0.0))
    if momentum_threshold < 0:
        errors.append(f"momentum_threshold 必須 >= 0, 得到 {momentum_threshold}")

    if errors:
        raise ValueError("無效參數:\n" + "\n".join(errors))


def _calculate_signals(momentum_shifted: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    計算動量信號

    邏輯:
    - 多頭: 動量 > +threshold
    - 空頭: 動量 < -threshold
    - 空倉: -threshold <= 動量 <= +threshold 或 NaN

    Args:
        momentum_shifted: 前一期的動量值 (shift(1) 後)
        threshold: 動量門檻值 (預設 0.0 = 無門檻)

    Returns:
        np.ndarray: 交易信號 (1=多, -1=空, 0=空倉)
    """
    signals = np.zeros_like(momentum_shifted, dtype=np.int8)
    signals[momentum_shifted > threshold] = 1
    signals[momentum_shifted < -threshold] = -1
    return signals


class MomentumStrategy(BaseStrategy):
    """
    動量 CTA 策略類
    """

    def __init__(self, name: str = "Momentum"):
        super().__init__()
        self.name = name
        self.parameters = {}
        self.debug_data = {}

    def get_required_lookback(self, **params) -> int:
        """返回策略所需的最小 lookback 期，這對於預熱數據至關重要"""
        return int(params.get('lookback_period', 250))

    def generate_signals(self, df: pd.DataFrame, **params) -> pd.Series:
        """
        生成交易信號

        Args:
            df: OHLCV數據 (必須包含 'close' 列)
            lookback_period: 動量計算窗口
            leverage: 槓桿倍數
            momentum_threshold: 動量進場門檻值 (預設 0.0)

        Returns:
            pd.Series: 交易信號序列
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame 缺少必需列: 'close'")

        lookback_period = int(params.get('lookback_period', 20))

        # 如果數據長度不足以計算回看，則返回空信號
        if lookback_period >= len(df):
            return pd.Series(0, index=df.index, dtype=np.int8)

        validate_params(params, len(df))

        momentum_threshold = float(params.get('momentum_threshold', 0.0))

        df_copy = df.copy()

        # ==================== 計算動量指標 ====================
        momentum = df_copy['close'].pct_change(periods=lookback_period)
        self.debug_data['momentum'] = momentum

        # ==================== 防止lookahead bias ====================
        momentum_shifted = momentum.shift(1)

        # ==================== 計算信號 ====================
        signals_arr = _calculate_signals(momentum_shifted.values, threshold=momentum_threshold)

        self.parameters = params
        return pd.Series(signals_arr, index=df.index)

    def __call__(self, df: pd.DataFrame, **params) -> pd.Series:
        """使策略物件可調用"""
        return self.generate_signals(df, **params)

    def get_parameter_grid(self) -> Dict[str, Any]:
        """返回用於優化的參數網格"""
        return {
            'lookback_period': range(20, 201, 20),
            'leverage': [1, 2, 3],
            'momentum_threshold': [0.0, 0.05, 0.10, 0.15, 0.20]
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        """返回策略的默認參數"""
        return {
            'lookback_period': 60,
            'leverage': 1,
            'momentum_threshold': 0.0
        }
