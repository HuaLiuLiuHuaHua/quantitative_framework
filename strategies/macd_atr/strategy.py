"""
MACD-ATR Strategy
MACD-ATR 策略
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

try:
    import ta
except ImportError:
    raise ImportError("缺少 ta 庫。請安裝: pip install ta")

try:
    import numba
except ImportError:
    raise ImportError("缺少 numba 庫。請安裝: pip install numba")

@numba.jit(nopython=True)
def _calculate_signals_nb(
    close: np.ndarray,
    macd_prev: np.ndarray,
    macd_signal_prev: np.ndarray,
    atr_prev: np.ndarray,
    atr_multiplier: float,
    initial_bars: int
) -> np.ndarray:
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    current_signal = 0
    entry_price = 0.0
    stop_loss_price = 0.0

    for i in range(initial_bars, n):
        current_price = close[i]
        macd_value = macd_prev[i]
        macd_signal_value = macd_signal_prev[i]
        atr_value = atr_prev[i]

        if np.isnan(current_price) or np.isnan(macd_value) or np.isnan(macd_signal_value) or np.isnan(atr_value):
            signals[i] = current_signal
            continue

        if current_signal == 0:
            if macd_value > macd_signal_value:
                current_signal = 1
                entry_price = current_price
                stop_loss_price = entry_price - atr_multiplier * atr_value
            elif macd_value < macd_signal_value:
                current_signal = -1
                entry_price = current_price
                stop_loss_price = entry_price + atr_multiplier * atr_value
        elif current_signal == 1:
            if current_price < stop_loss_price:
                current_signal = 0
                entry_price = 0.0
                stop_loss_price = 0.0
        elif current_signal == -1:
            if current_price > stop_loss_price:
                current_signal = 0
                entry_price = 0.0
                stop_loss_price = 0.0
        
        signals[i] = current_signal

    return signals

class MACDATRStrategy:
    """
    MACD-ATR 策略類 (Numba 優化版)
    """

    def __init__(self, name: str = "MACD_ATR"):
        self.name = name
        self.parameters = {}

    def generate_signals(self, df: pd.DataFrame, **params) -> pd.Series:
        """
        生成交易信號
        """
        macd_fast = params.get('macd_fast', 12)
        macd_slow = params.get('macd_slow', 26)
        macd_signal = params.get('macd_signal', 9)
        atr_window = params.get('atr_window', 14)
        atr_multiplier = params.get('atr_multiplier', 2.0)

        df_copy = df.copy()

        macd = ta.trend.MACD(df_copy['close'], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
        df_copy['MACD'] = macd.macd()
        df_copy['MACD_Signal'] = macd.macd_signal()
        df_copy['ATR'] = ta.volatility.AverageTrueRange(df_copy['high'], df_copy['low'], df_copy['close'], window=atr_window).average_true_range()

        df_copy['MACD_prev'] = df_copy['MACD'].shift(1)
        df_copy['MACD_Signal_prev'] = df_copy['MACD_Signal'].shift(1)
        df_copy['ATR_prev'] = df_copy['ATR'].shift(1)

        df_copy.ffill(inplace=True)
        df_copy.bfill(inplace=True)

        initial_bars = max(macd_slow, atr_window) + 5
        if len(df_copy) < initial_bars:
            return pd.Series(0, index=df_copy.index)

        signals_arr = _calculate_signals_nb(
            df_copy['close'].to_numpy(),
            df_copy['MACD_prev'].to_numpy(),
            df_copy['MACD_Signal_prev'].to_numpy(),
            df_copy['ATR_prev'].to_numpy(),
            float(atr_multiplier),
            initial_bars
        )

        self.parameters = params
        return pd.Series(signals_arr, index=df.index)

    def __call__(self, df: pd.DataFrame, **params) -> pd.Series:
        return self.generate_signals(df, **params)