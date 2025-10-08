"""
SMA-ATR Strategy
SMA-ATR 策略
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
    print("警告: 缺少 ta 庫，將使用手動方式計算 ATR。建議安裝: pip install ta")

try:
    import numba
except ImportError:
    raise ImportError("缺少 numba 庫。請安裝: pip install numba")

@numba.jit(nopython=True)
def _calculate_signals_nb(
    close: np.ndarray,
    sma_prev: np.ndarray,
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
        sma_value = sma_prev[i]
        atr_value = atr_prev[i]

        if np.isnan(current_price) or np.isnan(sma_value) or np.isnan(atr_value):
            signals[i] = current_signal
            continue

        if current_signal == 0:
            if current_price > sma_value:
                current_signal = 1
                entry_price = current_price
                stop_loss_price = entry_price - atr_multiplier * atr_value
            elif current_price < sma_value:
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

class SMAATRStrategy:
    """
    SMA-ATR 策略類 (Numba 優化版)
    """

    def __init__(self, name: str = "SMA_ATR"):
        self.name = name
        self.parameters = {}

    def generate_signals(self, df: pd.DataFrame, **params) -> pd.Series:
        """
        生成交易信號
        """
        sma_period = int(params.get('sma_period', 20))
        atr_window = int(params.get('atr_window', 14))
        atr_multiplier = float(params.get('atr_multiplier', 2.0))

        df_copy = df.copy()

        df_copy['SMA'] = df_copy['close'].rolling(window=sma_period).mean()
        
        try:
            import ta
            df_copy['ATR'] = ta.volatility.AverageTrueRange(df_copy['high'], df_copy['low'], df_copy['close'], window=atr_window).average_true_range()
        except ImportError:
            df_copy['H-L'] = df_copy['high'] - df_copy['low']
            df_copy['H-PC'] = abs(df_copy['high'] - df_copy['close'].shift(1))
            df_copy['L-PC'] = abs(df_copy['low'] - df_copy['close'].shift(1))
            df_copy['TR'] = df_copy[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            df_copy['ATR'] = df_copy['TR'].rolling(window=atr_window).mean()

        df_copy['SMA_prev'] = df_copy['SMA'].shift(1)
        df_copy['ATR_prev'] = df_copy['ATR'].shift(1)

        df_copy.ffill(inplace=True)
        df_copy.bfill(inplace=True)

        initial_bars = max(sma_period, atr_window) + 5
        if len(df_copy) < initial_bars:
            return pd.Series(0, index=df_copy.index)

        signals_arr = _calculate_signals_nb(
            df_copy['close'].to_numpy(),
            df_copy['SMA_prev'].to_numpy(),
            df_copy['ATR_prev'].to_numpy(),
            atr_multiplier,
            initial_bars
        )

        self.parameters = params
        return pd.Series(signals_arr, index=df.index)

    def __call__(self, df: pd.DataFrame, **params) -> pd.Series:
        return self.generate_signals(df, **params)