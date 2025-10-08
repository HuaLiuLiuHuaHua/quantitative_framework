"""
LSTM SOL Strategy (Simplified Version without LSTM)
LSTM SOL 策略 (簡化版，無 LSTM)

這是一個基於技術指標組合的簡化版本，源自原始的 LSTM Walk-Forward 策略。
由於缺少訓練好的 LSTM 模型，我們使用純技術指標信號替代機器學習預測。
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np
from typing import Dict, Any

from .indicators import calculate_indicators

try:
    import numba
except ImportError:
    raise ImportError("缺少 numba 庫。請安裝: pip install numba")

@numba.jit(nopython=True)
def _calculate_signals_nb(
    close_prices: np.ndarray,
    rsi_values: np.ndarray,
    macd_values: np.ndarray,
    macd_signal_values: np.ndarray,
    sma_values: np.ndarray,
    rsi_long_threshold: float,
    rsi_short_threshold: float,
    rsi_long_exit: float,
    rsi_short_exit: float,
    sma_long_buffer: float,
    sma_short_buffer: float,
    initial_bars: int
) -> np.ndarray:
    n = len(close_prices)
    signals = np.zeros(n, dtype=np.int8)
    current_signal = 0

    for i in range(initial_bars, n):
        long_entry = (
            rsi_values[i] > rsi_long_threshold and
            close_prices[i] > sma_values[i] and
            macd_values[i] > macd_signal_values[i]
        )
        
        short_entry = (
            rsi_values[i] < rsi_short_threshold and
            close_prices[i] < sma_values[i] and
            macd_values[i] < macd_signal_values[i]
        )

        long_exit_condition = (
            rsi_values[i] < rsi_long_exit or
            close_prices[i] < sma_values[i] * sma_long_buffer or
            macd_values[i] < macd_signal_values[i]
        )

        short_exit_condition = (
            rsi_values[i] > rsi_short_exit or
            close_prices[i] > sma_values[i] * sma_short_buffer or
            macd_values[i] > macd_signal_values[i]
        )

        if current_signal == 0:
            if long_entry:
                current_signal = 1
            elif short_entry:
                current_signal = -1
        elif current_signal == 1:
            if long_exit_condition:
                current_signal = 0
        elif current_signal == -1:
            if short_exit_condition:
                current_signal = 0
        
        signals[i] = current_signal

    return signals

class LSTMSOLStrategy:
    """
    LSTM SOL 策略類 (簡化版, Numba 優化)
    """

    def __init__(self, name: str = "LSTM_SOL_Simplified"):
        self.name = name
        self.parameters = {}

    def generate_signals(self, df: pd.DataFrame, **params) -> pd.Series:
        """
        生成交易信號
        """
        symbol = params.get('symbol', 'SOL/USD')
        rsi_window = params.get('rsi_window', 14)
        rsi_long_threshold = params.get('rsi_long_threshold', 55)
        rsi_short_threshold = params.get('rsi_short_threshold', 45)
        rsi_exit_offset = params.get('rsi_exit_offset', 5)
        sma_period = params.get('sma_period', 20)
        sma_long_buffer = params.get('sma_long_buffer', 0.98)
        sma_short_buffer = params.get('sma_short_buffer', 1.02)
        macd_fast = params.get('macd_fast', 12)
        macd_slow = params.get('macd_slow', 26)
        macd_signal = params.get('macd_signal', 9)

        df_copy = df.copy()
        df_copy = calculate_indicators(
            df_copy,
            symbol_name=symbol,
            rsi_window=rsi_window,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            bb_window=sma_period
        )

        base_symbol = symbol.split("/")[0]
        close_col = f'{base_symbol}_Close'
        sma_col = f'SMA_{sma_period}'
        df_copy[sma_col] = df_copy[close_col].rolling(window=sma_period).mean()

        required_cols = [close_col, 'RSI', 'MACD', 'MACD_Signal', sma_col]
        if not all(col in df_copy.columns for col in required_cols):
            raise ValueError(f"缺少必要的指標列. 需要: {required_cols}")

        df_copy.ffill(inplace=True)
        df_copy.bfill(inplace=True)

        rsi_long_exit = max(0, rsi_short_threshold - rsi_exit_offset)
        rsi_short_exit = min(100, rsi_long_threshold + rsi_exit_offset)
        initial_bars = max(sma_period, macd_slow, rsi_window) + 5
        
        if len(df_copy) < initial_bars:
            return pd.Series(0, index=df_copy.index)

        signals_arr = _calculate_signals_nb(
            df_copy[close_col].to_numpy(),
            df_copy['RSI'].to_numpy(),
            df_copy['MACD'].to_numpy(),
            df_copy['MACD_Signal'].to_numpy(),
            df_copy[sma_col].to_numpy(),
            float(rsi_long_threshold),
            float(rsi_short_threshold),
            float(rsi_long_exit),
            float(rsi_short_exit),
            float(sma_long_buffer),
            float(sma_short_buffer),
            initial_bars
        )

        self.parameters = params
        return pd.Series(signals_arr, index=df.index)

    def __repr__(self):
        return f"<LSTMSOLStrategy(name='{self.name}', params={self.parameters})>"