"""
RSI-ATR Strategy
RSI-ATR 策略

策略原理：
- 進場：使用 RSI 指標判斷超買超賣
- 出場：使用 ATR 動態止損止盈

理論基礎：
RSI (Relative Strength Index) 是動量振盪指標，用於識別超買超賣區域。
ATR (Average True Range) 用於測量市場波動性，適合設定動態止損。

注意事項：
- RSI > 閾值時做多，RSI < 閾值時做空
- 使用 ATR 倍數作為止損距離
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
    rsi_prev: np.ndarray,
    atr_prev: np.ndarray,
    rsi_long_threshold: float,
    rsi_short_threshold: float,
    atr_multiplier: float,
    initial_bars: int
) -> np.ndarray:
    """
    使用 Numba 優化的信號計算函數。
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    current_signal = 0
    entry_price = 0.0
    stop_loss_price = 0.0

    for i in range(initial_bars, n):
        current_price = close[i]
        rsi_value = rsi_prev[i]
        atr_value = atr_prev[i]

        # 跳過無效數據
        if np.isnan(current_price) or np.isnan(rsi_value) or np.isnan(atr_value):
            signals[i] = current_signal
            continue

        # 無倉位時檢查進場條件
        if current_signal == 0:
            if rsi_value > rsi_long_threshold:
                current_signal = 1
                entry_price = current_price
                stop_loss_price = entry_price - atr_multiplier * atr_value
            elif rsi_value < rsi_short_threshold:
                current_signal = -1
                entry_price = current_price
                stop_loss_price = entry_price + atr_multiplier * atr_value

        # 持有多頭時檢查出場條件
        elif current_signal == 1:
            if current_price < stop_loss_price:
                current_signal = 0
                entry_price = 0.0
                stop_loss_price = 0.0

        # 持有空頭時檢查出場條件
        elif current_signal == -1:
            if current_price > stop_loss_price:
                current_signal = 0
                entry_price = 0.0
                stop_loss_price = 0.0

        signals[i] = current_signal
    
    return signals


class RSIATRStrategy:
    """
    RSI-ATR 策略類 (Numba 優化版)
    """

    def __init__(self, name: str = "RSI_ATR"):
        self.name = name
        self.parameters = {}

    def generate_signals(self, df: pd.DataFrame, **params) -> pd.Series:
        """
        生成交易信號
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing one or more required columns: {required_columns}")

        rsi_window = params.get('rsi_window', 14)
        rsi_long_threshold = params.get('rsi_long_threshold', 60)
        rsi_short_threshold = params.get('rsi_short_threshold', 40)
        atr_window = params.get('atr_window', 14)
        atr_multiplier = params.get('atr_multiplier', 2.0)

        if not (1 <= rsi_window <= 100 and isinstance(rsi_window, int)):
            raise ValueError(f"rsi_window 必須是 [1, 100] 範圍內的整數")
        if not (0 <= rsi_long_threshold <= 100):
            raise ValueError(f"rsi_long_threshold 必須在 [0, 100] 範圍內")
        if not (0 <= rsi_short_threshold <= 100):
            raise ValueError(f"rsi_short_threshold 必須在 [0, 100] 範圍內")
        if rsi_long_threshold <= rsi_short_threshold:
            raise ValueError("rsi_long_threshold 必須大於 rsi_short_threshold")
        if not (1 <= atr_window <= 100 and isinstance(atr_window, int)):
            raise ValueError(f"atr_window 必須是 [1, 100] 範圍內的整數")
        if not (0.1 <= atr_multiplier <= 10.0):
            raise ValueError(f"atr_multiplier 必須在 [0.1, 10.0] 範圍內")

        min_length = max(rsi_window, atr_window) + 10
        if len(df) < min_length:
            return pd.Series(0, index=df.index) # 如果數據不足，返回空信號而不是報錯

        df_copy = df.copy()

        df_copy['RSI'] = ta.momentum.RSIIndicator(df_copy['close'], window=rsi_window).rsi()
        df_copy['ATR'] = ta.volatility.AverageTrueRange(df_copy['high'], df_copy['low'], df_copy['close'], window=atr_window).average_true_range()

        df_copy['RSI_prev'] = df_copy['RSI'].shift(1)
        df_copy['ATR_prev'] = df_copy['ATR'].shift(1)

        # 填充 NaN 值
        df_copy.fillna(method='ffill', inplace=True)
        df_copy.fillna(method='bfill', inplace=True)

        # 調用 Numba 優化函數
        signals_arr = _calculate_signals_nb(
            df_copy['close'].to_numpy(),
            df_copy['RSI_prev'].to_numpy(),
            df_copy['ATR_prev'].to_numpy(),
            float(rsi_long_threshold),
            float(rsi_short_threshold),
            float(atr_multiplier),
            initial_bars=max(rsi_window, atr_window) + 1
        )

        self.parameters = params

        return pd.Series(signals_arr, index=df.index)

    def get_parameter_grid(self) -> Dict[str, Tuple]:
        return {
            'rsi_window': (10, 20, 2),
            'rsi_long_threshold': (55, 70, 5),
            'rsi_short_threshold': (30, 45, 5),
            'atr_window': (10, 20, 2),
            'atr_multiplier': (1.5, 3.0, 0.5)
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'rsi_window': 14,
            'rsi_long_threshold': 60,
            'rsi_short_threshold': 40,
            'atr_window': 14,
            'atr_multiplier': 2.0
        }

    def __call__(self, df: pd.DataFrame, **params) -> pd.Series:
        return self.generate_signals(df, **params)