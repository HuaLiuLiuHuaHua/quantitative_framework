"""
CVILLIQ CTA Strategy (Single Asset Trend Following)
CVILLIQ CTA策略 (單一標的趨勢跟隨)

根據Coefficient of Variation of Illiquidity (CVILLIQ) 因子識別流動性機會
進場條件 (動態門檻):
- 多頭: CVILLIQ > 滾動高百分位 (資產流動性不確定性高)
- 空頭: CVILLIQ < 滾動低百分位 (資產流動性穩定)
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
    驗證策略參數 (動態門檻版本)

    Args:
        params: 參數字典
        df_len: 數據長度

    Raises:
        ValueError: 如果參數無效
    """
    errors = []

    # CVILLIQ因子計算窗口驗證
    window = int(params.get('window', 20))
    if window < 2:
        errors.append("window 必須 >= 2")
    if window >= df_len:
        errors.append(f"window ({window}) >= 數據長度 ({df_len})")

    # 動態門檻計算窗口驗證
    threshold_window = int(params.get('threshold_window', 100))
    if threshold_window < 10:
        errors.append("threshold_window 必須 >= 10")
    if threshold_window >= df_len:
        errors.append(f"threshold_window ({threshold_window}) >= 數據長度 ({df_len})")

    # 百分位數驗證
    long_entry_quantile = float(params.get('long_entry_quantile', 0.8))
    if not (0 < long_entry_quantile < 1):
        errors.append(f"long_entry_quantile 必須在 (0, 1) 之間, 得到 {long_entry_quantile}")

    short_entry_quantile = float(params.get('short_entry_quantile', 0.2))
    if not (0 < short_entry_quantile < 1):
        errors.append(f"short_entry_quantile 必須在 (0, 1) 之間, 得到 {short_entry_quantile}")
    
    # 槓桿驗證
    leverage = int(params.get('leverage', 1))
    if leverage < 1:
        errors.append(f"leverage 必須 >= 1, 得到 {leverage}")

    if errors:
        raise ValueError("無效參數:\n" + "\n".join(errors))


def _calculate_signals(
    cvilliq_prev: np.ndarray,
    cvilliq_high_threshold: np.ndarray,
    cvilliq_low_threshold: np.ndarray,
    initial_bars: int
) -> np.ndarray:
    """
    計算CVILLIQ信號 (進場-出場版本)

    邏輯:
    - 策略允許空倉 (signal=0)。
    - 多頭進場: CVILLIQ_prev > cvilliq_high_threshold
    - 多頭出場: 當持有多單時，CVILLIQ_prev <= cvilliq_high_threshold
    - 空頭進場: CVILLIQ_prev < cvilliq_low_threshold
    - 空頭出場: 當持有空單時，CVILLIQ_prev >= cvilliq_low_threshold

    Args:
        cvilliq_prev: 前一期 CVILLIQ 值 (shift(1) 後)
        cvilliq_high_threshold: 多頭進場閾值
        cvilliq_low_threshold: 空頭進場閾值
        initial_bars: 有效數據開始索引 (用於診斷，不再實際使用)

    Returns:
        np.ndarray: 交易信號 (1=多, -1=空, 0=空倉)
    """
    n = len(cvilliq_prev)
    # 初始化為全部空倉(0)
    signals = np.zeros(n, dtype=np.int8)

    for i in range(1, n):
        # 處理NaN值: 如果數據無效，平倉並等待
        if (np.isnan(cvilliq_prev[i]) or
            np.isnan(cvilliq_high_threshold[i]) or
            np.isnan(cvilliq_low_threshold[i])):
            signals[i] = 0  # 數據無效時平倉
            continue

        previous_position = signals[i-1]

        is_long_entry_condition_met = cvilliq_prev[i] > cvilliq_high_threshold[i]
        is_short_entry_condition_met = cvilliq_prev[i] < cvilliq_low_threshold[i]

        # 根據上一期倉位決定本期操作
        if previous_position == 1:  # 當前持有多單
            if not is_long_entry_condition_met:
                signals[i] = 0  # 平多倉
            else:
                signals[i] = 1  # 續抱多單
        elif previous_position == -1:  # 當前持有空單
            if not is_short_entry_condition_met:
                signals[i] = 0  # 平空倉
            else:
                signals[i] = -1  # 續抱空單
        else:  # 當前為空倉
            if is_long_entry_condition_met:
                signals[i] = 1  # 開多倉
            elif is_short_entry_condition_met:
                signals[i] = -1  # 開空倉
            else:
                signals[i] = 0  # 保持空倉

    return signals


class CVILLIQStrategy(BaseStrategy):
    """
    CVILLIQ CTA 策略類

    將CVILLIQ因子作為流動性機會識別指標
    """

    def __init__(self, name: str = "CVILLIQ"):
        super().__init__()
        self.name = name
        self.parameters = {}
        self.debug_data = {}

    def get_lookback(self) -> int:
        """返回策略所需的最小lookback期"""
        # 為簡化起見，返回一個足夠大的固定值，以容納所有滾動窗口
        return 250

    def generate_signals(self, df: pd.DataFrame, **params) -> pd.Series:
        """
        生成交易信號 (動態門檻版本)

        Args:
            df: OHLCV數據 (必須包含 'close' 和 'volume' 列)
            window: CVILLIQ計算窗口
            threshold_window: 動態門檻計算窗口
            long_entry_quantile: 做多進場百分位
            short_entry_quantile: 做空進場百分位
            leverage: 槓桿倍數

        Returns:
            pd.Series: 交易信號序列
        """
        # 驗證 DataFrame 包含必需列
        required_cols = ['close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame 缺少必需列: {set(required_cols) - set(df.columns)}")

        # 驗證參數
        validate_params(params, len(df))

        window = int(params.get('window', 20))
        threshold_window = int(params.get('threshold_window', 100))
        long_entry_quantile = float(params.get('long_entry_quantile', 0.8))
        short_entry_quantile = float(params.get('short_entry_quantile', 0.2))

        df_copy = df.copy()

        # ==================== 計算CVILLIQ指標 ====================
        returns = df_copy['close'].pct_change()
        illiq_daily = np.abs(returns.values) / (df_copy['volume'].values + 1e-9)
        illiq_series = pd.Series(illiq_daily, index=df_copy.index)

        illiq_std = illiq_series.rolling(window=window).std()
        illiq_mean = illiq_series.rolling(window=window).mean()

        df_copy['CVILLIQ'] = illiq_std / (illiq_mean + 1e-9)
        self.debug_data['cvilliq'] = df_copy['CVILLIQ']

        # ==================== 計算動態CVILLIQ門檻值 ====================
        min_periods = int(threshold_window * 0.8) # 至少需要80%的數據才計算

        cvilliq_high_threshold = df_copy['CVILLIQ'].rolling(
            window=threshold_window, min_periods=min_periods
        ).quantile(long_entry_quantile)

        cvilliq_low_threshold = df_copy['CVILLIQ'].rolling(
            window=threshold_window, min_periods=min_periods
        ).quantile(short_entry_quantile)

        # ==================== 防止lookahead bias ====================
        # 使用 shift(1) 確保信號只使用前一期的指標和門檻值
        df_copy['CVILLIQ_prev'] = df_copy['CVILLIQ'].shift(1)
        df_copy['CVILLIQ_High_Threshold_prev'] = cvilliq_high_threshold.shift(1)
        df_copy['CVILLIQ_Low_Threshold_prev'] = cvilliq_low_threshold.shift(1)

        # ==================== 計算信號 ====================
        # 對初期的NaN進行前向填充，確保信號盡早有效
        df_copy.ffill(inplace=True)

        signals_arr = _calculate_signals(
            df_copy['CVILLIQ_prev'].values,
            df_copy['CVILLIQ_High_Threshold_prev'].values,
            df_copy['CVILLIQ_Low_Threshold_prev'].values,
            initial_bars=window + threshold_window
        )

        self.parameters = params
        return pd.Series(signals_arr, index=df.index)

    def __call__(self, df: pd.DataFrame, **params) -> pd.Series:
        """使策略物件可調用"""
        return self.generate_signals(df, **params)
