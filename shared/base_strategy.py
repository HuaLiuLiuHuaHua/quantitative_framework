"""
Base Strategy Class
策略基類 - 所有策略必須繼承此類

設計原則：
- 最簡化設計，只需實現 generate_signals() 方法
- 策略邏輯與回測/優化/驗證分離
"""

import pandas as pd
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """
    策略基類

    所有自定義策略必須繼承此類並實現 generate_signals() 方法
    """

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, **params) -> pd.Series:
        """
        生成交易信號（必須實現）

        Args:
            df: OHLCV數據，必須包含以下列：
                - open: 開盤價
                - high: 最高價
                - low: 最低價
                - close: 收盤價
                - volume: 成交量
                - timestamp或index: 時間索引
            **params: 策略參數（如 lookback=20, threshold=0.02 等）

        Returns:
            pd.Series: 交易信號序列，與df同索引
                - 1: 做多（買入）
                - -1: 做空（賣出）
                - 0: 空倉（無持倉）

        注意事項：
            1. 必須避免前視偏差（look-ahead bias）
               - 使用 shift(1) 確保信號基於前一期數據
               - 使用 rolling(n-1) 不包括當前K線
            2. 信號應為整數類型（1, -1, 0）
            3. 返回的Series長度必須與輸入df相同
            4. 處理NaN值（初始幾根K線可能無法計算指標）

        Example:
            >>> def generate_signals(self, df, lookback=20):
            >>>     # 計算通道（嚴格避免前視偏差）
            >>>     # shift(1)先排除當前K棒，再rolling計算歷史最高/最低價
            >>>     upper = df['close'].shift(1).rolling(lookback).max()
            >>>     lower = df['close'].shift(1).rolling(lookback).min()
            >>>
            >>>     # 生成信號
            >>>     signal = pd.Series(0, index=df.index)
            >>>     signal.loc[df['close'] > upper] = 1
            >>>     signal.loc[df['close'] < lower] = -1
            >>>
            >>>     # 前向填充保持倉位
            >>>     signal = signal.replace(0, pd.NA).ffill().fillna(0)
            >>>
            >>>     return signal
        """
        raise NotImplementedError("子類必須實現 generate_signals() 方法")
