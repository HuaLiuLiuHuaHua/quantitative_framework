"""
Donchian Channel Breakout Strategy
唐奇安通道突破策略

策略原理：
- 突破N日最高價時做多
- 跌破N日最低價時做空
- 使用MCPT-Main嚴格邏輯避免前視偏差

理論基礎：
Donchian Channel是最早的趨勢跟蹤系統之一，由Richard Donchian在1960年代開發。
該策略基於價格突破歷史區間的動量效應。

注意事項：
- 必須使用shift(1)避免前視偏差
- lookback-1是因為當前K棒不應包含在計算中
- 信號在收盤時產生，下一根K棒開盤執行
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import pandas as pd
from typing import Dict, Any, Tuple


class DonchianStrategy:
    """
    唐奇安通道突破策略

    使用MCPT-Main的嚴格邏輯避免前視偏差：
    - upper = close.shift(1).rolling(lookback).max()
    - lower = close.shift(1).rolling(lookback).min()

    參數：
        lookback: 回溯期（K棒數量）

    信號：
        1: 做多（突破上軌）
        -1: 做空（跌破下軌）
        0: 空倉（無信號）
    """

    def __init__(self, name: str = "Donchian"):
        """
        初始化策略

        Args:
            name: 策略名稱
        """
        self.name = name
        self.parameters = {}

    def generate_signals(self, df: pd.DataFrame, **params) -> pd.Series:
        """
        生成交易信號

        使用MCPT-Main的嚴格邏輯避免前視偏差：
        1. shift(1)先排除當前K棒，使用歷史收盤價
        2. rolling(lookback)計算前lookback根K棒的最高/最低價
        3. 當前收盤價突破通道時產生信號

        Args:
            df: OHLCV數據，必須包含'close'列
            lookback: 回溯期（默認20）

        Returns:
            pd.Series: 交易信號序列
                1 = 做多（突破上軌）
                -1 = 做空（跌破下軌）
                0 = 空倉（區間內）

        Example:
            >>> strategy = DonchianStrategy()
            >>> signals = strategy.generate_signals(df, lookback=20)
        """
        # 獲取參數
        lookback = params.get('lookback', 20)

        # 參數驗證
        if not isinstance(lookback, int) or lookback < 2:
            raise ValueError(f"lookback必須是大於等於2的整數，當前值: {lookback}")

        if len(df) < lookback:
            raise ValueError(f"數據長度({len(df)})小於lookback({lookback})")

        # 複製數據避免修改原始數據
        df = df.copy()

        # MCPT-Main嚴格邏輯：避免前視偏差
        # 1. shift(1)先排除當前K棒，使用前一根K棒的收盤價
        # 2. rolling(lookback)計算前lookback根K棒的最高/最低價
        # 這樣確保通道值完全基於歷史數據，不包含當前K棒信息
        upper = df['close'].shift(1).rolling(lookback).max()
        lower = df['close'].shift(1).rolling(lookback).min()

        # 初始化信號為0
        signals = pd.Series(0, index=df.index)

        # 突破上軌
        signals[df['close'] > upper] = 1

        # 跌破下軌
        signals[df['close'] < lower] = -1

        # 在通道內：保持前一個信號（趨勢跟蹤）
        # 使用forward fill保持倉位
        signals = signals.replace(0, pd.NA).ffill().fillna(0)

        return signals

    def get_parameter_grid(self) -> Dict[str, Tuple[int, int, int]]:
        """
        獲取參數優化網格

        Returns:
            Dict: 參數網格字典 {參數名: (最小值, 最大值, 步長)}

        Example:
            >>> strategy = DonchianStrategy()
            >>> grid = strategy.get_parameter_grid()
            >>> print(grid)  # {'lookback': (10, 100, 5)}
        """
        return {
            'lookback': (10, 100, 5)  # 回溯期從10到100，步長5
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        """
        獲取默認參數

        Returns:
            Dict: 默認參數字典

        Example:
            >>> strategy = DonchianStrategy()
            >>> params = strategy.get_default_parameters()
            >>> print(params)  # {'lookback': 20}
        """
        return {
            'lookback': 20  # 默認使用20日通道
        }

    def validate_parameters(self, **params) -> bool:
        """
        驗證參數有效性

        Args:
            **params: 策略參數

        Returns:
            bool: 參數是否有效

        Example:
            >>> strategy = DonchianStrategy()
            >>> is_valid = strategy.validate_parameters(lookback=20)
            >>> print(is_valid)  # True
        """
        lookback = params.get('lookback', 20)

        # 檢查類型
        if not isinstance(lookback, (int, float)):
            return False

        # 檢查範圍
        if lookback < 2:
            return False

        return True

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        獲取策略信息

        Returns:
            Dict: 策略信息字典
        """
        return {
            'name': self.name,
            'description': self.__doc__,
            'parameters': self.get_default_parameters(),
            'parameter_grid': self.get_parameter_grid(),
            'signal_types': {
                1: '做多（突破上軌）',
                -1: '做空（跌破下軌）',
                0: '空倉（區間內）'
            }
        }

    def __call__(self, df: pd.DataFrame, **params) -> pd.Series:
        """使策略對象可調用"""
        return self.generate_signals(df, **params)

    def __str__(self) -> str:
        return f"{self.name}Strategy"

    def __repr__(self) -> str:
        return f"DonchianStrategy(name='{self.name}')"


if __name__ == "__main__":
    # 測試策略
    print("=" * 70)
    print("Donchian策略測試")
    print("=" * 70)

    # 創建測試數據
    import numpy as np
    np.random.seed(42)

    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    prices = 50000 * (1 + np.random.randn(100) * 0.02).cumprod()

    test_data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(100) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(100)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(100)) * 0.01),
        'close': prices,
        'volume': np.random.uniform(100, 1000, 100)
    }, index=dates)

    # 測試策略
    strategy = DonchianStrategy()

    print(f"\n策略名稱: {strategy.name}")
    print(f"默認參數: {strategy.get_default_parameters()}")
    print(f"參數網格: {strategy.get_parameter_grid()}")

    # 生成信號
    signals = strategy.generate_signals(test_data, lookback=20)

    print(f"\n信號統計:")
    print(f"做多信號: {(signals == 1).sum()}")
    print(f"做空信號: {(signals == -1).sum()}")
    print(f"空倉信號: {(signals == 0).sum()}")

    print(f"\n前10個信號:")
    print(signals.head(10))

    # 測試參數驗證
    print(f"\n參數驗證測試:")
    print(f"lookback=20: {strategy.validate_parameters(lookback=20)}")
    print(f"lookback=1: {strategy.validate_parameters(lookback=1)}")
    print(f"lookback='invalid': {strategy.validate_parameters(lookback='invalid')}")

    print("\n測試完成!")
    print("=" * 70)
