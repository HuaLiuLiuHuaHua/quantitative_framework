"""
MA Crossover Strategy
移動平均線交叉策略

策略原理：
- 計算短期和長期簡單移動平均線（SMA）
- 短期均線上穿長期均線時做多（金叉）
- 短期均線下穿長期均線時做空（死叉）
- 使用嚴格邏輯避免前視偏差

理論基礎：
移動平均交叉策略是最經典的趨勢跟蹤策略之一。金叉和死叉信號被廣泛用於
識別趨勢轉折點。短期均線反映近期價格動向，長期均線代表整體趨勢。

注意事項：
- 必須使用shift(1)避免前視偏差
- 信號在收盤時產生，下一根K棒開盤執行
- 在震盪市場中可能產生頻繁假信號
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


class MACrossoverStrategy:
    """
    移動平均線交叉策略

    使用嚴格邏輯避免前視偏差：
    - short_ma = close.rolling(short_period).mean()
    - long_ma = close.rolling(long_period).mean()
    - 通過比較當前和前一根K棒的均線值判斷交叉

    參數：
        short_period: 短期均線週期（K棒數量）
        long_period: 長期均線週期（K棒數量）

    信號：
        1: 做多（金叉 - 短期均線上穿長期均線）
        -1: 做空（死叉 - 短期均線下穿長期均線）
        0: 保持前一個信號
    """

    def __init__(self, name: str = "MA_Crossover"):
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

        使用嚴格邏輯避免前視偏差：
        1. 計算短期和長期移動平均線
        2. 比較當前K棒和前一根K棒的均線位置判斷交叉
        3. 交叉發生時產生信號（信號在下一根K棒開盤時執行）

        Args:
            df: OHLCV數據，必須包含'close'列
            short_period: 短期均線週期（默認10）
            long_period: 長期均線週期（默認30）

        Returns:
            pd.Series: 交易信號序列
                1 = 做多（金叉）
                -1 = 做空（死叉）
                0 = 保持前一個信號

        Example:
            >>> strategy = MACrossoverStrategy()
            >>> signals = strategy.generate_signals(df, short_period=10, long_period=30)
        """
        # 獲取參數
        short_period = params.get('short_period', 10)
        long_period = params.get('long_period', 30)

        # 參數驗證
        if not isinstance(short_period, int) or short_period < 1:
            raise ValueError(f"short_period必須是大於等於1的整數，當前值: {short_period}")

        if not isinstance(long_period, int) or long_period < 1:
            raise ValueError(f"long_period必須是大於等於1的整數，當前值: {long_period}")

        if short_period >= long_period:
            raise ValueError(f"short_period({short_period})必須小於long_period({long_period})")

        if len(df) < long_period:
            raise ValueError(f"數據長度({len(df)})小於long_period({long_period})")

        # 複製數據避免修改原始數據
        df = df.copy()

        # 計算移動平均線（不shift，先計算當前均線值）
        short_ma = df['close'].rolling(window=short_period).mean()
        long_ma = df['close'].rolling(window=long_period).mean()

        # 初始化信號為0
        signals = pd.Series(0, index=df.index)

        # 計算前一根K棒的均線位置（用於判斷交叉）
        # shift(1)確保我們比較的是前一根K棒的均線值
        short_ma_prev = short_ma.shift(1)
        long_ma_prev = long_ma.shift(1)

        # 金叉：短期均線從下方上穿長期均線 -> 做多
        # 條件：前一根K棒short_ma <= long_ma，當前K棒short_ma > long_ma
        # 使用當前K棒的均線(short_ma, long_ma)和前一根的均線(short_ma_prev, long_ma_prev)比較
        golden_cross = (short_ma_prev <= long_ma_prev) & (short_ma > long_ma)
        signals[golden_cross] = 1

        # 死叉：短期均線從上方下穿長期均線 -> 做空
        # 條件：前一根K棒short_ma >= long_ma，當前K棒short_ma < long_ma
        death_cross = (short_ma_prev >= long_ma_prev) & (short_ma < long_ma)
        signals[death_cross] = -1

        # 保持信號：使用forward fill保持倉位
        signals = signals.replace(0, pd.NA).ffill().fillna(0)

        return signals

    def get_parameter_grid(self) -> Dict[str, Tuple[int, int, int]]:
        """
        獲取參數優化網格

        Returns:
            Dict: 參數網格字典 {參數名: (最小值, 最大值, 步長)}

        Example:
            >>> strategy = MACrossoverStrategy()
            >>> grid = strategy.get_parameter_grid()
            >>> print(grid)
            {'short_period': (5, 30, 5), 'long_period': (20, 100, 10)}
        """
        return {
            'short_period': (5, 30, 5),    # 短期均線從5到30，步長5
            'long_period': (20, 100, 10)   # 長期均線從20到100，步長10
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        """
        獲取默認參數

        Returns:
            Dict: 默認參數字典

        Example:
            >>> strategy = MACrossoverStrategy()
            >>> params = strategy.get_default_parameters()
            >>> print(params)  # {'short_period': 10, 'long_period': 30}
        """
        return {
            'short_period': 10,  # 默認10日短期均線
            'long_period': 30    # 默認30日長期均線
        }

    def validate_parameters(self, **params) -> bool:
        """
        驗證參數有效性

        Args:
            **params: 策略參數

        Returns:
            bool: 參數是否有效

        Example:
            >>> strategy = MACrossoverStrategy()
            >>> is_valid = strategy.validate_parameters(short_period=10, long_period=30)
            >>> print(is_valid)  # True
        """
        short_period = params.get('short_period', 10)
        long_period = params.get('long_period', 30)

        # 檢查類型
        if not isinstance(short_period, (int, float)):
            return False
        if not isinstance(long_period, (int, float)):
            return False

        # 檢查範圍
        if short_period < 1:
            return False
        if long_period < 1:
            return False

        # 檢查邏輯關係
        if short_period >= long_period:
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
                1: '做多（金叉）',
                -1: '做空（死叉）',
                0: '保持倉位'
            }
        }

    def __call__(self, df: pd.DataFrame, **params) -> pd.Series:
        """使策略對象可調用"""
        return self.generate_signals(df, **params)

    def __str__(self) -> str:
        return f"{self.name}Strategy"

    def __repr__(self) -> str:
        return f"MACrossoverStrategy(name='{self.name}')"


if __name__ == "__main__":
    # 測試策略
    print("=" * 70)
    print("MA Crossover策略測試")
    print("=" * 70)

    # 創建測試數據
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
    strategy = MACrossoverStrategy()

    print(f"\n策略名稱: {strategy.name}")
    print(f"默認參數: {strategy.get_default_parameters()}")
    print(f"參數網格: {strategy.get_parameter_grid()}")

    # 生成信號
    signals = strategy.generate_signals(test_data, short_period=10, long_period=30)

    print(f"\n信號統計:")
    print(f"做多信號: {(signals == 1).sum()}")
    print(f"做空信號: {(signals == -1).sum()}")
    print(f"空倉信號: {(signals == 0).sum()}")

    print(f"\n前20個信號:")
    print(signals.head(20))

    # 測試參數驗證
    print(f"\n參數驗證測試:")
    print(f"short=10, long=30: {strategy.validate_parameters(short_period=10, long_period=30)}")
    print(f"short=30, long=10: {strategy.validate_parameters(short_period=30, long_period=10)}")
    print(f"short='invalid': {strategy.validate_parameters(short_period='invalid', long_period=30)}")

    print("\n測試完成!")
    print("=" * 70)
