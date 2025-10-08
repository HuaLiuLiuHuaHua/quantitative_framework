"""
Bollinger Bands-ATR Strategy
布林帶-ATR 策略

策略原理：
- 進場：使用 Bollinger Bands 指標判斷超買超賣
- 出場：使用 ATR 動態止損止盈

理論基礎：
Bollinger Bands 是波動率指標，由中軌（移動平均線）和上下軌（標準差）組成。
價格觸及下軌表示超賣，觸及上軌表示超買。
ATR (Average True Range) 用於測量市場波動性，適合設定動態止損。

注意事項：
- 價格 < 下軌時做多，價格 > 上軌時做空
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


class BBATRStrategy:
    """
    Bollinger Bands-ATR 策略類

    進場使用 Bollinger Bands 指標，出場使用 ATR 止損。

    參數：
        bb_window: Bollinger Bands 計算週期
        bb_std: Bollinger Bands 標準差倍數
        atr_window: ATR 計算週期
        atr_multiplier: ATR 止損倍數

    信號：
        1: 做多
        -1: 做空
        0: 空倉
    """

    def __init__(self, name: str = "BB_ATR"):
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

        Args:
            df: OHLCV數據，必須包含 open, high, low, close, volume
            bb_window: BB 窗口期 (range: [2, 100], 默認 20)
            bb_std: BB 標準差倍數 (range: [0.5, 5.0], 默認 2.0)
            atr_window: ATR 窗口期 (range: [1, 100], 默認 14)
            atr_multiplier: ATR 止損倍數 (range: [0.1, 10.0], 默認 2.0)

        Returns:
            pd.Series: 交易信號序列
                1 = 做多
                -1 = 做空
                0 = 空倉
        """
        # 驗證輸入數據
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # 獲取參數並確保類型正確
        bb_window = int(params.get('bb_window', 20))
        bb_std = float(params.get('bb_std', 2.0))
        atr_window = int(params.get('atr_window', 14))
        atr_multiplier = float(params.get('atr_multiplier', 2.0))

        # 驗證數據長度
        min_length = max(bb_window, atr_window) + 10
        if len(df) < min_length:
            raise ValueError(f"Insufficient data: need at least {min_length} bars, got {len(df)}")

        # 複製數據避免修改原始數據
        df = df.copy()

        # 計算 Bollinger Bands
        bb = ta.volatility.BollingerBands(
            df['close'],
            window=bb_window,
            window_dev=bb_std
        )
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Middle'] = bb.bollinger_mavg()

        # 計算 ATR
        df['ATR'] = ta.volatility.AverageTrueRange(
            df['high'],
            df['low'],
            df['close'],
            window=atr_window
        ).average_true_range()

        # 使用 shift(1) 避免前視偏差
        df['BB_Upper_prev'] = df['BB_Upper'].shift(1)
        df['BB_Lower_prev'] = df['BB_Lower'].shift(1)
        df['ATR_prev'] = df['ATR'].shift(1)

        # 初始化信號為0
        signals = pd.Series(0, index=df.index)

        # 向前填充 NaN 值
        df = df.ffill().bfill()

        # 生成信號邏輯
        current_signal = 0
        entry_price = None
        stop_loss_price = None

        start_index = max(bb_window, atr_window) + 1

        for i in range(start_index, len(df)):
            current_price = df['close'].iloc[i]
            # 使用前一根 K 棒的指標值避免前視偏差
            bb_upper = df['BB_Upper_prev'].iloc[i]
            bb_lower = df['BB_Lower_prev'].iloc[i]
            atr_value = df['ATR_prev'].iloc[i]

            # 跳過無效數據
            if pd.isna(current_price) or pd.isna(bb_upper) or pd.isna(bb_lower) or pd.isna(atr_value):
                signals.iloc[i] = current_signal
                continue

            # 無倉位時檢查進場條件
            if current_signal == 0:
                # 價格跌破下軌（超賣）
                if current_price < bb_lower:
                    current_signal = -1
                    entry_price = current_price
                    # 計算止損價格（使用入場時的 ATR）
                    stop_loss_price = entry_price - atr_multiplier * atr_value
                # 價格突破上軌（超買）
                elif current_price > bb_upper:
                    current_signal = 1
                    entry_price = current_price
                    # 計算止損價格（使用入場時的 ATR）
                    stop_loss_price = entry_price + atr_multiplier * atr_value

            # 持有多頭時檢查出場條件
            elif current_signal == 1 and entry_price is not None and stop_loss_price is not None:
                # ATR 止損：價格跌破止損價
                if current_price < stop_loss_price:
                    current_signal = 0
                    entry_price = None
                    stop_loss_price = None

            # 持有空頭時檢查出場條件
            elif current_signal == -1 and entry_price is not None and stop_loss_price is not None:
                # ATR 止損：價格突破止損價
                if current_price > stop_loss_price:
                    current_signal = 0
                    entry_price = None
                    stop_loss_price = None

            signals.iloc[i] = current_signal

        # 保存參數
        self.parameters = {
            'bb_window': bb_window,
            'bb_std': bb_std,
            'atr_window': atr_window,
            'atr_multiplier': atr_multiplier
        }

        return signals

    def get_parameter_grid(self) -> Dict[str, Tuple]:
        """
        獲取參數優化網格

        Returns:
            Dict: 參數網格字典
        """
        return {
            'bb_window': (15, 25, 5),
            'bb_std': (1.5, 2.5, 0.5),
            'atr_window': (10, 20, 2),
            'atr_multiplier': (1.5, 3.0, 0.5)
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        """
        獲取默認參數

        Returns:
            Dict: 默認參數字典
        """
        return {
            'bb_window': 20,
            'bb_std': 2.0,
            'atr_window': 14,
            'atr_multiplier': 2.0
        }

    def validate_parameters(self, **params) -> bool:
        """
        驗證參數有效性

        Args:
            **params: 策略參數

        Returns:
            bool: 參數是否有效
        """
        try:
            bb_window = params.get('bb_window', 20)
            bb_std = params.get('bb_std', 2.0)
            atr_window = params.get('atr_window', 14)
            atr_multiplier = params.get('atr_multiplier', 2.0)

            # 檢查類型和範圍
            if not (isinstance(bb_window, (int, float)) and 2 <= bb_window <= 100):
                return False
            if not (isinstance(bb_std, (int, float)) and 0.5 <= bb_std <= 5.0):
                return False
            if not (isinstance(atr_window, (int, float)) and 1 <= atr_window <= 100):
                return False
            if not (isinstance(atr_multiplier, (int, float)) and 0.1 <= atr_multiplier <= 10.0):
                return False

            return True
        except (ValueError, TypeError):
            return False

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
                1: '做多（價格跌破布林帶下軌）',
                -1: '做空（價格突破布林帶上軌）',
                0: '空倉（ATR止損出場）'
            }
        }

    def __call__(self, df: pd.DataFrame, **params) -> pd.Series:
        """使策略對象可調用"""
        return self.generate_signals(df, **params)

    def __str__(self) -> str:
        return f"{self.name}Strategy"

    def __repr__(self) -> str:
        return f"BBATRStrategy(name='{self.name}')"


if __name__ == "__main__":
    # 測試策略
    print("=" * 70)
    print("Bollinger Bands-ATR策略測試")
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
    strategy = BBATRStrategy()

    print(f"\n策略名稱: {strategy.name}")
    print(f"默認參數: {strategy.get_default_parameters()}")

    # 生成信號
    signals = strategy.generate_signals(test_data)

    print(f"\n信號統計:")
    print(f"做多信號: {(signals == 1).sum()}")
    print(f"做空信號: {(signals == -1).sum()}")
    print(f"空倉信號: {(signals == 0).sum()}")

    print(f"\n前20個信號:")
    print(signals.head(20))

    print("\n測試完成!")
    print("=" * 70)
