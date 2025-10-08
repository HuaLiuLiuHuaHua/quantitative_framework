"""
Technical Indicators Calculator
技術指標計算模組

計算策略所需的所有技術指標，使用 ta 庫實現。

指標列表：
1. RSI (14) - 相對強弱指數
2. MACD (12, 26, 9) - 指數平滑異同移動平均線
3. Bollinger Bands (20) - 布林帶
4. OBV - 能量潮
5. ATR - 真實波幅
6. SMA (5, 10, 20) - 簡單移動平均線
"""

import pandas as pd
import numpy as np

try:
    import ta
except ImportError:
    raise ImportError(
        "缺少 ta 庫。請安裝: pip install ta"
    )


def calculate_indicators(
    df: pd.DataFrame,
    symbol_name: str = "SOL/USD",
    rsi_window: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_window: int = 20
) -> pd.DataFrame:
    """
    計算所有技術指標

    Args:
        df: OHLCV 數據，必須包含 open, high, low, close, volume
        symbol_name: 交易對名稱 (用於列名)
        rsi_window: RSI 窗口期
        macd_fast: MACD 快線週期
        macd_slow: MACD 慢線週期
        macd_signal: MACD 信號線週期
        bb_window: 布林帶窗口期

    Returns:
        pd.DataFrame: 包含所有指標的數據框
    """
    df = df.copy()

    # 提取基礎符號 (e.g., "SOL" from "SOL/USD")
    base_symbol = symbol_name.split("/")[0]

    # 重命名 close 列為符號專用列名 (如果需要)
    if 'close' in df.columns and f'{base_symbol}_Close' not in df.columns:
        df.rename(columns={'close': f'{base_symbol}_Close'}, inplace=True)

    close_col = f'{base_symbol}_Close'

    # 確保必要的列存在
    required_cols = ['open', 'high', 'low', close_col, 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要的列: {missing_cols}")

    # 1. RSI (相對強弱指數)
    df['RSI'] = ta.momentum.RSIIndicator(
        df[close_col],
        window=rsi_window
    ).rsi()

    # 2. MACD (指數平滑異同移動平均線)
    macd = ta.trend.MACD(
        close=df[close_col],
        window_fast=macd_fast,
        window_slow=macd_slow,
        window_sign=macd_signal
    )
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    # 3. Bollinger Bands (布林帶)
    bb = ta.volatility.BollingerBands(
        df[close_col],
        window=bb_window
    )
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()

    # 4. OBV (能量潮)
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
        df[close_col],
        df['volume']
    ).on_balance_volume()

    # 5. ATR (真實波幅)
    df['ATR'] = ta.volatility.AverageTrueRange(
        df['high'],
        df['low'],
        df[close_col]
    ).average_true_range()

    # 6. SMA (簡單移動平均線)
    df['SMA_5'] = df[close_col].rolling(window=5).mean()
    df['SMA_10'] = df[close_col].rolling(window=10).mean()
    df['SMA_20'] = df[close_col].rolling(window=20).mean()

    return df


def get_prediction_features():
    """
    獲取預測用的特徵列表 (12個)

    這些特徵會被用於 LSTM 模型輸入 (如果有模型的話)
    或用於簡化版策略的信號生成

    Returns:
        list: 特徵名稱列表
    """
    return [
        'open',
        'high',
        'low',
        'close',  # 注意：這裡使用 'close'，在實際使用時需要對應到 '{symbol}_Close'
        'volume',
        'RSI',
        'MACD',
        'MACD_Signal',
        'BB_Upper',
        'BB_Lower',
        'OBV',
        'ATR'
    ]


def get_strategy_features():
    """
    獲取策略邏輯用的特徵列表 (包含 SMA)

    這些特徵用於生成交易信號和執行策略邏輯

    Returns:
        list: 特徵名稱列表
    """
    prediction_features = get_prediction_features()
    return prediction_features + ['SMA_5', 'SMA_10', 'SMA_20']


if __name__ == "__main__":
    # 測試指標計算
    print("測試技術指標計算...")

    # 創建測試數據
    test_data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(110, 120, 100),
        'low': np.random.uniform(90, 100, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    })

    # 計算指標
    df_with_indicators = calculate_indicators(test_data, symbol_name="TEST/USD")

    print(f"\n計算完成！")
    print(f"原始數據形狀: {test_data.shape}")
    print(f"帶指標數據形狀: {df_with_indicators.shape}")
    print(f"\n新增指標列: {[col for col in df_with_indicators.columns if col not in test_data.columns]}")

    print(f"\n預測特徵 (12個): {get_prediction_features()}")
    print(f"\n策略特徵 (15個): {get_strategy_features()}")

    print(f"\n前5筆數據:")
    print(df_with_indicators.head())
