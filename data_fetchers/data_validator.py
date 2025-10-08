"""
Data Validator
數據驗證器 - 採用Medium框架的驗證邏輯

功能：
1. 檢查OHLC關係的合法性
2. 檢查負價格和異常值
3. 檢查缺失值
4. 提供詳細的驗證報告
"""

import pandas as pd
import numpy as np


def validate_ohlc_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    驗證OHLC數據的完整性和合法性

    Args:
        df: OHLC數據，必須包含 open, high, low, close, volume 列
        verbose: 是否打印詳細驗證信息

    Returns:
        pd.DataFrame: 驗證後的數據（副本）

    Raises:
        ValueError: 如果數據格式不正確

    驗證項目：
        1. 必要列存在性檢查
        2. OHLC關係合法性（high >= open/close >= low）
        3. 負價格檢查
        4. 缺失值檢查
        5. 數據類型檢查
    """
    # 創建副本避免修改原始數據
    df = df.copy()

    # 1. 檢查必要列是否存在
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺少必要列: {missing_columns}")

    if verbose:
        print("=" * 60)
        print("數據驗證報告")
        print("=" * 60)

    # 2. 確保數據類型為數值型
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            if verbose:
                print(f"警告: {col} 列不是數值型，嘗試轉換...")
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                raise ValueError(f"無法將 {col} 轉換為數值型: {e}")

    # 3. 檢查缺失值
    missing_count = df[numeric_columns].isnull().sum()
    if missing_count.any():
        if verbose:
            print("\n警告: 發現缺失值")
            for col in numeric_columns:
                if missing_count[col] > 0:
                    print(f"  {col}: {missing_count[col]} 個缺失值")
        # 不自動刪除，保留NaN讓策略處理
    else:
        if verbose:
            print("\n[OK] 無缺失值")

    # 4. 檢查OHLC關係合法性
    # high 應該 >= open, close, low
    # low 應該 <= open, close, high
    invalid_high = (df['high'] < df['low']) | (df['high'] < df['open']) | (df['high'] < df['close'])
    invalid_low = (df['low'] > df['high']) | (df['low'] > df['open']) | (df['low'] > df['close'])

    invalid_rows = invalid_high | invalid_low
    invalid_count = invalid_rows.sum()

    if invalid_count > 0:
        if verbose:
            print(f"\n警告: 發現 {invalid_count} 行OHLC關係不合法")
            print(f"  佔總數據的 {invalid_count/len(df)*100:.2f}%")
            print(f"  建議檢查數據來源")
    else:
        if verbose:
            print("\n[OK] OHLC關係合法")

    # 5. 檢查負價格或零價格
    negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
    if negative_prices > 0:
        if verbose:
            print(f"\n警告: 發現 {negative_prices} 行包含負價格或零價格")
    else:
        if verbose:
            print("\n[OK] 無負價格或零價格")

    # 6. 檢查負成交量
    negative_volume = (df['volume'] < 0).sum()
    if negative_volume > 0:
        if verbose:
            print(f"\n警告: 發現 {negative_volume} 行包含負成交量")
    else:
        if verbose:
            print("\n[OK] 無負成交量")

    # 7. 數據統計摘要
    if verbose:
        print("\n" + "=" * 60)
        print("數據統計摘要")
        print("=" * 60)
        print(f"總行數: {len(df)}")
        if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
            try:
                print(f"時間範圍: {df.index.min()} 至 {df.index.max()}")
            except:
                pass
        print(f"\n價格範圍:")
        print(f"  最高價: ${df['high'].max():.2f}")
        print(f"  最低價: ${df['low'].min():.2f}")
        print(f"\n成交量:")
        print(f"  平均成交量: {df['volume'].mean():.2f}")
        print(f"  最大成交量: {df['volume'].max():.2f}")
        print("=" * 60)

    return df


def validate_ohlc_relationships(df: pd.DataFrame) -> pd.DataFrame:
    """
    僅驗證OHLC關係，用於輕量級驗證

    Args:
        df: OHLC數據

    Returns:
        pd.DataFrame: 包含驗證結果的布爾DataFrame

    Returns columns:
        - valid_high: high >= open, close, low
        - valid_low: low <= open, close, high
        - valid_ohlc: 整體是否合法
    """
    result = pd.DataFrame(index=df.index)

    result['valid_high'] = (df['high'] >= df['low']) & \
                           (df['high'] >= df['open']) & \
                           (df['high'] >= df['close'])

    result['valid_low'] = (df['low'] <= df['high']) & \
                          (df['low'] <= df['open']) & \
                          (df['low'] <= df['close'])

    result['valid_ohlc'] = result['valid_high'] & result['valid_low']

    return result


if __name__ == "__main__":
    # 測試範例
    print("數據驗證器測試")
    print("-" * 60)

    # 創建測試數據
    test_data = pd.DataFrame({
        'open': [100, 101, 99, 102],
        'high': [105, 106, 103, 107],
        'low': [98, 100, 97, 101],
        'close': [102, 103, 100, 105],
        'volume': [1000, 1100, 900, 1200]
    })

    print("測試數據:")
    print(test_data)
    print()

    # 驗證數據
    validated = validate_ohlc_data(test_data, verbose=True)

    print("\n測試完成!")
