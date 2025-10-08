"""
Bar Permutation for MCPT-Main
K線排列組合模塊 - 實作MCPT-Main論文的高級排列邏輯

理論基礎（White 2000）：
1. 將對數價格分解為兩個獨立部分：
   - 跳空組件 (r_o): log(open_t / close_t-1)
   - 日內組件 (r_h, r_l, r_c): 高、低、收相對於開盤的變化

2. 排列策略：
   - perm1: 僅對日內組件排列（保留跳空順序）
   - perm2: 僅對跳空組件排列（保留日內順序）
   - 兩種排列都保留OHLC的內部關係

3. 為什麼這樣做：
   - 保留市場的時間序列結構
   - 避免破壞K線內部的價格關係
   - 生成更現實的排列序列

參考文獻：
White, H. (2000). "A Reality Check for Data Snooping." Econometrica, 68(5), 1097-1126.
"""

import pandas as pd
import numpy as np
from typing import Optional


def advanced_permutation(
    df: pd.DataFrame,
    start_index: int = 0,
    seed: Optional[int] = None,
    mode: str = 'both'
) -> pd.DataFrame:
    """
    執行高級K線排列（MCPT-Main方法）

    將價格數據分解為跳空和日內組件，並進行獨立排列，
    保留OHLC內部關係和部分時間序列結構。

    Args:
        df: OHLCV數據DataFrame，必須包含 ['open', 'high', 'low', 'close'] 列
        start_index: 開始排列的索引位置（默認0，從頭開始）
        seed: 隨機種子（用於可重現性）
        mode: 排列模式
            - 'both': 同時排列跳空和日內（默認）
            - 'intraday': 僅排列日內組件
            - 'gap': 僅排列跳空組件

    Returns:
        pd.DataFrame: 排列後的OHLCV數據，保持原有的列結構

    Example:
        >>> # 基本使用
        >>> permuted_df = advanced_permutation(df, start_index=100, seed=42)

        >>> # 僅排列日內組件
        >>> permuted_df = advanced_permutation(df, mode='intraday')

        >>> # 用於MCPT測試
        >>> for i in range(1000):
        ...     perm_df = advanced_permutation(df, start_index=100, seed=i)
        ...     # 在排列數據上運行策略

    注意：
        - start_index之前的數據保持不變（用於指標計算）
        - 排列保留每根K線的OHLC關係
        - 返回的DataFrame保持原有的時間索引
    """
    if seed is not None:
        np.random.seed(seed)

    # 複製數據以避免修改原始數據
    df_perm = df.copy()

    # 確保start_index有效
    if start_index >= len(df) - 1:
        return df_perm

    # 提取需要排列的部分
    df_to_permute = df.iloc[start_index:].copy()
    n = len(df_to_permute)

    if n <= 1:
        return df_perm

    # ==================== 步驟1: 計算對數價格 ====================
    # 注意：需要使用start_index-1的收盤價來計算第一個跳空
    if start_index > 0:
        prev_close = df.iloc[start_index - 1]['close']
    else:
        prev_close = df_to_permute.iloc[0]['open']

    # 計算對數價格
    log_open = np.log(df_to_permute['open'].values)
    log_high = np.log(df_to_permute['high'].values)
    log_low = np.log(df_to_permute['low'].values)
    log_close = np.log(df_to_permute['close'].values)

    # ==================== 步驟2: 分解為跳空和日內組件 ====================
    # 跳空組件 (r_o): log(open_t / close_t-1)
    prev_log_close = np.concatenate([[np.log(prev_close)], log_close[:-1]])
    r_o = log_open - prev_log_close

    # 日內組件（相對於開盤價）
    r_h = log_high - log_open  # 高點相對開盤
    r_l = log_low - log_open   # 低點相對開盤
    r_c = log_close - log_open # 收盤相對開盤

    # ==================== 步驟3: 執行排列 ====================
    if mode in ['both', 'gap']:
        # 排列跳空組件
        r_o_perm = r_o.copy()
        np.random.shuffle(r_o_perm)
    else:
        r_o_perm = r_o

    if mode in ['both', 'intraday']:
        # 排列日內組件（保持三個組件的對應關係）
        perm_indices = np.random.permutation(n)
        r_h_perm = r_h[perm_indices]
        r_l_perm = r_l[perm_indices]
        r_c_perm = r_c[perm_indices]
    else:
        r_h_perm = r_h
        r_l_perm = r_l
        r_c_perm = r_c

    # ==================== 步驟4: 重構價格 ====================
    # 重構對數價格
    log_open_perm = prev_log_close[0] + np.cumsum(r_o_perm)
    log_high_perm = log_open_perm + r_h_perm
    log_low_perm = log_open_perm + r_l_perm
    log_close_perm = log_open_perm + r_c_perm

    # 轉換回原始價格
    open_perm = np.exp(log_open_perm)
    high_perm = np.exp(log_high_perm)
    low_perm = np.exp(log_low_perm)
    close_perm = np.exp(log_close_perm)

    # ==================== 步驟5: 確保OHLC關係正確 ====================
    # 確保 high >= max(open, close) 且 low <= min(open, close)
    # 使用向量化操作替代循環以提升性能
    high_perm = np.maximum.reduce([high_perm, open_perm, close_perm])
    low_perm = np.minimum.reduce([low_perm, open_perm, close_perm])

    # ==================== 步驟6: 更新DataFrame ====================
    df_perm.iloc[start_index:, df_perm.columns.get_loc('open')] = open_perm
    df_perm.iloc[start_index:, df_perm.columns.get_loc('high')] = high_perm
    df_perm.iloc[start_index:, df_perm.columns.get_loc('low')] = low_perm
    df_perm.iloc[start_index:, df_perm.columns.get_loc('close')] = close_perm

    # Volume保持不變（或也可以排列，根據需求）
    # 這裡選擇保持不變，因為volume與價格的關係較弱

    return df_perm


def simple_permutation(
    df: pd.DataFrame,
    start_index: int = 0,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    簡單K線排列（整根K線排列）

    直接對整根K線進行排列，不分解為跳空和日內組件。
    這是較簡單但較不保守的排列方法。

    Args:
        df: OHLCV數據DataFrame
        start_index: 開始排列的索引位置
        seed: 隨機種子

    Returns:
        pd.DataFrame: 排列後的OHLCV數據

    Example:
        >>> permuted_df = simple_permutation(df, start_index=100, seed=42)

    注意：
        這種方法會破壞跳空的時間順序，可能生成不太現實的序列
    """
    if seed is not None:
        np.random.seed(seed)

    df_perm = df.copy()

    if start_index >= len(df) - 1:
        return df_perm

    # 獲取需要排列的部分
    df_to_permute = df.iloc[start_index:].copy()

    # 生成隨機排列索引
    perm_indices = np.random.permutation(len(df_to_permute))

    # 對OHLCV進行排列
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df_to_permute.columns:
            df_perm.iloc[start_index:, df_perm.columns.get_loc(col)] = \
                df_to_permute[col].iloc[perm_indices].values

    return df_perm


def validate_ohlc(df: pd.DataFrame) -> bool:
    """
    驗證OHLC數據的有效性

    檢查以下條件：
    1. high >= open
    2. high >= close
    3. low <= open
    4. low <= close
    5. 所有價格 > 0

    Args:
        df: OHLCV數據DataFrame

    Returns:
        bool: 數據是否有效
    """
    if df.empty:
        return False

    # 檢查所有價格為正
    if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        return False

    # 檢查OHLC關係
    valid_high = (df['high'] >= df['open']) & (df['high'] >= df['close'])
    valid_low = (df['low'] <= df['open']) & (df['low'] <= df['close'])

    return valid_high.all() and valid_low.all()


if __name__ == "__main__":
    # 測試Bar Permutation
    print("Bar Permutation測試\n")

    # 創建測試數據
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='1H')

    # 生成真實的OHLC數據
    returns = np.random.normal(0.0001, 0.02, 1000)
    close_prices = 30000 * (1 + returns).cumprod()

    data = pd.DataFrame({
        'open': close_prices * (1 + np.random.normal(0, 0.001, 1000)),
        'high': close_prices * (1 + np.abs(np.random.normal(0.002, 0.001, 1000))),
        'low': close_prices * (1 - np.abs(np.random.normal(0.002, 0.001, 1000))),
        'close': close_prices,
        'volume': np.random.uniform(100, 1000, 1000)
    }, index=dates)

    # 確保OHLC關係正確
    for i in range(len(data)):
        data.iloc[i, data.columns.get_loc('high')] = max(
            data.iloc[i]['high'],
            data.iloc[i]['open'],
            data.iloc[i]['close']
        )
        data.iloc[i, data.columns.get_loc('low')] = min(
            data.iloc[i]['low'],
            data.iloc[i]['open'],
            data.iloc[i]['close']
        )

    print("原始數據統計:")
    print(f"數據點數: {len(data)}")
    print(f"收盤價範圍: {data['close'].min():.2f} - {data['close'].max():.2f}")
    print(f"OHLC有效性: {validate_ohlc(data)}")
    print()

    # 測試高級排列
    print("=" * 70)
    print("測試1: 高級排列（同時排列跳空和日內）")
    print("=" * 70)
    perm_df = advanced_permutation(data, start_index=100, seed=123)
    print(f"排列後收盤價範圍: {perm_df['close'].min():.2f} - {perm_df['close'].max():.2f}")
    print(f"OHLC有效性: {validate_ohlc(perm_df)}")
    print(f"前100根K線是否保持不變: {data.iloc[:100].equals(perm_df.iloc[:100])}")
    print()

    # 測試僅日內排列
    print("=" * 70)
    print("測試2: 僅排列日內組件")
    print("=" * 70)
    perm_intraday = advanced_permutation(data, start_index=100, seed=123, mode='intraday')
    print(f"OHLC有效性: {validate_ohlc(perm_intraday)}")
    print()

    # 測試僅跳空排列
    print("=" * 70)
    print("測試3: 僅排列跳空組件")
    print("=" * 70)
    perm_gap = advanced_permutation(data, start_index=100, seed=123, mode='gap')
    print(f"OHLC有效性: {validate_ohlc(perm_gap)}")
    print()

    # 測試簡單排列
    print("=" * 70)
    print("測試4: 簡單排列（整根K線）")
    print("=" * 70)
    simple_perm = simple_permutation(data, start_index=100, seed=123)
    print(f"OHLC有效性: {validate_ohlc(simple_perm)}")
    print()

    # 比較不同排列方法的統計特性
    print("=" * 70)
    print("統計特性比較")
    print("=" * 70)

    original_returns = data['close'].pct_change().dropna()
    perm_returns = perm_df['close'].pct_change().dropna()
    intraday_returns = perm_intraday['close'].pct_change().dropna()
    gap_returns = perm_gap['close'].pct_change().dropna()
    simple_returns = simple_perm['close'].pct_change().dropna()

    print(f"{'指標':<20} {'原始':<12} {'高級排列':<12} {'日內排列':<12} {'跳空排列':<12} {'簡單排列':<12}")
    print("-" * 80)
    print(f"{'平均收益率':<20} {original_returns.mean():>11.6f} {perm_returns.mean():>11.6f} "
          f"{intraday_returns.mean():>11.6f} {gap_returns.mean():>11.6f} {simple_returns.mean():>11.6f}")
    print(f"{'波動率':<20} {original_returns.std():>11.6f} {perm_returns.std():>11.6f} "
          f"{intraday_returns.std():>11.6f} {gap_returns.std():>11.6f} {simple_returns.std():>11.6f}")

    # 測試多次排列的穩定性
    print("\n" + "=" * 70)
    print("測試5: 多次排列穩定性（100次）")
    print("=" * 70)

    volatilities = []
    for i in range(100):
        perm_test = advanced_permutation(data, start_index=100, seed=i)
        perm_test_returns = perm_test['close'].pct_change().dropna()
        volatilities.append(perm_test_returns.std())

    print(f"100次排列的波動率統計:")
    print(f"平均: {np.mean(volatilities):.6f}")
    print(f"標準差: {np.std(volatilities):.6f}")
    print(f"最小: {np.min(volatilities):.6f}")
    print(f"最大: {np.max(volatilities):.6f}")
    print(f"原始波動率: {original_returns.std():.6f}")

    print("\n所有測試完成!")
