"""
Data Loader Module
數據載入模組

提供統一的數據載入接口,優先從本地讀取已下載的數據,
加快測試速度,避免重複下載。

功能:
1. 優先從本地 data/ 目錄載入 .csv 文件
2. 支持1h和1d兩種時間週期數據
3. 自動篩選日期範圍
4. 找不到數據時返回None,由調用方決定是否下載

使用示例:
    from shared.data_loader import load_local_data

    # 載入1小時數據
    df = load_local_data(
        data_source="1h",
        start_date="2024-01-01",
        end_date="2024-12-31"
    )

    # 載入1日數據
    df = load_local_data(
        data_source="1d",
        start_date="2022-10-01",
        end_date="2024-09-30"
    )
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_local_data(
    data_source: str = "1h",
    symbol: str = "BTCUSDT",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    verbose: bool = True
) -> Optional[pd.DataFrame]:
    """
    從本地載入數據文件

    優先從 quantitative_framework/data/ 目錄載入已下載的 .csv 文件。
    如果找不到數據,返回None,由調用方決定是否下載新數據。

    Args:
        data_source: 數據來源 ("1h" 或 "1d")
        symbol: 交易對符號 ("BTCUSDT", "SOLUSDT", 等)
        start_date: 開始日期 (格式: "YYYY-MM-DD")
        end_date: 結束日期 (格式: "YYYY-MM-DD")
        verbose: 是否打印詳細信息

    Returns:
        pd.DataFrame: 載入的OHLCV數據,如果找不到則返回None

    Example:
        >>> df = load_local_data(data_source="1h", symbol="SOLUSDT", start_date="2024-01-01", end_date="2024-12-31")
        >>> if df is not None:
        ...     print(f"Data loaded: {len(df)} rows")
        ... else:
        ...     print("No data found, need to download")
    """
    # 驗證參數
    if data_source not in ["1h", "1d"]:
        raise ValueError(f"data_source必須是 '1h' 或 '1d',當前值: {data_source}")

    # 獲取data目錄路徑
    # 從 shared/ 目錄向上兩層到 quantitative_framework/,然後進入 data/
    data_dir = Path(__file__).parents[1] / "data"

    if verbose:
        print("=" * 70)
        print("數據載入 (優先使用本地數據)")
        print("=" * 70)

    # 檢查data目錄是否存在
    if not data_dir.exists():
        if verbose:
            print(f"數據目錄不存在: {data_dir}")
        return None

    # 根據symbol和data_source搜索對應的csv文件
    pattern = f"{symbol}_{data_source}_*.csv"
    csv_files = list(data_dir.glob(pattern))

    if not csv_files:
        if verbose:
            print(f"未找到本地數據文件 (搜索: {pattern})")
            print(f"搜索路徑: {data_dir}")
        return None

    # 使用最新的文件 (按修改時間排序)
    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)

    if verbose:
        print(f"找到本地數據: {latest_file.name}")
        print("載入數據...")

    try:
        # 讀取csv文件
        df = pd.read_csv(latest_file, parse_dates=['timestamp'])

        # 確保index是DatetimeIndex
        if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 篩選日期範圍 (如果指定)
        if start_date is not None or end_date is not None:
            original_len = len(df)

            if start_date is not None:
                df = df[df.index >= start_date]

            if end_date is not None:
                df = df[df.index <= end_date]

            if verbose:
                print(f"日期篩選: {original_len} -> {len(df)} 筆")

        # 檢查是否有數據
        if len(df) == 0:
            if verbose:
                print(f"警告: 篩選後數據為空 (日期範圍: {start_date} 至 {end_date})")
            return None

        if verbose:
            print(f"數據載入成功！")
            print(f"數據範圍: {df.index[0]} 至 {df.index[-1]}")
            print(f"數據筆數: {len(df)}")
            print(f"欄位: {df.columns.tolist()}")

        return df

    except Exception as e:
        if verbose:
            print(f"載入數據時發生錯誤: {e}")
        return None


def load_data_with_fallback(
    data_source: str = "1h",
    symbol: str = "BTCUSDT",
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    fetcher_func = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    優先從本地載入數據,如果失敗則使用fetcher函數下載

    這是一個便利函數,結合了本地載入和API下載的邏輯。

    Args:
        data_source: 數據來源 ("1h" 或 "1d")
        symbol: 交易對符號 ("BTCUSDT", "SOLUSDT", 等)
        start_date: 開始日期
        end_date: 結束日期
        fetcher_func: 數據獲取函數 (如果本地找不到數據則調用)
        verbose: 是否打印詳細信息

    Returns:
        pd.DataFrame: OHLCV數據

    Example:
        >>> from data_fetchers.bybit_sol_1h_fetcher import fetch_bybit_sol_1h_data
        >>> df = load_data_with_fallback(
        ...     data_source="1h",
        ...     symbol="SOLUSDT",
        ...     start_date="2024-01-01",
        ...     end_date="2024-12-31",
        ...     fetcher_func=fetch_bybit_sol_1h_data,
        ...     verbose=True
        ... )
    """
    # 步驟1: 嘗試從本地載入
    df = load_local_data(
        data_source=data_source,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        verbose=verbose
    )

    # 步驟2: 如果找不到數據且提供了fetcher函數,則下載
    if df is None and fetcher_func is not None:
        if verbose:
            print("\n本地數據不可用,開始下載新數據...")
            print("=" * 70)

        df = fetcher_func(
            start_date=start_date,
            end_date=end_date,
            save=True,
            verbose=verbose
        )

        # 確保index是DatetimeIndex
        if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

    # 步驟3: 如果仍然沒有數據,拋出錯誤
    if df is None:
        raise ValueError(
            f"無法載入數據: 本地找不到 {data_source} 數據,且未提供fetcher_func"
        )

    return df


if __name__ == "__main__":
    # 測試數據載入功能
    print("=" * 70)
    print("數據載入模組測試")
    print("=" * 70)

    # 測試1: 載入1d數據
    print("\n測試1: 載入1日數據")
    df_1d = load_local_data(
        data_source="1d",
        start_date="2022-10-01",
        end_date="2024-09-30",
        verbose=True
    )

    if df_1d is not None:
        print(f"\n✓ 1日數據載入成功")
        print(f"  形狀: {df_1d.shape}")
        print(f"  前3筆:\n{df_1d.head(3)}")
    else:
        print("\n✗ 1日數據載入失敗")

    # 測試2: 載入1h數據
    print("\n" + "=" * 70)
    print("測試2: 載入1小時數據")
    df_1h = load_local_data(
        data_source="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        verbose=True
    )

    if df_1h is not None:
        print(f"\n✓ 1小時數據載入成功")
        print(f"  形狀: {df_1h.shape}")
        print(f"  前3筆:\n{df_1h.head(3)}")
    else:
        print("\n✗ 1小時數據載入失敗 (可能尚未下載)")

    print("\n" + "=" * 70)
    print("測試完成!")
    print("=" * 70)
