"""
Data Loader Module
數據載入模組
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Callable

def load_local_data(
    data_source: str = "1h",
    symbol: str = "BTCUSDT",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    verbose: bool = True,
    lookback_days: int = 0
) -> Optional[pd.DataFrame]:
    """
    從本地載入數據文件，支持動態日期範圍調整

    Args:
        data_source (str): 數據時間框架，如 "1h", "1d"
        symbol (str): 交易對符號，如 "BTCUSDT"
        start_date (Optional[str]): 起始日期 (YYYY-MM-DD)
        end_date (Optional[str]): 結束日期 (YYYY-MM-DD)
        verbose (bool): 是否打印詳細信息
        lookback_days (int): 額外回溯天數，用於計算技術指標

    Returns:
        Optional[pd.DataFrame]: 載入的數據，如果失敗則返回 None

    Behavior:
        - 優先加載請求日期範圍內的數據
        - 如果數據範圍與請求範圍部分重疊，返回重疊部分
        - 如果完全沒有重疊，返回 None
        - lookback_days 會自動調整 effective_start_date 以包含足夠歷史數據
    """
    data_dir = Path(__file__).parents[1] / "data"
    if not data_dir.exists():
        if verbose:
            print(f"數據目錄不存在: {data_dir}")
        return None

    pattern = f"{symbol}_{data_source}_*.csv"
    csv_files = list(data_dir.glob(pattern))
    if not csv_files:
        if verbose:
            print(f"未找到本地數據文件 (搜索: {pattern}) in {data_dir}")
        return None

    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    if verbose:
        print(f"找到本地數據: {latest_file.name}")

    try:
        df = pd.read_csv(latest_file, index_col='timestamp', parse_dates=True)

        effective_start_date = start_date
        if start_date and lookback_days > 0:
            start_date_dt = pd.to_datetime(start_date)
            # 加載比回溯期多一點的數據以應對非交易日，對於加密貨幣可能不是必須但無害
            adjusted_start_dt = start_date_dt - pd.Timedelta(days=lookback_days + 5)
            effective_start_date = adjusted_start_dt.strftime('%Y-%m-%d')

        # 保留原始數據範圍用於檢查重疊
        original_start = df.index.min()
        original_end = df.index.max()

        if effective_start_date:
            df = df[df.index >= effective_start_date]
        if end_date:
            df = df[df.index <= end_date]

        if df.empty:
            # 檢查是否有任何時間重疊
            if effective_start_date and end_date:
                req_start = pd.to_datetime(effective_start_date)
                req_end = pd.to_datetime(end_date)

                # 如果數據範圍與請求範圍完全不重疊，才返回 None
                if original_end < req_start or original_start > req_end:
                    if verbose:
                        print(f"警告: 數據範圍({original_start.date()} 至 {original_end.date()})與請求範圍({req_start.date()} 至 {req_end.date()})完全不重疊")
                    return None

                # 有重疊：返回重疊部分（交集）的數據
                if verbose:
                    print(f"提示: {symbol} 數據範圍({original_start.date()} 至 {original_end.date()})與請求範圍({req_start.date()} 至 {req_end.date()})部分重疊")

                # 計算實際可用範圍（交集）
                actual_start = max(original_start, req_start)
                actual_end = min(original_end, req_end)

                # 重新讀取並篩選到交集範圍
                df = pd.read_csv(latest_file, index_col='timestamp', parse_dates=True)
                df = df[(df.index >= actual_start) & (df.index <= actual_end)]

                if df.empty:
                    if verbose:
                        print(f"警告: 交集範圍內無有效數據")
                    return None

                return df
            else:
                if verbose:
                    print(f"警告: 篩選後數據為空 (日期範圍: {start_date} 至 {end_date})")
                return None
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
    fetcher_func: Optional[Callable] = None,
    fallback_to_fetch: bool = True, # 新增參數
    verbose: bool = True
) -> pd.DataFrame:
    """
    優先從本地載入數據, 如果失敗可選擇是否使用fetcher函數下載
    """
    df = load_local_data(
        data_source=data_source,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        verbose=verbose
    )

    if df is None:
        if fallback_to_fetch and fetcher_func is not None:
            if verbose:
                print("\n本地數據不可用, 開始下載新數據...")
            df = fetcher_func(start_date=start_date, end_date=end_date, save=True, verbose=verbose)
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
        else:
            # 如果不允許 fallback 或沒有 fetcher，則直接報錯
            raise FileNotFoundError(
                f"在 'data/' 目錄下找不到 {symbol}_{data_source} 的本地數據，且未啟用自動下載。"
                f"請先運行對應的 data_fetcher 腳本下載數據。"
            )

    if df is None:
        raise ValueError("數據加載最終失敗。")

    return df