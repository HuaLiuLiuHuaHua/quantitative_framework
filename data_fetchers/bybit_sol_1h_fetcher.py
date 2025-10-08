"""
Bybit SOL 1H Data Fetcher
Bybit SOL 1小時K線數據抓取器

整合設計：
- 基於 BTC fetcher 架構
- 支援 SOL/USDT 線性合約數據
- 自動保存到 data/ 資料夾
"""

import urllib.request
import urllib.parse
import urllib.error
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from typing import Optional, Union
import socket

# Support both direct execution and module import
try:
    from .data_validator import validate_ohlc_data
except ImportError:
    from data_validator import validate_ohlc_data


def _to_millis(dt: Union[str, datetime]) -> int:
    """將日期轉換為毫秒時間戳"""
    if isinstance(dt, str):
        try:
            if len(dt.strip()) == 10:
                dt_obj = datetime.strptime(dt, "%Y-%m-%d")
            else:
                dt_obj = datetime.fromisoformat(dt)
        except Exception:
            dt_obj = pd.to_datetime(dt).to_pydatetime()
    elif isinstance(dt, datetime):
        dt_obj = dt
    else:
        raise ValueError("start_date / end_date must be str or datetime")

    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=timezone.utc)

    return int(dt_obj.timestamp() * 1000)


def fetch_bybit_sol_1h_data(
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    days: int = 365,
    limit: int = 1000,
    sleep_per_request: float = 0.15,
    chunk_days: int = 365,  # 1H使用365天分塊：每塊約8760條小時K線數據(365*24)
    save: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    抓取Bybit SOL/USDT 1小時K線數據

    Args:
        start_date: 開始日期 (str "YYYY-MM-DD" 或 datetime對象)
        end_date: 結束日期 (str "YYYY-MM-DD" 或 datetime對象)
        days: 如果未指定日期範圍，抓取最近N天的數據
        limit: 每次請求的最大記錄數 (1-1000)
        sleep_per_request: 每次請求後的休眠時間(秒)
        chunk_days: 大範圍數據的分塊大小(天)
        save: 是否自動保存到data/資料夾
        verbose: 是否打印詳細信息

    Returns:
        pd.DataFrame: OHLCV數據，包含以下列：
            - timestamp: 時間索引
            - open: 開盤價
            - high: 最高價
            - low: 最低價
            - close: 收盤價
            - volume: 成交量

    Example:
        >>> # 抓取2020-2025年的SOL 1小時數據
        >>> df = fetch_bybit_sol_1h_data(
        ...     start_date="2020-01-01",
        ...     end_date="2025-01-01",
        ...     verbose=True
        ... )
    """
    base_url = "https://api.bybit.com/v5/market/kline"
    category = "linear"
    symbol = "SOLUSDT"  # SOL/USDT 線性合約
    interval = "60"  # 60分鐘 = 1小時

    # 計算時間範圍
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    if start_date is None and end_date is None:
        end_ms = now_ms
        start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    else:
        if start_date is not None:
            start_ms = _to_millis(start_date)
            if end_date is not None:
                end_ms = _to_millis(end_date)
            else:
                # FIX: When end_date is None, use current time
                end_ms = now_ms
        elif end_date is not None:
            end_ms = _to_millis(end_date)
            start_ms = int((datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    # Validate that start_date < end_date
    if start_ms >= end_ms:
        raise ValueError(f"start_date must be before end_date. Got start={pd.to_datetime(start_ms, unit='ms')}, end={pd.to_datetime(end_ms, unit='ms')}")

    # 調整limit範圍
    if limit <= 0 or limit > 1000:
        limit = 1000

    # 判斷是否需要分塊抓取
    total_days = (end_ms - start_ms) // (24 * 60 * 60 * 1000)
    if total_days > chunk_days:
        if verbose:
            print(f"[INFO] 大範圍數據 ({total_days} 天), 使用分塊抓取...")
        all_chunks = []
        current_start = start_ms

        while current_start < end_ms:
            chunk_end = min(current_start + (chunk_days * 24 * 60 * 60 * 1000), end_ms)
            if verbose:
                print(f"[INFO] 抓取區塊: {pd.to_datetime(current_start, unit='ms').date()} 至 {pd.to_datetime(chunk_end, unit='ms').date()}")

            chunk_df = _fetch_single_range(base_url, category, symbol, interval, current_start, chunk_end, limit, sleep_per_request, verbose)
            if not chunk_df.empty:
                all_chunks.append(chunk_df)

            # 使用 chunk_end 作為下一個起點，依賴後續的去重邏輯處理重複數據
            current_start = chunk_end

        if all_chunks:
            df = pd.concat(all_chunks, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        else:
            df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    else:
        df = _fetch_single_range(base_url, category, symbol, interval, start_ms, end_ms, limit, sleep_per_request, verbose)

    if df.empty:
        raise Exception("無法抓取數據，請檢查日期範圍和網絡連接")

    # 驗證數據（Medium邏輯）
    df = validate_ohlc_data(df, verbose=verbose)

    # 自動保存到data/資料夾
    if save:
        start_str = pd.to_datetime(start_ms, unit='ms').strftime('%Y-%m-%d')
        end_str = pd.to_datetime(end_ms, unit='ms').strftime('%Y-%m-%d')
        filename = f"SOLUSDT_1h_{start_str}_{end_str}.csv"

        # data資料夾位於項目根目錄
        data_dir = Path(__file__).parents[1] / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        filepath = data_dir / filename

        # 保存前格式化 timestamp 為 ISO 8601 格式，保留時區信息
        df_to_save = df.copy()
        df_to_save['timestamp'] = df_to_save['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
        df_to_save.to_csv(filepath, index=False)
        if verbose:
            print(f"\n[SAVE] 數據已保存到: {filepath}")

    return df


def _fetch_single_range(
    base_url: str,
    category: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    sleep_per_request: float,
    verbose: bool = False
) -> pd.DataFrame:
    """抓取單一時間範圍的數據（內部函數）"""
    MAX_RETRIES = 3
    all_klines = []
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "limit": str(limit),
        "start": str(start_ms),
        "end": str(end_ms),
    }

    try:
        retry_count = 0
        while True:
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Mozilla/5.0 (compatible; quantitative-framework/1.0)")

            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw = resp.read()
                retry_count = 0  # Reset retry count on successful request
            except socket.timeout:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    raise Exception(f"請求超時，已達到最大重試次數 ({MAX_RETRIES})")
                if verbose:
                    print(f"[WARNING] 請求超時，重試中... ({retry_count}/{MAX_RETRIES})")
                time.sleep(sleep_per_request * 2)
                continue
            except Exception as ex:
                if verbose:
                    print(f"[ERROR] 請求失敗: {ex}")
                raise

            try:
                payload = json.loads(raw.decode())
            except Exception as ex:
                if verbose:
                    print(f"[ERROR] JSON解析失敗: {ex}")
                raise

            # 檢查API錯誤
            if payload.get("retCode") is not None and payload.get("retCode") != 0:
                raise Exception(f"API錯誤: {payload.get('retMsg', payload)}")

            # 提取K線數據
            klines = None
            if isinstance(payload.get("result"), dict) and "list" in payload["result"]:
                klines = payload["result"]["list"]
            elif "data" in payload and isinstance(payload["data"], list):
                klines = payload["data"]
            else:
                klines = payload.get("result") or payload.get("data") or []

            if not klines:
                break

            # 解析K線數據並收集所有時間戳
            batch_timestamps = []
            for k in klines:
                try:
                    ts_ms = int(k[0])
                    open_p = float(k[1])
                    high_p = float(k[2])
                    low_p = float(k[3])
                    close_p = float(k[4])
                    vol_p = float(k[5])
                except Exception:
                    if isinstance(k, dict):
                        # Add explicit None checks before converting
                        ts_raw = k.get("start") or k.get("t") or k.get("open_time") or k.get("timestamp")
                        if ts_raw is None:
                            if verbose:
                                print(f"[WARNING] 跳過無效記錄: 缺少時間戳 - {k}")
                            continue

                        open_raw = k.get("open")
                        high_raw = k.get("high")
                        low_raw = k.get("low")
                        close_raw = k.get("close")
                        vol_raw = k.get("volume") or k.get("vol") or k.get("trade_count")

                        if any(v is None for v in [open_raw, high_raw, low_raw, close_raw]):
                            if verbose:
                                print(f"[WARNING] 跳過無效記錄: 缺少價格數據 - {k}")
                            continue

                        try:
                            ts_ms = int(ts_raw)
                            open_p = float(open_raw)
                            high_p = float(high_raw)
                            low_p = float(low_raw)
                            close_p = float(close_raw)
                            vol_p = float(vol_raw) if vol_raw is not None else 0.0
                        except (ValueError, TypeError) as e:
                            if verbose:
                                print(f"[WARNING] 跳過無效記錄: 數據轉換失敗 - {k}, 錯誤: {e}")
                            continue
                    else:
                        raise

                batch_timestamps.append(ts_ms)

                # 只添加時間範圍內的數據
                if start_ms <= ts_ms <= end_ms:
                    all_klines.append({
                        "timestamp": pd.to_datetime(ts_ms, unit="ms"),
                        "open": open_p,
                        "high": high_p,
                        "low": low_p,
                        "close": close_p,
                        "volume": vol_p,
                    })

            # 檢查是否需要繼續分頁
            if len(klines) < int(params["limit"]):
                # Received fewer records than requested - no more data available
                break

            if not batch_timestamps:
                # No valid timestamps parsed - cannot continue
                break

            # Determine API return order and calculate next boundary
            # Bybit v5 returns data in DESCENDING order (newest first)
            min_ts = min(batch_timestamps)
            max_ts = max(batch_timestamps)

            # Check if we've covered the entire requested range
            if min_ts <= start_ms:
                # We've reached or passed the start boundary
                break

            # Update end parameter to fetch older data (before the oldest timestamp in this batch)
            # For descending order: we need to move the 'end' boundary backwards
            next_end = min_ts - 1

            if next_end <= start_ms:
                # No more data in range
                break

            # Safety check: prevent infinite loop
            if str(next_end) == params.get("end"):
                if verbose:
                    print(f"[WARNING] 分頁無進展，停止抓取")
                break

            # Update parameters for next request
            params["end"] = str(next_end)

            if verbose:
                print(f"[INFO] 已抓取 {len(all_klines)} 條記錄，繼續抓取更早數據...")

            time.sleep(sleep_per_request)

        if all_klines:
            return pd.DataFrame(all_klines).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        else:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    except (socket.timeout, urllib.error.URLError) as e:
        # 網絡錯誤：返回空 DataFrame
        if verbose:
            print(f"[ERROR] 網絡請求失敗: {e}")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        # 數據解析錯誤：返回空 DataFrame
        if verbose:
            print(f"[ERROR] 數據解析失敗: {e}")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    except Exception as e:
        # 未預期的嚴重錯誤：重新拋出
        if verbose:
            print(f"[ERROR] 發生未預期錯誤: {e}")
        raise


if __name__ == "__main__":
    # 默認行為：抓取 2020-01-01 至今的完整數據
    try:
        df = fetch_bybit_sol_1h_data(
            start_date="2020-01-01",
            end_date=None,
            save=True,
            verbose=True
        )

        print(f"\n完成！共抓取 {len(df)} 條數據")
        print(f"時間範圍: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
    except Exception as e:
        print(f"\n[ERROR] 執行失敗: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
