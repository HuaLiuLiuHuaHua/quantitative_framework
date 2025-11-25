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

try:
    from .data_validator import validate_ohlc_data
except ImportError:
    from data_validator import validate_ohlc_data

def _to_millis(dt: Union[str, datetime]) -> int:
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

def fetch_bybit_dot_1d_data(
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    days: int = 365,
    limit: int = 1000,
    sleep_per_request: float = 0.15,
    chunk_days: int = 1000,
    save: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    base_url = "https://api.bybit.com/v5/market/kline"
    category = "linear"
    symbol = "DOTUSDT"
    interval = "D"

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
                end_ms = now_ms
        elif end_date is not None:
            end_ms = _to_millis(end_date)
            start_ms = int((datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    if start_ms >= end_ms:
        raise ValueError(f"start_date must be before end_date. Got start={pd.to_datetime(start_ms, unit='ms')}, end={pd.to_datetime(end_ms, unit='ms')}")

    if limit <= 0 or limit > 1000:
        limit = 1000

    total_days = (end_ms - start_ms) // (24 * 60 * 60 * 1000)
    if total_days > chunk_days:
        if verbose:
            print(f"[INFO] Large data range ({total_days} days), using chunking...")
        all_chunks = []
        current_start = start_ms

        while current_start < end_ms:
            chunk_end = min(current_start + (chunk_days * 24 * 60 * 60 * 1000), end_ms)
            if verbose:
                print(f"[INFO] Fetching chunk: {pd.to_datetime(current_start, unit='ms').date()} to {pd.to_datetime(chunk_end, unit='ms').date()}")

            chunk_df = _fetch_single_range(base_url, category, symbol, interval, current_start, chunk_end, limit, sleep_per_request, verbose)
            if not chunk_df.empty:
                all_chunks.append(chunk_df)

            current_start = chunk_end

        if all_chunks:
            df = pd.concat(all_chunks, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        else:
            df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    else:
        df = _fetch_single_range(base_url, category, symbol, interval, start_ms, end_ms, limit, sleep_per_request, verbose)

    if df.empty:
        raise Exception("Failed to fetch data, please check date range and network connection")

    df = validate_ohlc_data(df, verbose=verbose)

    if save:
        start_str = pd.to_datetime(start_ms, unit='ms').strftime('%Y-%m-%d')
        end_str = pd.to_datetime(end_ms, unit='ms').strftime('%Y-%m-%d')
        filename = f"DOTUSDT_1d_{start_str}_{end_str}.csv"

        data_dir = Path(__file__).parents[1] / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        filepath = data_dir / filename

        df_to_save = df.copy()
        df_to_save['timestamp'] = df_to_save['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
        df_to_save.to_csv(filepath, index=False)
        if verbose:
            print(f"\n[SAVE] Data saved to: {filepath}")

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
                retry_count = 0
            except socket.timeout:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    raise Exception(f"Request timed out after {MAX_RETRIES} retries")
                if verbose:
                    print(f"[WARNING] Request timed out, retrying... ({retry_count}/{MAX_RETRIES})")
                time.sleep(sleep_per_request * 2)
                continue
            except Exception as ex:
                if verbose:
                    print(f"[ERROR] Request failed: {ex}")
                raise

            try:
                payload = json.loads(raw.decode())
            except Exception as ex:
                if verbose:
                    print(f"[ERROR] JSON parsing failed: {ex}")
                raise

            if payload.get("retCode") is not None and payload.get("retCode") != 0:
                raise Exception(f"API Error: {payload.get('retMsg', payload)}")

            klines = payload.get("result", {}).get("list", [])

            if not klines:
                break

            batch_timestamps = []
            for k in klines:
                try:
                    ts_ms = int(k[0])
                    open_p = float(k[1])
                    high_p = float(k[2])
                    low_p = float(k[3])
                    close_p = float(k[4])
                    vol_p = float(k[5])
                    batch_timestamps.append(ts_ms)
                    if start_ms <= ts_ms <= end_ms:
                        all_klines.append({
                            "timestamp": pd.to_datetime(ts_ms, unit="ms"),
                            "open": open_p,
                            "high": high_p,
                            "low": low_p,
                            "close": close_p,
                            "volume": vol_p,
                        })
                except (ValueError, TypeError) as e:
                    if verbose:
                        print(f"[WARNING] Skipping invalid record: {k}, Error: {e}")
                    continue

            if len(klines) < int(params["limit"]):
                break

            if not batch_timestamps:
                break

            min_ts = min(batch_timestamps)
            if min_ts <= start_ms:
                break

            next_end = min_ts - 1
            if str(next_end) == params.get("end"):
                if verbose:
                    print("[WARNING] Pagination not progressing, stopping fetch.")
                break

            params["end"] = str(next_end)

            if verbose:
                print(f"[INFO] Fetched {len(all_klines)} records, continuing to fetch older data...")

            time.sleep(sleep_per_request)

        if all_klines:
            return pd.DataFrame(all_klines).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        else:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    except Exception as e:
        if verbose:
            print(f"[ERROR] An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    try:
        df = fetch_bybit_dot_1d_data(
            start_date="2020-01-01",
            end_date=None,
            save=True,
            verbose=True
        )
        print(f"\nSuccess! Fetched {len(df)} records.")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    except Exception as e:
        print(f"\n[ERROR] Execution failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
