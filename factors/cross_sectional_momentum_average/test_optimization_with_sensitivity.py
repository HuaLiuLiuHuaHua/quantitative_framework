"""
均線交叉因子市場中性策略 - 參數優化與敏感性分析

功能:
1. 自適應搜索策略(網格搜索 vs 隨機搜索+補充鄰域)
2. 敏感性分析(夏普>1.25 + 穩健性調整後分數)
3. 生成完整報告和圖表(包含因子vs BTC權益+回撤對比)

作者: Claude Code
創建日期: 2025-10-16
"""

import sys
import json
import os
from venv import logger
import numpy as np
import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from factors.cross_sectional_momentum_average.factor import MovingAverageCrossFactor
from shared.cross_sectional_backtest import CrossSectionalBacktester
from shared.data_loader import load_local_data
from shared.visualization import plot_walkforward_performance
from shared.sensitivity import (
    get_parameter_neighborhood,
    calculate_custom_robustness_score,
    filter_robust_params
)


# ==================== 輔助函數 ====================

def percentile_to_n(universe_size: int, percentile: int) -> int:
    """將百分位數轉換為標的數量"""
    return round(universe_size * percentile / 100)


def calculate_stop_loss(leverage: int) -> float:
    """根據槓桿倍數計算止損點位"""
    return -1.0 / leverage


def evaluate_single_params(params: dict, **kwargs) -> dict:
    """評估單個參數組合的輔助函數(用於並行計算)"""
    # 禁用子進程的進度條和警告
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["PYTHONWARNINGS"] = "ignore"

    try:
        # 解析參數
        short_window = int(params["short_window"])
        long_window = int(params["long_window"])
        long_short_percentile = int(params["long_short_percentile"])
        leverage = int(params["leverage"])
        rebalance_freq_days = int(params["rebalance_freq_days"])

        # 無效組合: 短線週期大於等於長線週期
        if short_window >= long_window:
            return {
                **params, "sharpe_ratio": -99, "turnover": 0, "subniverse_sharpe": -99,
                "avg_monthly_return": 0.0, "std_monthly_return": 0.0, "top_n": 0
            }

        # 計算止損
        stop_loss = calculate_stop_loss(leverage)

        # 從kwargs獲取配置
        start_date = kwargs["start_date"]
        end_date = kwargs["end_date"]
        universe_csv_path = kwargs["universe_csv_path"]
        universe_top_n = kwargs["universe_top_n"]
        commission_bps = kwargs["commission_bps"]
        slippage_bps = kwargs["slippage_bps"]
        price_data_cache = kwargs.get("price_data_cache", {})

        # 讀取宇宙CSV
        universe_df = pd.read_csv(universe_csv_path)
        universe_df["timestamp"] = pd.to_datetime(universe_df["timestamp"])

        # 計算平均宇宙大小和標的數量
        avg_universe_size = len([s for s in universe_df.iloc[0, 1:] if pd.notna(s) and s != ""])
        top_n = percentile_to_n(avg_universe_size, long_short_percentile)

        # 初始化均線交叉因子
        ma_cross_factor = MovingAverageCrossFactor()

        # 創建回測器
        backtester = CrossSectionalBacktester(
            factor_calculator=ma_cross_factor,
            start_date=start_date,
            end_date=end_date,
            timeframe="1d",
            top_n=top_n,
            bottom_n=top_n,
            position_weight=None,  # 等權重
            rebalance_freq=f"{rebalance_freq_days}D",
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            universe_csv_path=universe_csv_path,
            universe_top_n=universe_top_n
        )

        # 使用預加載的數據
        backtester.all_data = price_data_cache

        # 計算因子
        factor_df = ma_cross_factor.calculate(price_data_cache, short_window=short_window, long_window=long_window, min_volume=0)

        # 運行回測
        results = backtester.run(factor_df=factor_df, stop_loss=stop_loss)

        # 提取績效指標
        perf_summary = results.get('performance_summary', {})
        sharpe_ratio = perf_summary.get('sharpe_ratio', -99)
        turnover = perf_summary.get('turnover', 0)
        subniverse_sharpe = perf_summary.get('sub_universe_sharpe', -99)
        daily_returns = perf_summary.get('daily_return', pd.Series(dtype=float))
        if not daily_returns.empty:
            monthly_returns = (1 + daily_returns).resample('M').prod() - 1
            avg_monthly_return = monthly_returns.mean()
            std_monthly_return = monthly_returns.std()
        else:
            avg_monthly_return = 0
            std_monthly_return = 0
        avg_monthly_return = 0 if pd.isna(avg_monthly_return) else avg_monthly_return
        std_monthly_return = 0 if pd.isna(std_monthly_return) else std_monthly_return

        return {
            **params,
            "sharpe_ratio": float(sharpe_ratio),
            "turnover": float(turnover),
            "subniverse_sharpe": float(subniverse_sharpe),
            "avg_monthly_return": float(avg_monthly_return),
            "std_monthly_return": float(std_monthly_return),
            "top_n": top_n
        }

    except Exception as e:
        import traceback
        import sys
        print(f"參數評估失敗 {params}: {e}", file=sys.stderr)
        print("完整錯誤堆棧:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

        # Also save to file for debugging
        with open("error_log.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n=== 參數評估失敗 ===\n")
            f.write(f"參數: {params}\n")
            f.write(f"錯誤: {e}\n")
            traceback.print_exc(file=f)

        return {
            **params,
            "sharpe_ratio": -99,
            "turnover": 0,
            "subniverse_sharpe": -99,
            "avg_monthly_return": 0.0,
            "std_monthly_return": 0.0,
            "top_n": 0
        }


def generate_all_combinations(param_grid: dict) -> list:
    """生成所有參數組合"""
    import itertools
    keys = param_grid.keys()
    values = [list(v) for v in param_grid.values()]
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return combinations


def preload_price_data(universe_csv_path: str, project_root: Path, start_date: str, end_date: str, max_lookback: int) -> dict:
    """預加載所有需要的價格數據到緩存中"""
    print("正在預加載所有價格數據...")
    universe_df = pd.read_csv(universe_csv_path)
    universe_df["timestamp"] = pd.to_datetime(universe_df["timestamp"], dayfirst=True)

    # 只收集測試期間（start_date到end_date之間）存在的標的
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    filtered_universe = universe_df[
        (universe_df["timestamp"] >= start_date_dt) &
        (universe_df["timestamp"] <= end_date_dt)
    ]

    all_symbols = set()
    for _, row in filtered_universe.iterrows():
        all_symbols.update([s for s in row[1:] if pd.notna(s) and s != ""])

    logger.info(f"從宇宙文件中篩選出 {len(filtered_universe)} 個時間點（{start_date} 至 {end_date}）")
    logger.info(f"共 {len(all_symbols)} 個唯一標的需要加載")
    print(f"DEBUG: Found {len(all_symbols)} unique symbols in filtered universe.")

    price_data_cache = {}
    for symbol in tqdm(all_symbols, desc="加載價格數據"):
        # Use load_local_data WITHOUT lookback_days preloading
        symbol_with_suffix = f"{symbol}USDT"
        df = load_local_data(
            symbol=symbol_with_suffix,
            data_source="1d",
            start_date=start_date,
            end_date=end_date,
            lookback_days=0  # 從start_date開始加載，不前移日期
        )
        if df is not None and not df.empty:
            # IMPORTANT: Store with USDT suffix to match backtester's _get_universe_for_date() return format
            # See cross_sectional_backtest.py line 101: [f"{ticker}USDT" for ticker in universe]
            price_data_cache[symbol_with_suffix] = df
        else:
            print(f"[警告] {symbol_with_suffix} 數據為空或加載失敗")

    print(f"\n成功加載 {len(price_data_cache)} 個標的的數據")
    print(f"DEBUG: Loaded symbols: {list(price_data_cache.keys())[:10]}...")
    return price_data_cache


def run_optimization(
    param_grid: dict,
    start_date: str,
    end_date: str,
    universe_csv_path: str,
    universe_top_n: int,
    commission_bps: float,
    slippage_bps: float,
    price_data_cache: dict,
    n_trials: int = 300,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    運行參數優化

    自適應策略:
    - 總組合數 < n_trials → 完整網格搜索
    - 總組合數 >= n_trials → 隨機搜索
    """
    # 計算總組合數
    total_combinations = np.prod([len(list(v)) for v in param_grid.values()])

    print(f"參數空間大小: {total_combinations} 個組合")
    print(f"隨機搜索次數: {n_trials}")

    # 決定搜索策略
    if total_combinations <= n_trials:
        print("→ 使用完整網格搜索")
        param_combinations = generate_all_combinations(param_grid)
    else:
        print("→ 使用隨機搜索")
        param_combinations = []
        for _ in range(n_trials):
            params = {}
            for name, values in param_grid.items():
                params[name] = random.choice(list(values))
            param_combinations.append(params)

    print(f"實際評估組合數: {len(param_combinations)}\n")

    # 構建kwargs
    kwargs = {
        "start_date": start_date,
        "end_date": end_date,
        "universe_csv_path": universe_csv_path,
        "universe_top_n": universe_top_n,
        "commission_bps": commission_bps,
        "slippage_bps": slippage_bps,
        "price_data_cache": price_data_cache
    }

    # 並行評估參數
    num_workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    worker = partial(evaluate_single_params, **kwargs)

    results = []
    print("開始參數優化...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, params) for params in param_combinations]

        for future in tqdm(futures, desc="參數優化進度", ncols=100):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"參數評估失敗: {e}")

    results_df = pd.DataFrame(results)
    return results_df


def supplement_neighborhood_for_best(
    best_params: dict,
    param_grid: dict,
    results_df: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    補充最優參數的鄰域點

    用於隨機搜索模式下,確保最優參數的鄰域都被評估過
    """
    print("\n補充最優參數的鄰域點...")

    # 獲取所有鄰域參數
    neighborhood_params = []
    for param_name, param_value in best_params.items():
        if param_name not in param_grid:
            continue

        neighbors = get_parameter_neighborhood(
            param_name=param_name,
            param_value=param_value,
            param_grid=param_grid,
            step_size=2
        )

        for neighbor_val in neighbors:
            if neighbor_val == param_value:
                continue  # 跳過中心點

            # 創建鄰域參數組合
            neighbor_params = best_params.copy()
            neighbor_params[param_name] = neighbor_val

            # 檢查是否已經評估過
            exists = False
            for _, row in results_df.iterrows():
                match = True
                for pname, pval in neighbor_params.items():
                    if pname in results_df.columns and row[pname] != pval:
                        match = False
                        break
                if match:
                    exists = True
                    break

            if not exists:
                neighborhood_params.append(neighbor_params)

    if not neighborhood_params:
        print("所有鄰域點已存在,無需補充")
        return results_df

    print(f"需要補充 {len(neighborhood_params)} 個鄰域點")

    # 評估鄰域點
    worker = partial(evaluate_single_params, **kwargs)
    new_results = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(worker, params) for params in neighborhood_params]

        for future in tqdm(futures, desc="補充鄰域點", ncols=100):
            try:
                new_results.append(future.result())
            except Exception as e:
                print(f"鄰域點評估失敗: {e}")

    # 合併結果
    new_results_df = pd.DataFrame(new_results)
    results_df = pd.concat([results_df, new_results_df], ignore_index=True)

    print(f"補充完成,總計 {len(results_df)} 個參數組合\n")
    return results_df


# ==================== 主函數 ====================

def main():
    """主函數"""
    print("=" * 80)
    print("均線交叉因子市場中性策略 - 參數優化與敏感性分析")
    print("=" * 80)
    print()

    # ========== 用戶配置區 ==========

    # 優化期間
    START_DATE = "2022-11-1"
    END_DATE = "2024-12-31"

    # 參數網格(全部使用range())
    PARAM_GRID = {
        "short_window": range(1, 450, 45),
        "long_window": range(1, 450, 45),
        "long_short_percentile": range(10, 20, 10),
        "leverage": range(1, 2, 1),
        "rebalance_freq_days": range(1, 31, 3)
    }

    # 隨機搜索次數 (因為參數組合數較小,會自動轉為網格搜索)
    N_TRIALS = 1000

    # 敏感性分析配置
    SHARPE_THRESHOLD = 1.25  # 夏普比率閾值

    # 交易成本
    COMMISSION_BPS = 20  # 0.2%
    SLIPPAGE_BPS = 5     # 0.05%

    # 動態宇宙配置
    UNIVERSE_CSV_PATH = str(project_root / "data" / "40.csv")
    UNIVERSE_TOP_N = 40

    # Debug mode
    DEBUG_MODE = False  # 設置為True以啟用詳細日誌輸出

    # 並行計算設置
    N_JOBS = -1  # -1表示使用所有CPU核心

    # ========== 配置區結束 ==========

    print(f"優化期間: {START_DATE} 至 {END_DATE}")
    print(f"參數空間:")
    for param_name, param_range in PARAM_GRID.items():
        print(f"  {param_name}: {list(param_range)[:3]}...{list(param_range)[-3:]}")
    print(f"夏普比率閾值: {SHARPE_THRESHOLD}")
    print(f"隨機搜索次數: {N_TRIALS}")
    print()

    # Validate lookback period vs available data
    max_lookback = max(PARAM_GRID["long_window"]) if "long_window" in PARAM_GRID and PARAM_GRID["long_window"] else 0
    available_days = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days

    if max_lookback > available_days:
        print(f"警告: 最大lookback期間 ({max_lookback}天) 超過測試期間 ({available_days}天)")
        print(f"       前{max_lookback}天將作為warmup期（無交易），實際交易期為{available_days - max_lookback}天")
        print(f"       建議: 增加測試期間或減小lookback參數範圍\n")

    # Step 0: 預加載所有價格數據
    price_data_cache = preload_price_data(UNIVERSE_CSV_PATH, project_root, START_DATE, END_DATE, max_lookback)

    # Validate cache keys format
    if price_data_cache:
        sample_keys = list(price_data_cache.keys())[:3]
        print(f"DEBUG: Sample cache keys: {sample_keys}")
        all_have_usdt = all(key.endswith("USDT") for key in price_data_cache.keys())
        if not all_have_usdt:
            print("[ERROR] Cache validation failed: Not all keys have USDT suffix!")
            return
        else:
            print(f"✓ Cache validation passed: All {len(price_data_cache)} keys have correct format\n")

    # Step 1: 運行參數優化
    print("【Step 1】運行參數優化")
    results_df = run_optimization(
        param_grid=PARAM_GRID,
        start_date=START_DATE,
        end_date=END_DATE,
        universe_csv_path=UNIVERSE_CSV_PATH,
        universe_top_n=UNIVERSE_TOP_N,
        commission_bps=COMMISSION_BPS,
        slippage_bps=SLIPPAGE_BPS,
        price_data_cache=price_data_cache,
        n_trials=N_TRIALS,
        n_jobs=N_JOBS
    )

    print(f"優化完成,共評估 {len(results_df)} 個參數組合")
    print(f"有效結果數量: {(results_df['sharpe_ratio'] > -99).sum()}\n")

    # Step 2: 找到初步最優參數
    print("【Step 2】找到初步最優參數")
    print("-" * 80)

    valid_results = results_df[results_df['sharpe_ratio'] > -99].copy()
    if valid_results.empty:
        print("錯誤: 沒有有效的優化結果!")
        return

    best_idx = valid_results['sharpe_ratio'].idxmax()
    best_result = valid_results.loc[best_idx]

    param_names = list(PARAM_GRID.keys())
    initial_best_params = {name: best_result[name] for name in param_names}

    print(f"初步最優參數: {initial_best_params}")
    print(f"初步最優夏普比率: {best_result['sharpe_ratio']:.4f}\n")

    # Step 3: 補充鄰域點(如果使用隨機搜索)
    total_combinations = np.prod([len(list(v)) for v in PARAM_GRID.values()])
    if total_combinations > N_TRIALS:
        print("【Step 3】補充最優參數的鄰域點")
        print("-" * 80)

        # 構建kwargs用於補充, 傳入已加載的數據
        kwargs = {
            "start_date": START_DATE,
            "end_date": END_DATE,
            "universe_csv_path": UNIVERSE_CSV_PATH,
            "universe_top_n": UNIVERSE_TOP_N,
            "commission_bps": COMMISSION_BPS,
            "slippage_bps": SLIPPAGE_BPS,
            "price_data_cache": price_data_cache
        }

        results_df = supplement_neighborhood_for_best(
            best_params=initial_best_params,
            param_grid=PARAM_GRID,
            results_df=results_df,
            **kwargs
        )
    else:
        print("【Step 3】跳過(使用完整網格搜索,無需補充)\n")

    # Step 4: 敏感性分析
    print("【Step 4】敏感性分析")
    print("-" * 80)

    robust_df, best_robust_params = filter_robust_params(
        results_df=results_df,
        param_grid=PARAM_GRID,
        metric="sharpe_ratio",
        sharpe_threshold=SHARPE_THRESHOLD,
        top_n=10
    )

    if robust_df.empty:
        print(f"警告: 沒有參數組合的夏普比率超過 {SHARPE_THRESHOLD}")
        print("使用初步最優參數作為最終參數")
        best_robust_params = initial_best_params
        final_sharpe = best_result['sharpe_ratio']
        robustness_score = 0.0
    else:
        print(f"找到 {len(robust_df)} 個穩健參數組合")
        print(f"\n最佳穩健參數: {best_robust_params}")
        print(f"穩健性調整後分數: {robust_df.iloc[0]['robustness_score']:.4f}")
        final_sharpe = robust_df.iloc[0]['center_sharpe']
        robustness_score = robust_df.iloc[0]['robustness_score']

    print()

    # Step 5: 使用最佳穩健參數運行最終回測
    print("【Step 5】使用最佳穩健參數運行最終回測")
    print("-" * 80)

    # 解析最佳參數
    short_window = int(best_robust_params["short_window"])
    long_window = int(best_robust_params["long_window"])
    long_short_percentile = int(best_robust_params["long_short_percentile"])
    leverage = int(best_robust_params["leverage"])
    rebalance_freq_days = int(best_robust_params["rebalance_freq_days"])
    stop_loss = calculate_stop_loss(leverage)

    universe_df = pd.read_csv(UNIVERSE_CSV_PATH)
    avg_universe_size = len([s for s in universe_df.iloc[0, 1:] if pd.notna(s) and s != ""])
    top_n = percentile_to_n(avg_universe_size, long_short_percentile)

    print(f"最佳參數: short_window={short_window}, long_window={long_window}, percentile={long_short_percentile}%, leverage={leverage}x, rebalance={rebalance_freq_days}D")
    print(f"做多/做空標的數量: {top_n}")
    print(f"止損點位: {stop_loss:.2%}\n")

    # 創建回測器
    ma_cross_factor = MovingAverageCrossFactor()
    backtester = CrossSectionalBacktester(
        factor_calculator=ma_cross_factor,
        start_date=START_DATE,
        end_date=END_DATE,
        timeframe="1d",
        top_n=top_n,
        bottom_n=top_n,
        position_weight=None,
        rebalance_freq=f"{rebalance_freq_days}D",
        commission_bps=COMMISSION_BPS,
        slippage_bps=SLIPPAGE_BPS,
        universe_csv_path=UNIVERSE_CSV_PATH,
        universe_top_n=UNIVERSE_TOP_N
    )

    # 直接使用預加載的數據
    backtester.all_data = price_data_cache

    # 計算因子
    print("\n正在計算因子值...")
    factor_df = ma_cross_factor.calculate(price_data_cache, short_window=short_window, long_window=long_window, min_volume=0)
    
    # 打印因子數據統計
    print(f"因子數據統計:")
    print(f"  總行數: {len(factor_df)}")
    print(f"  總列數: {len(factor_df.columns)}")
    print(f"  非空值數量: {factor_df.count().sum()}")
    print(f"  包含的幣種: {list(factor_df.columns)[:5]}...")
    
    # 檢查幾個時間點的因子值
    print("\n因子值樣本(前3行):")
    print(factor_df.head(3))

    # 運行最終回測
    print("\n開始運行回測...")
    final_results = backtester.run(factor_df=factor_df, stop_loss=stop_loss)
        
    # 檢查回測結果
    if not final_results:
        print("警告: 回測結果為空!")
    else:
        equity = final_results.get('equity_curve')
        if equity is None or equity.empty:
            print("警告: 權益曲線為空!")
        else:
            print("\n權益曲線統計:")
            print(f"  起始值: {equity.iloc[0]:.4f}")
            print(f"  結束值: {equity.iloc[-1]:.4f}")
            print(f"  收益率: {(equity.iloc[-1]/equity.iloc[0] - 1)*100:.2f}%")    
    factor_equity = final_results.get('equity_curve')
    perf_summary = final_results.get('performance_summary', {})

    daily_returns = perf_summary.get('daily_return', pd.Series(dtype=float))
    if not daily_returns.empty:
        monthly_returns = (1 + daily_returns).resample('M').prod() - 1
        avg_monthly_return = monthly_returns.mean()
        std_monthly_return = monthly_returns.std()
    else:
        avg_monthly_return = 0
        std_monthly_return = 0
    avg_monthly_return = 0 if pd.isna(avg_monthly_return) else avg_monthly_return
    std_monthly_return = 0 if pd.isna(std_monthly_return) else std_monthly_return

    print("最終回測績效指標:")
    print(f"  夏普比率: {perf_summary.get('sharpe_ratio', 0):.4f}")
    print(f"  平均月回報: {avg_monthly_return:.4%}")
    print(f"  每月回報標準差: {std_monthly_return:.4%}")
    print(f"  換手率: {perf_summary.get('turnover', 0):.2%}")
    print(f"  Sub-universe夏普: {perf_summary.get('sub_universe_sharpe', 0):.4f}")
    print()

    # Step 6: 加載BTC數據作為基準（從第一笔交易日期開始）
    print("【Step 6】加載BTC數據作為基準")
    print("-" * 80)

    # Get first trade date from performance summary
    first_trade_date_str = perf_summary.get('first_trade_date')

    if first_trade_date_str is None:
        print("警告: 沒有實際交易發生，跳過BTC基準對比\n")
        btc_equity = None
    else:
        # Load BTC data starting from first trade date (NOT START_DATE)
        btc_data = load_local_data(
            symbol="BTCUSDT",
            data_source="1d",
            start_date=first_trade_date_str,
            end_date=END_DATE,
            lookback_days=0
        )

        if btc_data is not None and not btc_data.empty:
            # Check if BTC has enough data
            if len(btc_data) < 2:
                print(f"警告: BTC數據不足 (僅{len(btc_data)}行)，跳過基準對比\n")
                btc_equity = None
            else:
                # Align BTC to factor equity starting from first trade date
                first_trade_date = pd.to_datetime(first_trade_date_str)
                factor_equity_aligned = factor_equity[factor_equity.index >= first_trade_date]
                factor_equity_aligned = factor_equity_aligned / factor_equity_aligned.iloc[0]

                # Find common dates between BTC and factor equity
                common_dates = factor_equity_aligned.index.intersection(btc_data.index)
                if len(common_dates) == 0:
                    print("警告: BTC和因子數據沒有重疊日期，跳過基準對比\n")
                    btc_equity = None
                else:
                    # Calculate BTC returns only on common dates
                    btc_returns = btc_data.loc[common_dates, 'close'].pct_change().fillna(0)
                    initial_capital = factor_equity_aligned.loc[common_dates[0]]
                    btc_equity = (1 + btc_returns).cumprod() * initial_capital

                    # Also align factor equity to common dates for fair comparison
                    factor_equity_aligned = factor_equity_aligned.loc[common_dates]

                    print(f"成功加載BTC數據 (對齊至首次交易日期: {first_trade_date.date()}, {len(common_dates)}個交易日)\n")
        else:
            print("警告: 無法加載BTC數據，跳過基準對比\n")
            btc_equity = None

    # Step 7: 生成報告和圖表
    print("【Step 7】生成報告和圖表")
    print("-" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "results" / f"Optimization_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存優化結果
    results_df.to_csv(output_dir / "optimization_results.csv", index=False)
    print(f"✓ 保存優化結果: optimization_results.csv")

    # 保存敏感性分析結果
    if not robust_df.empty:
        robust_df.to_csv(output_dir / "sensitivity_analysis.csv", index=False)
        print(f"✓ 保存敏感性分析: sensitivity_analysis.csv")

    # 保存最佳參數
    best_params_summary = {
        "best_robust_params": {k: int(v) if isinstance(v, (np.integer, np.int64)) else v
                              for k, v in best_robust_params.items()},
        "performance_metrics": {
            "sharpe_ratio": float(final_sharpe),
            "avg_monthly_return": float(avg_monthly_return),
            "std_monthly_return": float(std_monthly_return),
            "turnover": float(perf_summary.get('turnover', 0)),
            "subniverse_sharpe": float(perf_summary.get('sub_universe_sharpe', 0)),
            "robustness_score": float(robustness_score)
        },
        "configuration": {
            "top_n": int(top_n),
            "stop_loss": float(stop_loss),
            "start_date": START_DATE,
            "end_date": END_DATE,
            "sharpe_threshold": SHARPE_THRESHOLD
        }
    }

    with open(output_dir / "best_robust_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params_summary, f, indent=4, ensure_ascii=False)
    print(f"✓ 保存最佳參數: best_robust_params.json")

    # 繪製權益曲線和回撤對比圖（使用對齊的權益曲線）
    if factor_equity is not None and not factor_equity.empty:
        # Use aligned equity curves for fair comparison
        first_trade_date_str = perf_summary.get('first_trade_date')
        if first_trade_date_str is not None:
            first_trade_date = pd.to_datetime(first_trade_date_str)
            factor_equity_aligned = factor_equity[factor_equity.index >= first_trade_date]
            plot_title = f"均線交叉因子策略 vs BTC基準 (從首次交易日期 {first_trade_date_str} 開始)"
        else:
            factor_equity_aligned = factor_equity
            plot_title = "均線交叉因子策略 vs BTC基準"

        plot_walkforward_performance(
            results_df=pd.DataFrame(),  # 不需要
            combined_equity_curve=factor_equity_aligned,
            buy_hold_equity_curve=btc_equity,
            title=plot_title,
            save_path=output_dir / "equity_and_drawdown.png",
            figsize=(15, 10)
        )
        print(f"✓ 保存圖表: equity_and_drawdown.png")

    print()
    print("=" * 80)
    print(f"所有結果已保存到: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
