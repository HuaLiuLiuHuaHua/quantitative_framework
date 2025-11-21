"""
CVILLIQ CTA Strategy - 參數優化 (簡化版本 - 無敏感性分析)

功能:
1. 自適應搜索策略 (網格搜索 vs 隨機搜索)
2. 並行參數優化 (4維參數空間)
3. 頂部參數篩選 (滿足條件的優質參數)
4. 簡化輸出 (只保留核心指標)

作者: Claude Code
"""

import sys
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
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

from strategies.cvilliq.strategy import CVILLIQStrategy
from shared.backtest import BacktestEngine
from shared.data_loader import load_local_data
from shared.visualization import plot_optimization_results_medium_style
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ==================== 輔助函數 ====================

# 失敗結果的默認值
FAILED_SHARPE = -999

def evaluate_single_params(params: dict, **kwargs) -> dict:
    """評估單個參數組合的輔助函數(用於並行計算)"""
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["PYTHONWARNINGS"] = "ignore"

    try:
        # 解析參數
        window = int(params["window"])
        long_threshold = float(params.get("long_threshold", 1.5))
        short_threshold = float(params.get("short_threshold", 0.5))
        leverage = int(params.get("leverage", 1))

        # 從kwargs獲取配置
        data = kwargs["data"]
        transaction_cost = kwargs["transaction_cost"]
        slippage = kwargs["slippage"]
        periods_per_year = kwargs["periods_per_year"]
        stop_loss_fee = kwargs.get("stop_loss_fee", 0.00055)

        # 生成策略信號
        strategy = CVILLIQStrategy()
        signals = strategy.generate_signals(
            data,
            window=window,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            leverage=leverage
        )

        # DEBUGGING: Print CVILLIQ stats for the first evaluation in each worker
        import threading
        if not hasattr(threading.current_thread(), "_printed_stats"):
            cvilliq_series = strategy.debug_data.get('cvilliq')
            if cvilliq_series is not None:
                print("\n--- CVILLIQ Stats (for one param set) ---")
                print(cvilliq_series.describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]))
                print("-------------------------------------------\n")
            threading.current_thread()._printed_stats = True
        # END DEBUGGING

        # 執行回測
        engine = BacktestEngine(
            data=data,
            signals=signals,
            transaction_cost=transaction_cost,
            slippage=slippage,
            initial_capital=100000,
            periods_per_year=periods_per_year,
            leverage=leverage,
            stop_loss_fee=stop_loss_fee
        )

        results = engine.run()

        # 提取績效指標 (僅保留4個核心指標)
        # 注：統一失敗默認值為 FAILED_SHARPE，避免混合結果篩選
        metrics = results.get('metrics', {})
        sharpe_ratio = metrics.get('sharpe_ratio', FAILED_SHARPE)
        calmar_ratio = metrics.get('calmar_ratio', FAILED_SHARPE)
        annual_return = metrics.get('annualized_return', FAILED_SHARPE)
        max_drawdown = metrics.get('max_drawdown', FAILED_SHARPE)

        return {
            **params,
            "sharpe_ratio": float(sharpe_ratio),
            "calmar_ratio": float(calmar_ratio),
            "annual_return": float(annual_return),
            "max_drawdown": float(max_drawdown)
        }

    except Exception as e:
        # 統一失敗默認值為 FAILED_SHARPE
        # 靜默失敗，避免干擾進度條顯示
        error_info = {
            **params,
            "sharpe_ratio": FAILED_SHARPE,
            "calmar_ratio": FAILED_SHARPE,
            "annual_return": FAILED_SHARPE,
            "max_drawdown": FAILED_SHARPE
        }
        # 添加診斷信息供後續分析
        if str(e):
            error_info["error_type"] = type(e).__name__
            error_info["error_msg"] = str(e)[:100]
        return error_info


def generate_all_combinations(param_grid: dict) -> list:
    """生成所有參數組合"""
    import itertools
    keys = param_grid.keys()
    values = [list(v) for v in param_grid.values()]
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return combinations


def run_optimization(
    param_grid: dict,
    data: pd.DataFrame,
    transaction_cost: float,
    slippage: float,
    periods_per_year: int,
    n_trials: int = 500,
    n_jobs: int = -1,
    stop_loss_fee: float = 0.00055,
    timeout: int = 300  # 5分鐘超時
) -> pd.DataFrame:
    """
    運行參數優化

    自適應策略:
    - 總組合數 < n_trials → 完整網格搜索
    - 總組合數 >= n_trials → 隨機搜索
    """
    # 計算總組合數
    total_combinations = 1
    for v in param_grid.values():
        total_combinations *= len(list(v))

    print(f"參數空間大小: {total_combinations:,} 個組合")
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

    print(f"實際評估組合數: {len(param_combinations):,}\n")

    # 構建kwargs
    kwargs = {
        "data": data,
        "transaction_cost": transaction_cost,
        "slippage": slippage,
        "periods_per_year": periods_per_year,
        "stop_loss_fee": stop_loss_fee
    }

    # 並行評估參數
    num_workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    worker = partial(evaluate_single_params, **kwargs)

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, params) for params in param_combinations]

        # 使用單一進度條顯示優化進度
        for future, params in tqdm(zip(futures, param_combinations), total=len(futures),
                                    desc="參數優化", unit="組合", ncols=100,
                                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except (TimeoutError, Exception):
                # 靜默處理失敗，避免干擾進度條
                results.append({
                    **params,
                    "sharpe_ratio": FAILED_SHARPE,
                    "calmar_ratio": FAILED_SHARPE,
                    "annual_return": FAILED_SHARPE,
                    "max_drawdown": FAILED_SHARPE,
                    "error_type": "timeout" if isinstance(future.exception(), TimeoutError) else type(future.exception()).__name__ if future.exception() else "unknown"
                })

    results_df = pd.DataFrame(results)
    return results_df


def plot_equity_and_drawdown(
    equity_curve: pd.Series,
    benchmark_data: pd.DataFrame,
    output_path: Path,
    strategy_name: str = "Strategy Equity",
    benchmark_name: str = "BTC Buy & Hold"
) -> None:
    """
    繪製策略權益曲線與回撤對比圖

    Normalizes both strategy and benchmark equity curves to start at 1.0
    to enable fair visual comparison of returns from the same starting point.

    Args:
        equity_curve: 策略權益曲線 (absolute values in currency)
        benchmark_data: 基準數據 (must contain 'close' column)
        output_path: 輸出路徑 (.png file)
        strategy_name: 策略名稱 (displayed in legend)
        benchmark_name: 基準名稱 (displayed in legend)

    Raises:
        ValueError: If equity_curve is empty or benchmark_data lacks 'close' column
    """
    # Validate inputs
    if equity_curve is None or len(equity_curve) == 0:
        raise ValueError("equity_curve is empty or None")
    if len(equity_curve) < 2:
        raise ValueError(f"equity_curve too short (length={len(equity_curve)}), need at least 2 points")
    if benchmark_data is None or benchmark_data.empty:
        raise ValueError("benchmark_data is empty or None")
    if 'close' not in benchmark_data.columns:
        raise ValueError("benchmark_data missing 'close' column")

    # Chart configuration constants
    FIGURE_WIDTH = 14  # inches, optimized for 1920px displays
    FIGURE_HEIGHT = 10  # inches
    EQUITY_DRAWDOWN_HEIGHT_RATIO = [3, 2]  # Equity gets more space than drawdown
    X_AXIS_MONTH_INTERVAL = 3  # Show labels every 3 months
    CHART_DPI = 150  # High quality for reports

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
                                    gridspec_kw={'height_ratios': EQUITY_DRAWDOWN_HEIGHT_RATIO})

    # ===== 修復1：確保權益曲線初始值有效 =====
    initial_capital = equity_curve.iloc[0]
    if initial_capital == 0 or np.isnan(initial_capital) or np.isinf(initial_capital):
        raise ValueError(f"Invalid initial equity value: {initial_capital}")
    
    # 計算策略和基準的歸一化權益曲線
    strategy_normalized = equity_curve / initial_capital

    # ===== 修復2：正確對齊基準數據與策略時間範圍 =====
    # 問題：舊的 reindex(method='ffill') 已棄用，且邏輯可能有問題
    # 改進：使用現代 pandas API 並更好地處理邊界情況
    
    # 找到策略時間範圍
    strategy_start = equity_curve.index[0]
    strategy_end = equity_curve.index[-1]
    
    # 過濾基準數據到策略時間範圍內
    benchmark_in_range = benchmark_data.loc[strategy_start:strategy_end, 'close'].copy()
    
    if benchmark_in_range.empty:
        # 如果沒有交集，使用前向填充從策略開始前開始
        benchmark_in_range = benchmark_data.loc[:strategy_end, 'close'].copy()
        if benchmark_in_range.empty:
            raise ValueError("No benchmark data available for the strategy period")
    
    # 使用前向填充處理缺失值
    benchmark_in_range = benchmark_in_range.ffill()
    
    # 對齊索引：確保基準數據完全覆蓋策略期間
    # 使用 reindex 然後使用 ffill() 方法（而不是棄用的 method='ffill' 參數）
    benchmark_aligned = benchmark_in_range.reindex(equity_curve.index).ffill()
    
    # 如果仍有 NaN，使用後向填充
    if benchmark_aligned.isna().any():
        benchmark_aligned = benchmark_aligned.bfill()
    
    # 驗證基準數據
    if benchmark_aligned.isna().any():
        raise ValueError("Cannot align benchmark data with strategy period - too many NaN values")
    
    # 計算基準的歸一化權益曲線
    benchmark_first_price = benchmark_aligned.iloc[0]
    if benchmark_first_price == 0 or np.isnan(benchmark_first_price):
        raise ValueError(f"Invalid initial benchmark price: {benchmark_first_price}")
    
    benchmark_normalized = benchmark_aligned / benchmark_first_price

    # ===== 修復3：正確計算最終收益率 =====
    strategy_final_return = (strategy_normalized.iloc[-1] - 1) * 100
    benchmark_final_return = (benchmark_normalized.iloc[-1] - 1) * 100

    # ===== 修復4：驗證權益曲線不是平坦的 =====
    if strategy_normalized.std() < 1e-6:
        print(f"警告: 策略權益曲線幾乎沒有變化 (std={strategy_normalized.std():.2e})")
        print(f"  最小值: {strategy_normalized.min():.6f}")
        print(f"  最大值: {strategy_normalized.max():.6f}")
        print(f"  平均值: {strategy_normalized.mean():.6f}")

    # ========== 上圖：權益曲線對比 ==========
    ax1.plot(strategy_normalized.index, strategy_normalized.values,
             label=strategy_name, color='#2E86AB', linewidth=1.5, alpha=0.9)
    ax1.plot(benchmark_normalized.index, benchmark_normalized.values,
             label=benchmark_name, color='#F77F00', linewidth=1.5, alpha=0.8)
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # 圖例顯示最終收益率
    ax1.legend([
        f'{strategy_name}: {strategy_final_return:.2f}%',
        f'{benchmark_name}: {benchmark_final_return:.2f}%',
        '初始資金 (1.0)'
    ], loc='upper left', fontsize=10, framealpha=0.9)

    ax1.set_ylabel('倍數', fontsize=12)
    ax1.set_title(f'{strategy_name} vs {benchmark_name} (從 {equity_curve.index[0].strftime("%Y-%m-%d")} 開始)',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('#F5F5F5')

    # ========== 下圖：回撤對比 ==========
    # ===== 修復5：使用更穩健的回撤計算方法 =====
    # 原始方法：strategy_drawdown = (strategy_normalized - strategy_cummax) / (strategy_cummax + epsilon)
    # 問題：加 epsilon 不符合回撤的標準定義
    # 修復：使用標準的回撤計算：(當前值 - 歷史最高) / 歷史最高
    
    strategy_cummax = strategy_normalized.expanding().max()
    # 避免除以零
    strategy_drawdown = np.where(
        strategy_cummax > 0,
        (strategy_normalized - strategy_cummax) / strategy_cummax * 100,
        0
    )

    benchmark_cummax = benchmark_normalized.expanding().max()
    benchmark_drawdown = np.where(
        benchmark_cummax > 0,
        (benchmark_normalized - benchmark_cummax) / benchmark_cummax * 100,
        0
    )

    # 計算最大回撤
    strategy_max_dd = np.min(strategy_drawdown)
    benchmark_max_dd = np.min(benchmark_drawdown)

    # 繪製回撤區域
    ax2.fill_between(equity_curve.index, strategy_drawdown, 0,
                     color='#2E86AB', alpha=0.4, label=f'{strategy_name}回撤')
    ax2.fill_between(equity_curve.index, benchmark_drawdown, 0,
                     color='#F77F00', alpha=0.3, label=f'{benchmark_name}回撤')

    # 繪製最大回撤參考線
    ax2.axhline(y=strategy_max_dd, color='#2E86AB', linestyle='--',
                linewidth=1, alpha=0.7)
    ax2.axhline(y=benchmark_max_dd, color='#F77F00', linestyle='--',
                linewidth=1, alpha=0.7)

    # 圖例顯示最大回撤
    ax2.legend([
        f'{strategy_name}回撤',
        f'{benchmark_name}回撤',
        f'{strategy_name}最大回撤: {strategy_max_dd:.2f}%',
        f'{benchmark_name}最大回撤: {benchmark_max_dd:.2f}%'
    ], loc='lower left', fontsize=10, framealpha=0.9)

    ax2.set_xlabel('時間', fontsize=12)
    ax2.set_ylabel('回撤 (%)', fontsize=12)
    ax2.set_title('Walk-Forward 回撤曲線比較', fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor('#F5F5F5')

    # 格式化 x 軸時間
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=X_AXIS_MONTH_INTERVAL))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    
    # ===== 修復6：確保輸出目錄存在 =====
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=CHART_DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存權益與回撤圖表: {output_path.name}")


def filter_top_params(
    results_df: pd.DataFrame,
    sharpe_threshold: float = 1.25,
    calmar_threshold: float = 2.5,
    annual_return_threshold: float = 0.5,
    max_drawdown_threshold: float = 0.2
) -> pd.DataFrame:
    """
    篩選符合條件的優質參數組合

    條件:
    - sharpe_ratio > sharpe_threshold (default 1.25)
    - calmar_ratio > calmar_threshold (default 2.5)
    - annual_return > annual_return_threshold (default 50%)
    - max_drawdown < max_drawdown_threshold (default 20%)

    Issue #9: 報告篩選統計信息
    """
    # 過濾失敗結果
    valid_df = results_df[results_df['sharpe_ratio'] > FAILED_SHARPE].copy()

    if valid_df.empty:
        print("警告: 沒有有效的優化結果")
        return pd.DataFrame()

    print(f"篩選統計 (Issue #9):")
    print(f"  總參數組合數: {len(results_df)}")
    print(f"  有效結果數: {len(valid_df)}")

    # 應用所有篩選條件
    filtered_df = valid_df[
        (valid_df['sharpe_ratio'] > sharpe_threshold) &
        (valid_df['calmar_ratio'] > calmar_threshold) &
        (valid_df['annual_return'] > annual_return_threshold) &
        (valid_df['max_drawdown'] < max_drawdown_threshold)
    ].copy()

    # 按夏普比率降序排列
    filtered_df = filtered_df.sort_values('sharpe_ratio', ascending=False)

    print(f"\n篩選條件:")
    print(f"  - 夏普比率 > {sharpe_threshold}")
    print(f"  - 卡瑪比率 > {calmar_threshold}")
    print(f"  - 年報酬 > {annual_return_threshold*100:.0f}%")
    print(f"  - 最大回撤 < {max_drawdown_threshold*100:.0f}%")
    print(f"\n符合條件的參數組合: {len(filtered_df)} 個")

    return filtered_df


# ==================== 主函數 ====================

def main():
    """主函數"""
    print("=" * 80)
    print("CVILLIQ CTA 策略 - 參數優化")
    print("=" * 80)
    print()

    # ========== 用戶配置區 ==========

    # 數據配置
    DATA_CONFIG = "BTCUSDT_1h"  # 單一標的 OHLCV 數據
    START_DATE = "2022-11-01"
    END_DATE = "2024-12-31"

    # 參數網格 (4維空間)
    # 修復 Issue: long_threshold 必須在 CVILLIQ 數據範圍內
    # CVILLIQ 數據分布: min=0.34, max=1.43, mean=0.71
    # 舊參數導致: long_threshold=1.6-3.1 永不觸發 (超出最大值 1.43)
    # 新參數調整: 讓多空信號都能正常生成
    PARAM_GRID = {
        "window": range(5, 450, 15),                    # CVILLIQ 計算窗口 [5, 20, 35, ..., 440]
        "long_threshold": np.arange(0.80, 1.50, 0.10).tolist(),   # ← 改為 0.80-1.50 (在數據範圍內)
        "short_threshold": np.arange(0.35, 0.80, 0.10).tolist(),   # ← 改為 0.35-0.80 (在數據範圍內)
        "leverage": [1]                    # 槓桿倍數
    }

    # 隨機搜索次數
    N_TRIALS = 10000

    # 頂部參數篩選條件
    SHARPE_THRESHOLD = 1.25
    CALMAR_THRESHOLD = 2.5
    ANNUAL_RETURN_THRESHOLD = 0.5  # 50%
    MAX_DRAWDOWN_THRESHOLD = 0.2   # 20%

    # 交易成本
    TRANSACTION_COST = 0.0002  # 0.02%
    SLIPPAGE = 0.0001          # 0.01%
    STOP_LOSS_FEE = 0.00055    # 0.055% (止損手續費)

    # 並行計算設置
    N_JOBS = -1  # -1表示使用所有CPU核心

    # ========== 配置區結束 ==========

    print(f"數據配置: {DATA_CONFIG}")
    print(f"優化期間: {START_DATE} 至 {END_DATE}")
    print(f"參數空間:")
    for param_name, param_range in PARAM_GRID.items():
        param_list = list(param_range) if hasattr(param_range, '__iter__') and not isinstance(param_range, str) else param_range
        if isinstance(param_list, list) and len(param_list) > 3:
            print(f"  {param_name}: {param_list[:3]}...{param_list[-3:]} ({len(param_list)} 個值)")
        else:
            print(f"  {param_name}: {param_list}")
    print(f"隨機搜索次數: {N_TRIALS}")
    print()

    # Step 0: 加載數據
    symbol, timeframe = DATA_CONFIG.rsplit('_', 1)
    print(f"【Step 0】加載 {symbol} {timeframe} 數據")
    print("-" * 80)

    df = load_local_data(
        symbol=symbol,
        data_source=timeframe,
        start_date=START_DATE,
        end_date=END_DATE,
        verbose=True
    )

    if df is None or df.empty:
        raise FileNotFoundError(f"無法加載 {symbol}_{timeframe} 的數據")

    print(f"✓ 數據加載成功: {len(df):,} 行\n")

    # 設定年化參數
    periods_per_year = 365 if timeframe == "1d" else 365 * 24

    # Step 1: 運行參數優化
    print("【Step 1】運行參數優化")
    print("-" * 80)

    results_df = run_optimization(
        param_grid=PARAM_GRID,
        data=df,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE,
        periods_per_year=periods_per_year,
        n_trials=N_TRIALS,
        n_jobs=N_JOBS,
        stop_loss_fee=STOP_LOSS_FEE
    )

    print(f"優化完成,共評估 {len(results_df):,} 個參數組合")
    print(f"有效結果數量: {(results_df['sharpe_ratio'] > FAILED_SHARPE).sum():,}\n")

    # Step 2: 篩選符合條件的優質參數
    print("【Step 2】篩選符合條件的優質參數")
    print("-" * 80)

    top_params_df = filter_top_params(
        results_df=results_df,
        sharpe_threshold=SHARPE_THRESHOLD,
        calmar_threshold=CALMAR_THRESHOLD,
        annual_return_threshold=ANNUAL_RETURN_THRESHOLD,
        max_drawdown_threshold=MAX_DRAWDOWN_THRESHOLD
    )

    # 選擇最優參數：優先從符合條件的參數中選擇，否則選擇全局最優
    param_names = list(PARAM_GRID.keys())

    if not top_params_df.empty:
        # 從符合條件的參數中選擇夏普比率最高的
        best_result = top_params_df.iloc[0]
        best_params = {name: best_result[name] for name in param_names}
        print(f"\n✓ 從符合條件的參數中選出最優參數:")
        print(f"前3個參數組合:")
        for idx, (_, row) in enumerate(top_params_df.head(3).iterrows(), 1):
            print(f"  {idx}. window={int(row['window'])}, "
                  f"long_thr={row['long_threshold']:.2f}, "
                  f"short_thr={row['short_threshold']:.2f}, "
                  f"lev={int(row['leverage'])} → "
                  f"夏普={row['sharpe_ratio']:.4f}, 卡瑪={row['calmar_ratio']:.4f}")
    else:
        # 沒有符合條件的參數，選擇全局最優
        valid_results = results_df[results_df['sharpe_ratio'] > FAILED_SHARPE].copy()
        if valid_results.empty:
            print("錯誤: 沒有有效的優化結果!")
            return

        best_idx = valid_results['sharpe_ratio'].idxmax()
        best_result = valid_results.loc[best_idx]
        best_params = {name: best_result[name] for name in param_names}
        print(f"\n⚠ 沒有符合所有條件的參數，選擇全局最優參數:")

    print(f"\n最優參數: {best_params}")
    print(f"  夏普比率: {best_result['sharpe_ratio']:.4f}")
    print(f"  卡瑪比率: {best_result['calmar_ratio']:.4f}")
    print(f"  年報酬: {best_result['annual_return']*100:.2f}%")
    print(f"  最大回撤: {best_result['max_drawdown']*100:.2f}%\n")

    # Step 3: 使用最優參數運行回測並生成權益圖表
    print("【Step 3】使用最優參數運行回測")
    print("-" * 80)

    try:
        # 生成最優參數的策略信號
        strategy = CVILLIQStrategy()
        best_signals = strategy.generate_signals(df, **best_params)

        # 執行回測
        best_engine = BacktestEngine(
            data=df,
            signals=best_signals,
            transaction_cost=TRANSACTION_COST,
            slippage=SLIPPAGE,
            initial_capital=100000,
            periods_per_year=periods_per_year,
            leverage=int(best_params['leverage']),
            stop_loss_fee=STOP_LOSS_FEE
        )
        best_backtest_results = best_engine.run()

        print(f"✓ 最優參數回測完成")
        print(f"  最終權益: ${best_backtest_results['equity_curve'].iloc[-1]:,.2f}")

    except Exception as e:
        print(f"⚠ 最優參數回測失敗: {str(e)}")
        print(f"   失敗參數: {best_params}")
        import traceback
        traceback.print_exc()
        best_backtest_results = None

    print()

    # Step 4: 生成報告和圖表
    print("【Step 4】生成報告和圖表")
    print("-" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "results" / f"Optimization_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成權益與回撤對比圖表
    if best_backtest_results is not None:
        try:
            plot_equity_and_drawdown(
                equity_curve=best_backtest_results['equity_curve'],
                benchmark_data=df,
                output_path=output_dir / "equity_and_drawdown.png",
                strategy_name="策略權益",
                benchmark_name=f"{DATA_CONFIG.split('_')[0]}持有"
            )
        except Exception as e:
            print(f"⚠ 權益回撤圖表生成失敗: {str(e)}")

    # 保存優化結果
    results_df_export = results_df.copy()
    # 只保留有效列 (排除 error_type 和 error_msg) (Issue #11)
    error_cols = [col for col in results_df_export.columns if col.startswith('error')]
    export_cols = [col for col in results_df_export.columns if not col.startswith('error')]

    # 診斷信息：統計失敗次數
    num_failures = (results_df['sharpe_ratio'] == FAILED_SHARPE).sum()
    if error_cols and num_failures > 0:
        print(f"⚠ 檢測到 {num_failures} 個失敗的參數組合（已移除錯誤列）")

    results_df_export = results_df_export[export_cols]
    results_df_export.to_csv(output_dir / "optimization_results.csv", index=False)
    print(f"✓ 保存優化結果: optimization_results.csv ({len(results_df_export):,} 行)")

    # 保存頂部參數
    if not top_params_df.empty:
        top_params_export = top_params_df[export_cols]
        top_params_export.to_csv(output_dir / "top_params.csv", index=False)
        print(f"✓ 保存頂部參數: top_params.csv ({len(top_params_export)} 行)")

    # 保存最佳參數 (Issue #14: 動態序列化，無硬編碼排除)
    def serialize_param(v) -> any:
        """動態序列化參數值"""
        if isinstance(v, (np.integer, np.int64, int)):
            return int(v)
        elif isinstance(v, (float, np.floating)):
            return float(v)
        else:
            return v

    best_params_summary = {
        "best_params_global": {k: serialize_param(v) for k, v in best_params.items()},
        "performance_metrics": {
            "sharpe_ratio": float(best_result['sharpe_ratio']),
            "calmar_ratio": float(best_result['calmar_ratio']),
            "annual_return": float(best_result['annual_return']),
            "max_drawdown": float(best_result['max_drawdown'])
        },
        "configuration": {
            "start_date": START_DATE,
            "end_date": END_DATE,
            "data_config": DATA_CONFIG,
            "sharpe_threshold": SHARPE_THRESHOLD,
            "calmar_threshold": CALMAR_THRESHOLD,
            "annual_return_threshold": ANNUAL_RETURN_THRESHOLD,
            "max_drawdown_threshold": MAX_DRAWDOWN_THRESHOLD
        }
    }

    with open(output_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params_summary, f, indent=4, ensure_ascii=False)
    print(f"✓ 保存最佳參數: best_params.json")

    # 繪製優化結果圖表
    results_df_copy = results_df.copy()
    if 'final_value' not in results_df_copy.columns:
        if 'annual_return' not in results_df_copy.columns:
            results_df_copy['annual_return'] = 0.0
        results_df_copy['final_value'] = 100000 * (1 + results_df_copy['annual_return'].fillna(0))

    try:
        plot_optimization_results_medium_style(
            results_df=results_df_copy,
            param_names=param_names,
            output_dir=output_dir
        )
        print(f"✓ 保存圖表: optimization_results_*.png")
    except Exception as e:
        print(f"⚠ 圖表生成失敗: {str(e)}")

    print()
    print("=" * 80)
    print(f"所有結果已保存到: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
