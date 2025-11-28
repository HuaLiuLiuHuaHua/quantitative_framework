#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CVILLIQ CTA Strategy - 參數優化 (動態門檻版本)

功能:
1. 自適應搜索策略 (網格搜索 vs 隨機搜索)
2. 並行參數優化
3. 頂部參數篩選 (滿足條件的優質參數)
4. 簡化輸出 (只保留核心指標)
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
from shared.sensitivity import filter_robust_params
from shared.visualization import plot_backtest_results, plot_optimization_results_medium_style


# ==================== 輔助函數 ====================

# 失敗結果的默認值
FAILED_SHARPE = -999

def evaluate_single_params(params: dict, **kwargs) -> dict:
    """評估單個參數組合的輔助函數(用於並行計算)"""
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["PYTHONWARNINGS"] = "ignore"

    try:
        # 解析參數
        leverage = int(params.get("leverage", 1))
        capital_allocation = float(params.get("capital_allocation", 1.0))
        final_leverage = leverage * capital_allocation

        # 從kwargs獲取配置
        data = kwargs["data"]
        transaction_cost = kwargs["transaction_cost"]
        slippage = kwargs["slippage"]
        periods_per_year = kwargs["periods_per_year"]
        stop_loss_fee = kwargs.get("stop_loss_fee", 0.00055)

        # 生成策略信號
        strategy = CVILLIQStrategy()
        signals = strategy.generate_signals(data, **params)

        # 執行回測
        engine = BacktestEngine(
            data=data,
            signals=signals,
            transaction_cost=transaction_cost,
            slippage=slippage,
            initial_capital=100000,
            periods_per_year=periods_per_year,
            leverage=final_leverage,
            stop_loss_fee=stop_loss_fee
        )

        results = engine.run()

        # 提取績效指標（使用小寫下劃線格式，與 BacktestEngine 返回的鍵名一致）
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
            "max_drawdown": float(max_drawdown),
            "error_type": None,
            "error_msg": None
        }

    except Exception as e:
        # 統一失敗默認值
        error_info = {
            **params,
            "sharpe_ratio": FAILED_SHARPE,
            "calmar_ratio": FAILED_SHARPE,
            "annual_return": FAILED_SHARPE,
            "max_drawdown": FAILED_SHARPE,
            "error_type": type(e).__name__,
            "error_msg": str(e)[:100]
        }
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
    stop_loss_fee: float = 0.00055,
    n_jobs: int = -1
) -> pd.DataFrame:
    """運行參數優化"""
    print("→ 生成所有可能的參數組合...")
    all_combinations = generate_all_combinations(param_grid)
    total_combinations = len(all_combinations)
    print(f"參數空間大小: {total_combinations:,} 個組合")

    # 1. 先移除 short_entry_quantile > long_entry_quantile 的無效組合
    param_combinations = all_combinations
    if "long_entry_quantile" in param_grid and "short_entry_quantile" in param_grid:
        original_count = len(param_combinations)
        # Corrected logic: Keep combinations where long_q >= short_q
        param_combinations = [
            p for p in param_combinations
            if p.get('long_entry_quantile', 1.0) >= p.get('short_entry_quantile', 0.0)
        ]
        filtered_count = original_count - len(param_combinations)
        if filtered_count > 0:
            print(f"過濾掉 {filtered_count:,} 個無效組合 (short_q > long_q)。")

    if not param_combinations:
        print("\n錯誤：過濾後沒有有效的參數組合可供評估。請檢查您的 PARAM_GRID。")
        return pd.DataFrame()

    print(f"有效組合數: {len(param_combinations):,}")

    # 2. 判斷參數組合是否 > N_TRIALS
    if len(param_combinations) > n_trials:
        print(f"→ 有效組合數大於 N_TRIALS ({n_trials})，將從中隨機抽樣。")
        param_combinations = random.sample(param_combinations, n_trials)
    else:
        print("→ 使用所有有效組合進行評估。")

    print(f"最終評估組合數: {len(param_combinations):,}\n")


    kwargs = {
        "data": data, "transaction_cost": transaction_cost, "slippage": slippage,
        "periods_per_year": periods_per_year, "stop_loss_fee": stop_loss_fee
    }
    
    # 3. 檢查 N_JOBS 是否有作用並行化
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    
    print(f"使用 {n_jobs} 個核心進行並行計算...")

    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # 使用 partial 函數將固定參數傳遞給 evaluate_single_params
        eval_func = partial(evaluate_single_params, **kwargs)
        
        # 使用 tqdm 顯示進度
        results_iterator = executor.map(eval_func, param_combinations)
        for result in tqdm(results_iterator, total=len(param_combinations), desc="參數優化", unit="組合", ncols=100):
            results.append(result)
            
    return pd.DataFrame(results)


# ==================== 主函數 ====================

def main():
    """主函數"""
    print("=" * 80)
    print("CVILLIQ CTA Strategy - Dynamic Threshold Optimization")
    print("=" * 80)
    print()

    # ========== 用戶配置區 ==========
    DATA_CONFIG = "BTCUSDT_1h"
    START_DATE = "2022-10-01"
    END_DATE = "2024-12-31"

    PARAM_GRID = {
        "window": range(5, 450, 15),                            # 因子計算窗口（簡化為 5 個值用於快速測試）
        "threshold_window": range(10, 450, 14),                 # 門檻計算窗口（簡化為 4 個值用於快速測試）
        "long_entry_quantile": np.arange(0.5, 0.6, 0.2),      # 做多百分位（簡化為 2 個值）
        "short_entry_quantile": np.arange(0.5, 0.6, 0.2),     # 做空百分位（簡化為 2 個值）
        "leverage": range(1, 2),                                        # 基礎槓桿
        "capital_allocation": np.arange(0.1, 1.01, 0.1)         # 資金分配
    }

    N_TRIALS = 10000  # 簡化參數範圍後，減少試驗次數以加快速度
    SHARPE_THRESHOLD = 1.25  # 降低閾值以增加找到符合條件結果的機會
    CALMAR_THRESHOLD = 2  # 降低閾值
    ANNUAL_RETURN_THRESHOLD = 0.5  # 不要求最小報酬
    MIN_ACCEPTABLE_DRAWDOWN = 0.3  # 允許最大回撤 30% 的策略（較為寬鬆）

    TRANSACTION_COST = 0.0002
    SLIPPAGE = 0.0001
    STOP_LOSS_FEE = 0.00055
    N_JOBS = -1
    # ========== 配置區結束 ==========

    print(f"Data: {DATA_CONFIG} | Period: {START_DATE} to {END_DATE}")

    symbol, timeframe = DATA_CONFIG.rsplit('_', 1)
    print(f"【Step 0】加載 {symbol} {timeframe} 數據...")
    df = load_local_data(symbol=symbol, data_source=timeframe, start_date=START_DATE, end_date=END_DATE)
    if df is None or df.empty:
        raise FileNotFoundError(f"無法加載數據 for {symbol}_{timeframe}")
    print(f"  Loaded {len(df):,} bars")

    periods_per_year = 365 * 24 if 'h' in timeframe else 365

    print("【Step 1】運行參數優化...")
    results_df = run_optimization(
        param_grid=PARAM_GRID, data=df, transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE, periods_per_year=periods_per_year, n_trials=N_TRIALS,
        stop_loss_fee=STOP_LOSS_FEE, n_jobs=N_JOBS
    )
    results_df.to_csv("debug_results.csv", index=False)
    print(f"  Evaluated {len(results_df):,} combinations")

    print("【Step 2】篩選有效結果並尋找最佳參數...")
    valid_results = results_df[results_df['sharpe_ratio'] > FAILED_SHARPE].copy()
    if valid_results.empty:
        print("錯誤: 沒有任何有效的優化結果!")
        results_df.to_csv("debug_cvilliq_optimization.csv", index=False)
        print("調試文件已保存至 debug_cvilliq_optimization.csv")
        return

    param_names = list(PARAM_GRID.keys())
    best_params = None
    best_result = None
    robust_df = pd.DataFrame() # Initialize empty dataframe for logging

    print("\n【Step 3】篩選最佳參數...")
    # 第一階段：篩選同時符合所有績效門檻的參數
    strong_candidates_df = valid_results[
        (valid_results['sharpe_ratio'] > SHARPE_THRESHOLD) &
        (valid_results['calmar_ratio'] > CALMAR_THRESHOLD) &
        (valid_results['annual_return'] > ANNUAL_RETURN_THRESHOLD) &
        (valid_results['max_drawdown'] > -MIN_ACCEPTABLE_DRAWDOWN)  # max_drawdown is negative
    ].copy()

    if not strong_candidates_df.empty:
        print(f"✅ 第一階段：找到 {len(strong_candidates_df)} 個滿足所有績效門檻的優質參數。")
        print("→ 現在對這些優質參數進行穩健性分析...")
        
        robust_df, best_robust_params = filter_robust_params(
            candidates_df=strong_candidates_df,
            all_results_df=results_df,
            param_grid=PARAM_GRID,
            metric="sharpe_ratio",
            sharpe_threshold=SHARPE_THRESHOLD,
            turnover_min=0,
            turnover_max=999,
            top_n=10 
        )

        if best_robust_params:
            print(f"✅ 穩健性分析完成，選出穩健性分數最高的參數為最佳參數: {best_robust_params}")
            best_params = best_robust_params
        else:
            print("⚠️ 穩健性分析未找到最佳參數，將從優質參數中選擇夏普比率最高者。")
            best_result_row = strong_candidates_df.loc[strong_candidates_df['sharpe_ratio'].idxmax()]
            best_params = {name: best_result_row[name] for name in param_names}
    
    else:
        print(f"⚠️ 第一階段：未找到滿足所有門檻 (夏普>{SHARPE_THRESHOLD}, 卡瑪>{CALMAR_THRESHOLD}, 年報酬>{ANNUAL_RETURN_THRESHOLD:.0%}, 最大回撤<{-MIN_ACCEPTABLE_DRAWDOWN:.0%}) 的參數。")
        print("→ 退回策略：直接選擇所有結果中夏普比率最高的參數。")
        best_result_row = valid_results.loc[valid_results['sharpe_ratio'].idxmax()]
        best_params = {name: best_result_row[name] for name in param_names}

    # 獲取並打印最終選定參數的性能
    if best_params:
        key_cols = list(best_params.keys())
        merged_df = pd.merge(valid_results, pd.DataFrame([best_params]), on=key_cols)
        if not merged_df.empty:
            best_result = merged_df.iloc[0]
            print(f"\n✅ 最終選定參數: {best_params}")
            print(f"性能: Sharpe={best_result['sharpe_ratio']:.3f}, Calmar={best_result['calmar_ratio']:.3f}, Return={best_result['annual_return']:.2%}, Drawdown={best_result['max_drawdown']:.2%}")
        else:
            print(f"警告: 在原始結果中找不到最佳參數 {best_params} 的數據，將無法顯示最終性能。")
            # Fallback for safety, though it shouldn't happen
            best_result = pd.Series(dtype='float64')
    else:
         print("錯誤: 最終未能確定任何最佳參數。")
         return

    print("\n【Step 4】準備輸出目錄...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "results" / f"Optimization_Dynamic_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n【Step 5】使用最佳參數運行最終回測並記錄績效...")
    best_backtest_results = None
    try:
        strategy = CVILLIQStrategy()
        best_signals = strategy.generate_signals(df, **best_params)
        
        final_leverage = best_params.get('leverage', 1) * best_params.get('capital_allocation', 1.0)
        best_engine = BacktestEngine(
            data=df, signals=best_signals, transaction_cost=TRANSACTION_COST, slippage=SLIPPAGE,
            initial_capital=100000, periods_per_year=periods_per_year, leverage=final_leverage,
            stop_loss_fee=STOP_LOSS_FEE
        )
        best_backtest_results = best_engine.run()
        final_equity = best_backtest_results['equity_curve'].iloc[-1]
        print(f"  Final equity: ${final_equity:,.2f}")

        final_metrics = best_backtest_results.get('metrics')
        if final_metrics:
            metrics_path = output_dir / "best_params_metrics.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(final_metrics, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f" 最終回測失敗: {e}")

    print("\n【Step 6】生成報告和圖表...")

    if best_backtest_results and 'equity_curve' in best_backtest_results and not best_backtest_results['equity_curve'].empty:
        try:
            plot_backtest_results(
                equity_curve=best_backtest_results['equity_curve'],
                data=df,
                signals=best_signals,
                title=f"Best Params (Sharpe: {best_result['sharpe_ratio']:.2f})",
                save_path=output_dir / "equity_and_drawdown.png",
                show_buy_and_hold=True
            )
        except Exception as e:
            print(f"  Error generating equity curve: {e}")

    results_df.to_csv(output_dir / "optimization_results_full.csv", index=False)

    if not results_df.empty:
        try:
            param_names_for_plot = [p for p in PARAM_GRID.keys() if len(list(PARAM_GRID[p])) > 1]
            plot_optimization_results_medium_style(
                results_df=results_df[results_df.sharpe_ratio > FAILED_SHARPE],
                param_names=param_names_for_plot,
                output_dir=output_dir,
                sharpe_threshold=SHARPE_THRESHOLD,
                calmar_threshold=CALMAR_THRESHOLD,
                annual_return_threshold=ANNUAL_RETURN_THRESHOLD,
                max_drawdown_threshold=-MIN_ACCEPTABLE_DRAWDOWN
            )
        except Exception as e:
            import traceback
            print(f"  Error generating parameter analysis: {e}")
            print(traceback.format_exc())

    if 'robust_df' in locals() and not robust_df.empty:
        robust_df.to_csv(output_dir / "optimization_results_robust.csv", index=False)
    
    with open(output_dir / "best_params.json", 'w', encoding='utf-8') as f:
        # Convert numpy types to native python types
        best_params_native = {k: v.item() if isinstance(v, np.generic) else v for k, v in best_params.items()}
        json.dump(best_params_native, f, ensure_ascii=False, indent=4)


    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
