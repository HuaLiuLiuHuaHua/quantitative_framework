"""
Docstring for strategies.momentum.test_optimization_with_sensitivity
Momentum Strategy - 參數優化

功能:
1. 並行參數優化 (網格搜索或隨機搜索)
2. 選擇最佳夏普比率的參數
3. 使用最佳參數運行最終回測並生成圖表
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

from strategies.momentum.strategy import MomentumStrategy
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
        # 從kwargs獲取配置
        data = kwargs["data"]
        transaction_cost = kwargs["transaction_cost"]
        slippage = kwargs["slippage"]
        periods_per_year = kwargs["periods_per_year"]
        stop_loss_fee = kwargs.get("stop_loss_fee", 0.00055)
        
        # 解析參數
        leverage = int(params.get("leverage", 1))
        capital_allocation = float(params.get("capital_allocation", 1.0))
        final_leverage = leverage * capital_allocation

        # 生成策略信號
        strategy = MomentumStrategy()
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

        metrics = results.get('metrics', {})
        sharpe_ratio = metrics.get('sharpe_ratio', FAILED_SHARPE)
        calmar_ratio = metrics.get('calmar_ratio', FAILED_SHARPE)
        annual_return = metrics.get('annualized_return', 0)
        max_drawdown = metrics.get('max_drawdown', 0)

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
        error_info = {
            **params,
            "sharpe_ratio": FAILED_SHARPE,
            "calmar_ratio": FAILED_SHARPE,
            "annual_return": 0,
            "max_drawdown": 0,
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
    param_combinations = generate_all_combinations(param_grid)
    total_combinations = len(param_combinations)
    print(f"參數空間大小: {total_combinations:,} 個組合")

    if len(param_combinations) > n_trials:
        print(f"→ 組合數大於 N_TRIALS ({n_trials})，將從中隨機抽樣。")
        param_combinations = random.sample(param_combinations, n_trials)
    else:
        print("→ 使用所有有效組合進行評估。" )

    print(f"最終評估組合數: {len(param_combinations):,}\n")

    kwargs = {
        "data": data, "transaction_cost": transaction_cost, "slippage": slippage,
        "periods_per_year": periods_per_year, "stop_loss_fee": stop_loss_fee
    }
    
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    
    print(f"使用 {n_jobs} 個核心進行並行計算...")

    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        eval_func = partial(evaluate_single_params, **kwargs)
        results_iterator = executor.map(eval_func, param_combinations)
        for result in tqdm(results_iterator, total=len(param_combinations), desc="參數優化", unit="組合", ncols=100):
            results.append(result)
            
    return pd.DataFrame(results)


# ==================== 主函數 ====================

def main():
    """主函數"""
    print("=" * 80)
    print("Momentum Strategy - Optimization")
    print("=" * 80)
    print()

    # ========== 用戶配置區 ==========
    ASSET = "BTCUSDT"
    TIMEFRAME = "1h"
    START_DATE = "2022-11-01"
    END_DATE = "2024-12-31"

    PARAM_GRID = {
        "lookback_period": range(5, 500, 2),
        "leverage": range(1, 7, 1),
        "capital_allocation": np.arange(0.1, 1.01, 0.3),
    }

    N_TRIALS = 10000
    TRANSACTION_COST = 0.0005
    SLIPPAGE = 0.0002
    STOP_LOSS_FEE = 0.00055  # 和 cvilliq 策略一樣的止損費用
    N_JOBS = -1

    # 圖表和篩選的性能閾值
    SHARPE_THRESHOLD = 1.25
    CALMAR_THRESHOLD = 2.0
    ANNUAL_RETURN_THRESHOLD = 0.5
    MIN_ACCEPTABLE_DRAWDOWN = 0.3
    # ========== 配置區結束 ==========

    print(f"Asset: {ASSET} ({TIMEFRAME}) | Period: {START_DATE} to {END_DATE}")

    print(f"【Step 0】加載 {ASSET} {TIMEFRAME} 數據...")
    df = load_local_data(symbol=ASSET, data_source=TIMEFRAME, start_date=START_DATE, end_date=END_DATE)
    if df is None or df.empty:
        raise FileNotFoundError(f"無法加載數據 for {ASSET}_{TIMEFRAME}")
    print(f"  Loaded {len(df):,} bars")

    periods_per_year = 365 if TIMEFRAME == '1d' else 365 * 24

    print("\n【Step 1】運行參數優化...")
    results_df = run_optimization(
        param_grid=PARAM_GRID, data=df, transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE, periods_per_year=periods_per_year, n_trials=N_TRIALS,
        stop_loss_fee=STOP_LOSS_FEE, n_jobs=N_JOBS
    )
    
    print("\n【Step 2】尋找初步最佳參數...")
    valid_results = results_df[results_df['sharpe_ratio'] > FAILED_SHARPE].copy()
    if valid_results.empty:
        print("錯誤: 沒有有效的優化結果!")
        results_df.to_csv("debug_momentum_optimization.csv", index=False)
        print("調試文件已保存至 debug_momentum_optimization.csv")
        return

    # 先找到夏普最高的作為備選
    initial_best_result = valid_results.loc[valid_results['sharpe_ratio'].idxmax()]
    param_names = list(PARAM_GRID.keys())
    initial_best_params = {name: initial_best_result[name] for name in param_names}
    print(f"初步最佳夏普參數: {initial_best_params} (Sharpe: {initial_best_result['sharpe_ratio']:.4f})")

    print("\n【Step 3】進行敏感性分析並篩選穩健參數...")
    
    # 篩選夏普比率大於閾值的參數作為候選
    candidates_df = valid_results[valid_results['sharpe_ratio'] > SHARPE_THRESHOLD].copy()
    
    if candidates_df.empty:
        print(f"⚠️ 警告: 沒有參數組合的夏普比率超過 {SHARPE_THRESHOLD}，將使用初步最佳參數。")
        robust_df = pd.DataFrame()
        best_robust_params = initial_best_params
    else:
        print(f"找到 {len(candidates_df)} 個夏普 > {SHARPE_THRESHOLD} 的候選參數，開始計算穩健性...")
        # 注意：這裡的 filter_robust_params 假設它能處理 'turnover' 列不存在的情況
        robust_df, best_robust_params = filter_robust_params(
            candidates_df=candidates_df,
            all_results_df=results_df,
            param_grid=PARAM_GRID,
            metric="sharpe_ratio",
            sharpe_threshold=SHARPE_THRESHOLD,
            turnover_min=0, # momentum 策略沒有 turnover，設為 0
            turnover_max=999, # momentum 策略沒有 turnover，設為很大值
            top_n=10
        )

    if not best_robust_params:
        print("警告: 敏感性分析未找到穩健參數，將使用初步最佳夏普參數。")
        best_params = initial_best_params
        best_result = initial_best_result
    else:
        print(f"✅ 找到最佳穩健參數: {best_robust_params}")
        # 從原始結果中找到這組最佳穩健參數對應的完整性能數據
        key_cols = list(best_robust_params.keys())
        merged_df = pd.merge(results_df, pd.DataFrame([best_robust_params]), on=key_cols)
        if not merged_df.empty:
            best_result = merged_df.iloc[0]
            best_params = best_robust_params
        else:
            # 作為備份，如果合併失敗
            print("警告: 無法在原始結果中找到最佳穩健參數的數據，退回使用初步最佳參數。")
            best_params = initial_best_params
            best_result = initial_best_result

    print(f"\n最終選定參數: {best_params}")
    print(f"性能: Sharpe={best_result['sharpe_ratio']:.3f}, Return={best_result['annual_return']:.2%}, Drawdown={best_result['max_drawdown']:.2%}")

    print("\n【Step 4】準備輸出目錄...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "results" / f"Optimization_Momentum_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n【Step 5】使用最佳參數運行最終回測並記錄績效...")
    try:
        strategy = MomentumStrategy()
        best_signals = strategy.generate_signals(df, **best_params)
        
        # 計算最終槓桿
        leverage = int(best_params.get('leverage', 1))
        capital_allocation = float(best_params.get('capital_allocation', 1.0))
        final_leverage = leverage * capital_allocation
        
        best_engine = BacktestEngine(
            data=df, signals=best_signals, transaction_cost=TRANSACTION_COST, slippage=SLIPPAGE,
            initial_capital=100000, periods_per_year=periods_per_year, leverage=final_leverage,
            stop_loss_fee=STOP_LOSS_FEE
        )
        best_backtest_results = best_engine.run()
        
        final_metrics = best_backtest_results.get('metrics')
        if final_metrics:
            metrics_path = output_dir / "best_params_metrics.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(final_metrics, f, ensure_ascii=False, indent=4)
        
        print("\n【Step 6】生成報告和圖表...")
        plot_backtest_results(
            equity_curve=best_backtest_results['equity_curve'],
            data=df,
            signals=best_signals,
            title=f"Momentum Strategy - Best Params (Sharpe: {final_metrics['sharpe_ratio']:.2f})",
            save_path=output_dir / "equity_and_drawdown.png"
        )
        
    except Exception as e:
        print(f" 最終回測或繪圖失敗: {e}")

    results_df.to_csv(output_dir / "optimization_results_full.csv", index=False)
    with open(output_dir / "best_params.json", 'w', encoding='utf-8') as f:
        # Convert numpy types to native python types for json serialization
        best_params_native = {k: (int(v) if isinstance(v, np.integer) else v) for k, v in best_params.items()}
        json.dump(best_params_native, f, ensure_ascii=False, indent=4)

    # 新增：生成參數分析圖表
    if not results_df.empty:
        try:
            param_names_for_plot = [p for p in PARAM_GRID.keys() if len(list(PARAM_GRID.get(p, []))) > 1]
            
            plot_optimization_results_medium_style(
                results_df=results_df[results_df.sharpe_ratio > FAILED_SHARPE],
                param_names=param_names_for_plot,
                output_dir=output_dir,
                sharpe_threshold=SHARPE_THRESHOLD,
                calmar_threshold=CALMAR_THRESHOLD,
                annual_return_threshold=ANNUAL_RETURN_THRESHOLD,
                max_drawdown_threshold=-MIN_ACCEPTABLE_DRAWDOWN
            )
            print("  成功生成參數分析圖表。")
        except Exception as e:
            import traceback
            print(f"  生成參數分析圖表時出錯: {e}")
            print(traceback.format_exc())

    if not robust_df.empty:
        robust_df.to_csv(output_dir / "optimization_results_robust.csv", index=False)

    print(f"\n所有結果已保存至: {output_dir}")

if __name__ == "__main__":
    main()