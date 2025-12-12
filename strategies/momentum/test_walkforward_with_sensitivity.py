# -*- coding: utf-8 -*-
"""
動量策略 - Walk-Forward 優化與敏感性分析

功能:
1. 使用新的增強版 WalkForward 模塊對動量策略進行滾動優化
2. 在每個訓練窗口中，進行多階段參數篩選和穩健性分析
3. 在測試窗口中應用最佳參數，並將所有測試窗口的權益曲線拼接起來
4. 生成完整的滾動回測性能報告

此版本與 CVILLIQ 策略的回測邏輯完全相同。
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import numpy as np

# 將項目根目錄添加到 Python 路徑
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from shared.data_loader import load_local_data
from shared.walkforward import WalkForward
from shared.visualization import plot_walkforward_performance
from strategies.momentum.strategy import MomentumStrategy


def main():
    print("=" * 80)
    print("動量策略 - Walk-Forward 優化與敏感性分析")
    print("=" * 80)
    print()

    # ========== 用戶配置區 ==========

    # 數據配置
    ASSET = "BTCUSDT"
    TIMEFRAME = "1h"
    START_DATE = "2022-12-01"
    END_DATE = "2025-11-30"

    # Walk-Forward 參數（小時數據使用小時數）
    TRAIN_HOURS = 18720
    TEST_HOURS = 720
    STEP_HOURS = 720

    # 參數網格 (用於調試的臨時參數，較為敏感)
    PARAM_GRID = {
        'lookback_period': range(5, 1000, 1),       # 回看周期: 1-3天 (原: 400-500)
        'momentum_threshold': np.arange(0.0, 0.01, 0.01), # 動量閾值: 2%-5% (原: 30%-60%)
        'leverage': range(1, 2, 1),
        'capital_allocation': np.arange(0.1, 1.01, 0.1), # 資本分配: 10%-100% (原: 100%)
    }

    # 回測配置
    BACKTEST_CONFIG = {
        'initial_capital': 100000,
        'commission_bps': 5,      # 5 bps
        'slippage_bps': 2,        # 2 bps
        'stop_loss_fee': 0.00055  # 5.5 bps
    }

    # 敏感性分析配置
    SENSITIVITY_CONFIG = {
        'sharpe_threshold': 1.25,
        'calmar_threshold': 2.0,
        'annual_return_threshold': 0.5,  # 50%
        'max_drawdown_threshold': 0.3     # 30%
    }

    # 隨機搜索次數和並行工作數
    N_TRIALS = 10000
    N_JOBS = -1  # 使用所有 CPU 核心

    # ========== 配置區結束 ==========

    print(f"配置:")
    print(f"  資產: {ASSET}")
    print(f"  時間範圍: {START_DATE} to {END_DATE}")
    print(f"  訓練窗口: {TRAIN_HOURS} 小時, 測試窗口: {TEST_HOURS} 小時, 步長: {STEP_HOURS} 小時")
    print()

    # Step 1: 加載數據
    print("【Step 1】加載數據...")
    data = load_local_data(
        symbol=ASSET,
        data_source=TIMEFRAME,
        start_date=START_DATE,
        end_date=END_DATE,
        lookback_days=int(TRAIN_HOURS / 24) + 60  # 額外加載數據以滿足最長 lookback
    )
    if data is None or data.empty:
        print(f"[錯誤] 無法加載 {ASSET} 的數據，程序終止。")
        return
    print(f"成功加載 {len(data)} 條數據。\n")

    # Step 2: 設置並運行 Walk-Forward 分析
    print("【Step 2】設置並運行 Walk-Forward 分析...")
    strategy = MomentumStrategy()  # Instantiate the strategy

    walk_forward = WalkForward(
        strategy=strategy,
        data=data,
        start_date=START_DATE,
        end_date=END_DATE,
        train_window_points=TRAIN_HOURS,
        test_window_points=TEST_HOURS,
        step_window_points=STEP_HOURS,
        param_grid=PARAM_GRID,
        backtest_config=BACKTEST_CONFIG,
        sensitivity_config=SENSITIVITY_CONFIG,
        n_trials=N_TRIALS,
        n_jobs=N_JOBS
    )

    results = walk_forward.run()

    if not results:
        print("[錯誤] Walk-Forward 分析未產生結果。")
        return

    print("\nWalk-Forward 分析完成。\n")

    # Step 3: 顯示並保存結果
    print("【Step 3】顯示、保存結果與繪圖...")

    perf_summary = results.get('performance_summary', {})
    if not perf_summary:
        print("[警告] 未能生成最終性能摘要。\n")
    else:
        print("總體性能摘要:")
        for key, value in perf_summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print()

    # 創建輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "results" / f"WalkForward_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存性能摘要
    report = {
        'performance_summary': {k: (v if not isinstance(v, pd.Timestamp) else str(v)) for k, v in perf_summary.items()},
        'walk_forward_config': {
            "asset": ASSET,
            "timeframe": TIMEFRAME,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "train_hours": TRAIN_HOURS,
            "test_hours": TEST_HOURS,
            "step_hours": STEP_HOURS,
        }
    }
    with open(output_dir / "summary_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    # 保存 Walk-Forward 結果
    walkforward_results_df = results.get('walkforward_results')
    if walkforward_results_df is not None and not walkforward_results_df.empty:
        walkforward_results_df.to_csv(output_dir / "walkforward_results.csv", index=False)

    # 保存參數演化
    params_evolution_df = results.get('params_evolution')
    if params_evolution_df is not None and not params_evolution_df.empty:
        params_evolution_df.to_csv(output_dir / "params_evolution.csv", index=False)

    # 繪製並保存權益曲線圖
    combined_equity = results.get('combined_equity')
    if combined_equity is not None and not combined_equity.empty:
        # 計算長期持有基準
        equity_start_date = combined_equity.index[0]
        buy_hold_data = data[data.index >= equity_start_date]['close']
        buy_hold_equity = BACKTEST_CONFIG['initial_capital'] * (buy_hold_data / buy_hold_data.iloc[0])
        buy_hold_equity = buy_hold_equity.reindex(combined_equity.index).ffill()

        plot_walkforward_performance(
            results_df=walkforward_results_df,
            combined_equity_curve=combined_equity,
            buy_hold_equity_curve=buy_hold_equity,
            title=f"{ASSET} Walk-Forward Performance",
            save_path=output_dir / "walkforward_equity_curve.png"
        )

    print(f"\n所有結果已保存到: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
