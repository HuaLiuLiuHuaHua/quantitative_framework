# -*- coding: utf-8 -*-
"""
CVILLIQ CTA Strategy - Walk-Forward Optimization with Sensitivity Analysis

功能:
1. 使用新的增強版 WalkForward 模塊對 CVILLIQ 策略進行滾動優化
2. 在每個訓練窗口中，進行多階段參數篩選和穩健性分析
3. 在測試窗口中應用最佳參數，並將所有測試窗口的權益曲線拼接起來
4. 生成完整的滾動回測性能報告

此版本與 momentum 策略的回測邏輯完全相同。

作者: Claude Code
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
from strategies.cvilliq.strategy import CVILLIQStrategy


def main():
    print("=" * 80)
    print("CVILLIQ 策略 - Walk-Forward 優化與敏感性分析")
    print("=" * 80)
    print()

    # ========== 用戶配置區 ==========

    # 數據配置
    ASSET = "ETHUSDT"
    TIMEFRAME = "1h"
    START_DATE = "2023-01-01"
    END_DATE = "2025-10-31"

    # Walk-Forward 參數（小時級數據使用小時數）
    TRAIN_WINDOW_POINTS = 18720  # 約780天（小時級）
    TEST_WINDOW_POINTS = 720     # 約30天
    STEP_WINDOW_POINTS = 720     # 約30天

    # 參數網格
    PARAM_GRID = {
        'window': range(162, 261, 20),                           # 因子計算窗口
        'threshold_window': range(160, 259, 20),                 # 門檻計算窗口
        'long_entry_quantile': np.arange(0.4, 0.8, 0.1),        # 做多百分位
        'short_entry_quantile': np.arange(0.2, 0.6, 0.1),       # 做空百分位
        'leverage': [1, 2],                                      # 基礎槓桿
        'capital_allocation': [1.0]                              # 資金分配
    }

    # 回測配置
    BACKTEST_CONFIG = {
        'initial_capital': 100000,
        'commission_bps': 2,       # 2 bps
        'slippage_bps': 1,         # 1 bps
        'stop_loss_fee': 0.00055   # 5.5 bps
    }

    # 敏感性分析配置
    SENSITIVITY_CONFIG = {
        'sharpe_threshold': 1.25,
        'calmar_threshold': 2.0,
        'annual_return_threshold': 0.5,   # 50%
        'max_drawdown_threshold': 0.3     # 30%
    }

    # 隨機搜索次數和並行工作數
    N_TRIALS = 10000
    N_JOBS = -1  # 使用所有 CPU 核心

    # ========== 配置區結束 ==========

    print(f"配置:")
    print(f"  資產: {ASSET}")
    print(f"  時間範圍: {START_DATE} to {END_DATE}")
    print(f"  訓練窗口: {TRAIN_WINDOW_POINTS} 小時, 測試窗口: {TEST_WINDOW_POINTS} 小時, 步長: {STEP_WINDOW_POINTS} 小時")
    print()

    # Step 1: 加載數據
    print("【Step 1】加載數據...")
    data = load_local_data(
        symbol=ASSET,
        data_source=TIMEFRAME,
        start_date=START_DATE,
        end_date=END_DATE,
        lookback_days=TRAIN_WINDOW_POINTS + 60  # 額外加載數據以滿足最長 lookback
    )
    if data is None or data.empty:
        print(f"[錯誤] 無法加載 {ASSET} 的數據，程序終止。")
        return
    print(f"成功加載 {len(data)} 條數據。\n")

    # Step 2: 設置並運行 Walk-Forward 分析
    print("【Step 2】設置並運行 Walk-Forward 分析...")
    strategy = CVILLIQStrategy()

    walk_forward = WalkForward(
        strategy=strategy,
        data=data,
        start_date=START_DATE,
        end_date=END_DATE,
        train_window_points=TRAIN_WINDOW_POINTS,
        test_window_points=TEST_WINDOW_POINTS,
        step_window_points=STEP_WINDOW_POINTS,
        param_grid=PARAM_GRID,
        backtest_config=BACKTEST_CONFIG,
        sensitivity_config=SENSITIVITY_CONFIG,
        n_trials=N_TRIALS,
        n_jobs=N_JOBS
    )

    results = walk_forward.run()

    if not results:
        print("【錯誤】Walk-Forward 分析未產生結果。")
        return

    print("\nWalk-Forward 分析完成。\n")

    # Step 3: 顯示並保存結果
    print("【Step 3】顯示並保存結果...")

    perf_summary = results.get('performance_summary', {})
    if not perf_summary:
        print("【警告】未能生成最終性能摘要。\n")
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
            "train_window_points": TRAIN_WINDOW_POINTS,
            "test_window_points": TEST_WINDOW_POINTS,
            "step_window_points": STEP_WINDOW_POINTS,
        }
    }
    with open(output_dir / "summary_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    # 保存 Walk-Forward 結果
    walkforward_results = results.get('walkforward_results')
    if walkforward_results is not None and not walkforward_results.empty:
        walkforward_results.to_csv(output_dir / "walkforward_results.csv", index=False)

    # 保存參數演化
    params_evolution_df = results.get('params_evolution')
    if params_evolution_df is not None and not params_evolution_df.empty:
        params_evolution_df.to_csv(output_dir / "params_evolution.csv", index=False)

    # 保存權益曲線
    combined_equity = results.get('combined_equity')
    if combined_equity is not None and not combined_equity.empty:
        combined_equity.to_csv(output_dir / "combined_equity_curve.csv")

    print(f"\n所有結果已保存到: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
