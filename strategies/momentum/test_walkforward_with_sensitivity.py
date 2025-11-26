"動量策略 - 滾動優化與敏感性分析 (單一資產)

功能:
1. 使用 WalkForward 模塊對單一資產策略進行滾動優化。
2. 在每個訓練窗口中，使用 Optimizer 和 Sensitivity 找到最佳穩健參數。
3. 在測試窗口中應用最佳參數，並將所有測試窗口的權益曲線拼接起來。
4. 生成完整的滾動回測性能報告和圖表。
"

import sys
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

# 將項目根目錄添加到 Python 路徑
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from shared.data_loader import load_local_data
from shared.walkforward import WalkForward
from strategies.momentum.strategy import MomentumStrategy

def main():
    print("=" * 80)
    print("動量策略 - 滾動優化與敏感性分析 (單一資產)")
    print("=" * 80)
    print()

    # ========== 用戶配置區 ==========
    ASSET = "BTCUSDT"
    START_DATE = "2023-01-01"
    END_DATE = "2025-10-31"
    TIMEFRAME = "1d"
    
    # Walk-Forward 參數
    TRAIN_DAYS = 365
    TEST_DAYS = 90
    STEP_DAYS = 90

    # 優化參數網格
    PARAM_GRID = {
        'lookback_period': range(20, 201, 20),
        'leverage': [1, 2, 3],
    }

    # 回測配置
    COMMISSION_BPS = 5
    SLIPPAGE_BPS = 2
    INITIAL_CAPITAL = 100000
    
    # 敏感性分析配置
    SENSITIVITY_CONFIG = {
        'sharpe_threshold': 1.0,
        'performance_metric': 'sharpe_ratio',
    }
    # ========== 配置區結束 ==========

    print("配置:")
    print(f"  資產: {ASSET}")
    print(f"  時間範圍: {START_DATE} to {END_DATE}")
    print(f"  訓練窗口: {TRAIN_DAYS} 天, 測試窗口: {TEST_DAYS} 天, 步長: {STEP_DAYS} 天")
    print()

    # Step 1: 加載數據
    print("【Step 1】加載數據...")
    data = load_local_data(
        symbol=ASSET,
        data_source=TIMEFRAME,
        start_date=START_DATE,
        end_date=END_DATE,
        lookback_days=TRAIN_DAYS + 60 # 額外加載數據以滿足最長 lookback
    )
    if data is None or data.empty:
        print(f"[錯誤] 無法加載 {ASSET} 的數據，程序終止。")
        return
    print(f"成功加載 {len(data)} 條數據。\n")

    # Step 2: 設置並運行 Walk-Forward 分析
    print("【Step 2】設置並運行 Walk-Forward 分析...")
    strategy = MomentumStrategy()

    walk_forward = WalkForward(
        strategy=strategy,
        data=data,
        start_date=START_DATE,
        end_date=END_DATE,
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
        step_days=STEP_DAYS,
        param_grid=PARAM_GRID,
        backtest_config={
            'initial_capital': INITIAL_CAPITAL,
            'commission_bps': COMMISSION_BPS,
            'slippage_bps': SLIPPAGE_BPS,
        },
        sensitivity_config=SENSITIVITY_CONFIG,
    )

    results = walk_forward.run()

    if not results:
        print("[錯誤] Walk-Forward 分析未產生結果。")
        return
        
    print("Walk-Forward 分析完成。\n")

    # Step 3: 顯示並保存結果
    print("【Step 3】顯示並保存結果...")
    
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

    # 保存性能摘要和參數演變
    report = {
        'performance_summary': {k: (v if not isinstance(v, pd.Timestamp) else str(v)) for k, v in perf_summary.items()},
        'walk_forward_config': {
            "asset": ASSET,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "step_days": STEP_DAYS,
        }
    }
    with open(output_dir / "summary_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
        
    params_evolution_df = results.get('params_evolution')
    if params_evolution_df is not None:
        params_evolution_df.to_csv(output_dir / "params_evolution.csv")

    # 繪製並保存圖表 (由 walk_forward.run() 內部完成)
    print(f"\n所有結果和圖表已保存到: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
