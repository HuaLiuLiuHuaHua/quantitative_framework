# 量化交易框架核心模塊使用指南

本指南展示如何使用量化交易框架的6個核心模塊。

## 目錄

1. [回測引擎 (BacktestEngine)](#1-回測引擎)
2. [K線排列 (Permutation)](#2-k線排列)
3. [參數優化 (ParameterOptimizer)](#3-參數優化)
4. [MCPT測試 (MonteCarloTester)](#4-mcpt測試)
5. [Walk-Forward分析 (WalkForwardAnalyzer)](#5-walk-forward分析)
6. [可視化 (Visualization)](#6-可視化)
7. [完整工作流程](#7-完整工作流程)

---

## 1. 回測引擎

### 基本使用

```python
from quantitative_framework.shared import BacktestEngine
import pandas as pd

# 加載數據
df = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)

# 定義策略信號（1=多頭, -1=空頭, 0=空倉）
signals = pd.Series(0, index=df.index)
ma_short = df['close'].rolling(20).mean()
ma_long = df['close'].rolling(50).mean()
signals[ma_short > ma_long] = 1
signals[ma_short < ma_long] = -1

# 創建回測引擎
engine = BacktestEngine(
    data=df,
    signals=signals,
    transaction_cost=0.0006,  # 0.06%交易成本
    slippage=0.0001,          # 0.01%滑價
    initial_capital=100000,
    periods_per_year=8760     # 1小時K線
)

# 執行回測
results = engine.run()

# 打印結果
engine.print_summary()

# 保存結果
save_path = engine.save_results(data_source="bybit_btc_1h")
```

### 快速回測（用於優化循環）

```python
from quantitative_framework.shared import quick_backtest

returns, metrics = quick_backtest(
    data=df,
    signals=signals,
    transaction_cost=0.0006,
    slippage=0.0001
)

print(f"夏普比率: {metrics['sharpe_ratio']:.3f}")
print(f"總收益率: {metrics['total_return']*100:.2f}%")
```

---

## 2. K線排列

### 高級排列（MCPT-Main方法）

```python
from quantitative_framework.shared import advanced_permutation

# 基本使用：同時排列跳空和日內組件
permuted_df = advanced_permutation(
    df=df,
    start_index=100,  # 前100根K線保持不變（用於指標計算）
    seed=42,
    mode='both'  # 'both', 'intraday', 'gap'
)

# 驗證OHLC有效性
from quantitative_framework.shared import validate_ohlc
is_valid = validate_ohlc(permuted_df)
print(f"OHLC有效性: {is_valid}")
```

### 生成多個排列（用於MCPT）

```python
permutations = []
for i in range(1000):
    perm_df = advanced_permutation(df, start_index=100, seed=i)
    permutations.append(perm_df)
```

---

## 3. 參數優化

### 網格搜索優化

```python
from quantitative_framework.shared import ParameterOptimizer

# 定義策略函數
def ma_cross_strategy(data, fast_period=10, slow_period=20):
    ma_fast = data['close'].rolling(fast_period).mean()
    ma_slow = data['close'].rolling(slow_period).mean()

    signals = pd.Series(0, index=data.index)
    signals[ma_fast > ma_slow] = 1
    signals[ma_fast < ma_slow] = -1

    return signals

# 創建優化器
optimizer = ParameterOptimizer(
    strategy_func=ma_cross_strategy,
    data=df,
    param_grid={
        'fast_period': [5, 10, 15, 20, 25],
        'slow_period': [20, 30, 40, 50, 60]
    },
    objective='sharpe_ratio',  # 'profit_factor', 'calmar_ratio', 'total_return'
    transaction_cost=0.0006,
    slippage=0.0001,
    n_jobs=-1  # 使用所有CPU核心
)

# 執行優化
results_df = optimizer.optimize()

# 獲取最佳參數
best_params = optimizer.get_best_params()
print(f"最佳參數: {best_params}")

# 參數敏感性分析
sensitivity = optimizer.get_param_sensitivity('fast_period')
print(sensitivity)

# 保存結果
save_path = optimizer.save_results(data_source="bybit_btc_1h")
```

### 快速優化（簡化接口）

```python
from quantitative_framework.shared import quick_optimize

best_params = quick_optimize(
    strategy_func=ma_cross_strategy,
    data=df,
    param_grid={'fast_period': [10, 20, 30], 'slow_period': [40, 50, 60]},
    objective='sharpe_ratio',
    n_jobs=-1
)
```

---

## 4. MCPT測試

### Normal MCPT（固定參數）

```python
from quantitative_framework.shared import MonteCarloTester

# 使用固定參數進行MCPT測試
tester = MonteCarloTester(
    strategy_func=ma_cross_strategy,
    data=df,
    strategy_params={'fast_period': 10, 'slow_period': 30},  # 固定參數
    start_index=100,
    n_permutations=1000,  # 排列次數
    mode='normal',
    metric='sharpe_ratio',
    transaction_cost=0.0006,
    slippage=0.0001,
    n_jobs=-1
)

# 執行測試
results = tester.run()

# 查看結果
print(f"p值: {results['p_value']:.4f}")
print(f"顯著性: {results['is_significant']}")
print(f"百分位數: {results['percentile']:.2f}%")

# 保存結果
tester.save_results(data_source="bybit_btc_1h")
```

### Walk-Forward MCPT（更嚴格）

```python
# 每個排列都重新優化參數
tester_wf = MonteCarloTester(
    strategy_func=ma_cross_strategy,
    data=df,
    strategy_params={'fast_period': [10, 20, 30], 'slow_period': [40, 50, 60]},  # 參數網格
    start_index=100,
    n_permutations=500,  # 較少排列（因為每次要優化）
    mode='walkforward',
    metric='sharpe_ratio',
    n_jobs=-1
)

results_wf = tester_wf.run()
```

---

## 5. Walk-Forward分析

### 滾動窗口優化和測試

```python
from quantitative_framework.shared import WalkForwardAnalyzer

analyzer = WalkForwardAnalyzer(
    strategy_func=ma_cross_strategy,
    data=df,
    param_grid={
        'fast_period': [5, 10, 15, 20, 25],
        'slow_period': [20, 30, 40, 50, 60]
    },
    train_window=1000,  # 訓練期1000根K線
    test_window=250,    # 測試期250根K線
    step=250,           # 每次滾動250根K線（不重疊）
    objective='sharpe_ratio',
    transaction_cost=0.0006,
    slippage=0.0001,
    n_jobs=-1
)

# 執行分析
results_df = analyzer.run()

# 查看樣本外績效
print(f"平均OOS夏普比率: {results_df['oos_sharpe_ratio'].mean():.3f}")
print(f"OOS/IS夏普比率: {analyzer.combined_metrics['is_oos_sharpe_ratio']:.3f}")
print(f"一致性: {analyzer.combined_metrics['consistency']*100:.2f}%")

# 保存結果
save_path = analyzer.save_results(data_source="bybit_btc_1h")

# 查看每個窗口的最佳參數
print(results_df[['window_id', 'optimal_params', 'oos_sharpe_ratio']])
```

---

## 6. 可視化

### 回測結果可視化

```python
from quantitative_framework.shared import plot_backtest_results

plot_backtest_results(
    equity_curve=results['equity_curve'],
    data=df,
    signals=signals,
    title="BTC 1H 移動平均策略回測",
    save_path=Path("results/backtest_plot.png")
)
```

### 參數優化結果可視化

```python
from quantitative_framework.shared import plot_optimization_results

plot_optimization_results(
    results_df=optimizer.results_df,
    param_names=['fast_period', 'slow_period'],
    metric='sharpe_ratio',
    top_n=10,
    title="參數優化熱圖",
    save_path=Path("results/optimization_plot.png")
)
```

### MCPT結果可視化

```python
from quantitative_framework.shared import plot_mcpt_distribution

plot_mcpt_distribution(
    original_score=results['original_score'],
    permutation_scores=np.array(results['permutation_scores']),
    p_value=results['p_value'],
    metric_name="夏普比率",
    title="MCPT排列測試",
    save_path=Path("results/mcpt_plot.png")
)
```

### Walk-Forward結果可視化

```python
from quantitative_framework.shared import plot_walkforward_performance

plot_walkforward_performance(
    results_df=analyzer.results_df,
    title="Walk-Forward分析",
    save_path=Path("results/walkforward_plot.png")
)
```

### 保存所有圖表

```python
from quantitative_framework.shared import save_all_plots

save_all_plots(
    backtest_results={
        'equity_curve': equity_curve,
        'data': df,
        'signals': signals
    },
    optimization_results={
        'results_df': optimizer.results_df,
        'param_names': ['fast_period', 'slow_period']
    },
    mcpt_results=mcpt_results,
    walkforward_results={'results_df': analyzer.results_df},
    output_dir=Path("results/plots")
)
```

---

## 7. 完整工作流程

### 標準量化策略研發流程

```python
import pandas as pd
import numpy as np
from pathlib import Path
from quantitative_framework.shared import *

# ============================================================
# 步驟1: 載入數據
# ============================================================
print("步驟1: 載入數據...")
df = pd.read_csv('bybit_btc_1h.csv', index_col=0, parse_dates=True)
print(f"數據範圍: {df.index[0]} 到 {df.index[-1]}")
print(f"數據點數: {len(df)}")

# ============================================================
# 步驟2: 定義策略
# ============================================================
print("\n步驟2: 定義策略...")

def my_strategy(data, fast_period=10, slow_period=30, atr_period=14):
    """雙均線+ATR過濾策略"""
    # 計算指標
    ma_fast = data['close'].rolling(fast_period).mean()
    ma_slow = data['close'].rolling(slow_period).mean()

    # ATR過濾
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    # 生成信號
    signals = pd.Series(0, index=data.index)

    # 均線交叉 + ATR確認
    long_condition = (ma_fast > ma_slow) & (atr > atr.rolling(50).mean())
    short_condition = (ma_fast < ma_slow) & (atr > atr.rolling(50).mean())

    signals[long_condition] = 1
    signals[short_condition] = -1

    return signals

# ============================================================
# 步驟3: 參數優化（In-Sample）
# ============================================================
print("\n步驟3: 執行參數優化...")

# 使用前70%數據進行優化
split_point = int(len(df) * 0.7)
train_data = df.iloc[:split_point]
test_data = df.iloc[split_point:]

optimizer = ParameterOptimizer(
    strategy_func=my_strategy,
    data=train_data,
    param_grid={
        'fast_period': [5, 10, 15, 20],
        'slow_period': [20, 30, 40, 50],
        'atr_period': [10, 14, 20]
    },
    objective='sharpe_ratio',
    transaction_cost=0.0006,
    slippage=0.0001,
    n_jobs=-1
)

opt_results = optimizer.optimize()
best_params = optimizer.get_best_params()
print(f"最佳參數: {best_params}")

# 保存優化結果
opt_save_path = optimizer.save_results(data_source="bybit_btc_1h_IS")

# 可視化優化結果
plot_optimization_results(
    results_df=opt_results,
    param_names=['fast_period', 'slow_period'],
    metric='sharpe_ratio',
    save_path=opt_save_path / "optimization_plot.png"
)

# ============================================================
# 步驟4: MCPT測試（驗證過度擬合）
# ============================================================
print("\n步驟4: 執行MCPT測試...")

mcpt_tester = MonteCarloTester(
    strategy_func=my_strategy,
    data=train_data,
    strategy_params=best_params,
    start_index=100,
    n_permutations=1000,
    mode='normal',
    metric='sharpe_ratio',
    n_jobs=-1
)

mcpt_results = mcpt_tester.run()
mcpt_save_path = mcpt_tester.save_results(data_source="bybit_btc_1h_MCPT")

# 可視化MCPT結果
plot_mcpt_distribution(
    original_score=mcpt_results['original_score'],
    permutation_scores=np.array(mcpt_results['permutation_scores']),
    p_value=mcpt_results['p_value'],
    metric_name="夏普比率",
    save_path=mcpt_save_path / "mcpt_plot.png"
)

if not mcpt_results['is_significant']:
    print("警告: MCPT測試未通過，策略可能過度擬合！")
else:
    print("MCPT測試通過，策略具有統計顯著性。")

# ============================================================
# 步驟5: Walk-Forward分析
# ============================================================
print("\n步驟5: 執行Walk-Forward分析...")

wf_analyzer = WalkForwardAnalyzer(
    strategy_func=my_strategy,
    data=df,  # 使用全部數據
    param_grid={
        'fast_period': [5, 10, 15, 20],
        'slow_period': [20, 30, 40, 50],
        'atr_period': [10, 14, 20]
    },
    train_window=1000,
    test_window=250,
    step=250,
    objective='sharpe_ratio',
    n_jobs=-1
)

wf_results = wf_analyzer.run()
wf_save_path = wf_analyzer.save_results(data_source="bybit_btc_1h_WF")

# 可視化Walk-Forward結果
plot_walkforward_performance(
    results_df=wf_results,
    save_path=wf_save_path / "walkforward_plot.png"
)

print(f"WF分析 - OOS/IS夏普比率: {wf_analyzer.combined_metrics['is_oos_sharpe_ratio']:.3f}")
print(f"WF分析 - 一致性: {wf_analyzer.combined_metrics['consistency']*100:.2f}%")

# ============================================================
# 步驟6: 樣本外測試（Out-of-Sample）
# ============================================================
print("\n步驟6: 執行樣本外測試...")

# 使用最佳參數在測試集上測試
oos_signals = my_strategy(test_data, **best_params)

oos_engine = BacktestEngine(
    data=test_data,
    signals=oos_signals,
    transaction_cost=0.0006,
    slippage=0.0001
)

oos_results = oos_engine.run()
oos_engine.print_summary()
oos_save_path = oos_engine.save_results(data_source="bybit_btc_1h_OOS")

# 可視化樣本外結果
plot_backtest_results(
    equity_curve=oos_results['equity_curve'],
    data=test_data,
    signals=oos_signals,
    title="樣本外測試結果",
    save_path=oos_save_path / "oos_backtest_plot.png"
)

# ============================================================
# 步驟7: 最終報告
# ============================================================
print("\n" + "="*70)
print("最終報告")
print("="*70)
print(f"1. 優化結果 (In-Sample):")
print(f"   - 最佳參數: {best_params}")
print(f"   - IS夏普比率: {opt_results.iloc[0]['sharpe_ratio']:.3f}")
print()
print(f"2. MCPT測試:")
print(f"   - p值: {mcpt_results['p_value']:.4f}")
print(f"   - 顯著性: {'通過' if mcpt_results['is_significant'] else '未通過'}")
print()
print(f"3. Walk-Forward分析:")
print(f"   - 平均OOS夏普: {wf_analyzer.combined_metrics['avg_oos_sharpe']:.3f}")
print(f"   - OOS/IS比率: {wf_analyzer.combined_metrics['is_oos_sharpe_ratio']:.3f}")
print(f"   - 一致性: {wf_analyzer.combined_metrics['consistency']*100:.2f}%")
print()
print(f"4. 樣本外測試:")
print(f"   - OOS夏普比率: {oos_results['metrics']['sharpe_ratio']:.3f}")
print(f"   - OOS總收益率: {oos_results['metrics']['total_return']*100:.2f}%")
print(f"   - OOS最大回撤: {oos_results['metrics']['max_drawdown']*100:.2f}%")
print("="*70)

# 判斷策略是否可以實盤
can_deploy = (
    mcpt_results['is_significant'] and
    wf_analyzer.combined_metrics['is_oos_sharpe_ratio'] > 0.7 and
    wf_analyzer.combined_metrics['consistency'] > 0.6 and
    oos_results['metrics']['sharpe_ratio'] > 1.0
)

if can_deploy:
    print("\n結論: 策略通過所有測試，可以考慮實盤部署！")
else:
    print("\n結論: 策略未通過所有測試，需要進一步優化或重新設計。")

print("\n所有結果已保存到results/目錄")
```

---

## 進階技巧

### 1. 自定義優化目標

```python
def custom_objective(metrics):
    """自定義優化目標：夏普比率/最大回撤"""
    if metrics['max_drawdown'] != 0:
        return metrics['sharpe_ratio'] / abs(metrics['max_drawdown'])
    return 0

optimizer = ParameterOptimizer(
    strategy_func=my_strategy,
    data=df,
    param_grid=param_grid,
    objective=custom_objective,  # 使用自定義函數
    n_jobs=-1
)
```

### 2. 並行處理控制

```python
# 單進程（用於調試）
optimizer = ParameterOptimizer(..., n_jobs=1)

# 使用4個進程
optimizer = ParameterOptimizer(..., n_jobs=4)

# 使用所有CPU核心
optimizer = ParameterOptimizer(..., n_jobs=-1)
```

### 3. 結果保存位置自定義

```python
from pathlib import Path

custom_dir = Path("my_results/strategy_v1")
save_path = engine.save_results(
    data_source="my_strategy",
    output_dir=custom_dir
)
```

---

## 常見問題

### Q1: 如何處理不同時間週期的數據？

調整`periods_per_year`參數：
- 1小時K線：8760 (24 * 365)
- 日K線：252 (交易日)
- 15分鐘K線：35040 (96 * 365)

### Q2: MCPT測試需要多少次排列？

- 快速測試：100-500次
- 正式測試：1000-2000次
- 嚴格測試：5000+次

### Q3: Walk-Forward窗口如何設置？

一般建議：
- 訓練窗口：測試窗口 = 3:1 到 5:1
- 測試窗口至少包含50-100筆交易
- 總共至少5-10個窗口

### Q4: 如何判斷策略是否過度擬合？

檢查以下指標：
1. MCPT p值 < 0.05
2. OOS/IS夏普比率 > 0.7
3. Walk-Forward一致性 > 60%
4. 樣本外夏普比率 > 1.0

---

## 參考資料

1. White, H. (2000). "A Reality Check for Data Snooping." Econometrica, 68(5), 1097-1126.
2. Prado, M. L. (2018). "Advances in Financial Machine Learning." Wiley.
3. Bailey, D. H., et al. (2014). "Pseudo-Mathematics and Financial Charlatanism." AMS Notices.

---

## 技術支持

如有問題，請查閱各模塊的文檔字符串，或參考test代碼：
- `backtest.py` 的 `if __name__ == "__main__"` 部分
- `optimizer.py` 的測試代碼
- 等等

祝您交易順利！
