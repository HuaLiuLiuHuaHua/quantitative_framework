# 量化交易框架 - 核心共享模塊

這是一個完整的量化交易研發框架，包含策略回測、參數優化、統計驗證和可視化功能。

## 模塊概覽

### 1. **base_strategy.py** - 策略基類
- 提供統一的策略接口
- 標準化的信號生成方法
- 便於策略擴展和管理

### 2. **metrics.py** - 績效指標計算器
- 13個核心績效指標
- 包含夏普比率、獲利因子、最大回撤等
- 支持自定義年化參數

### 3. **backtest.py** - 回測引擎 ⭐
- 完整的交易成本模擬（transaction_cost + slippage）
- 生成權益曲線和收益率序列
- 自動保存結果到`results/{data_source}_{date}/`
- 提供快速回測接口`quick_backtest()`

**核心類：** `BacktestEngine`

**主要功能：**
```python
engine = BacktestEngine(data, signals, transaction_cost=0.0006, slippage=0.0001)
results = engine.run()
engine.print_summary()
engine.save_results(data_source="bybit_btc_1h")
```

### 4. **permutation.py** - K線排列邏輯 ⭐
- 實作MCPT-Main的Bar Permutation（White 2000）
- 對數價格分解：跳空(r_o) + 日內(r_h, r_l, r_c)
- 保留OHLC內部關係
- 支援三種排列模式：both, intraday, gap

**核心函數：** `advanced_permutation()`

**理論基礎：**
```
log(P_t) = log(O_t) + [log(H_t) - log(O_t), log(L_t) - log(O_t), log(C_t) - log(O_t)]
         = 跳空組件 + 日內組件
```

### 5. **optimizer.py** - 並行參數優化器 ⭐
- 網格搜索（Grid Search）
- ProcessPoolExecutor並行計算
- 支持多種優化目標（sharpe_ratio, profit_factor, calmar_ratio等）
- 參數敏感性分析
- 自動保存完整結果

**核心類：** `ParameterOptimizer`

**主要功能：**
```python
optimizer = ParameterOptimizer(
    strategy_func=my_strategy,
    data=df,
    param_grid={'period': [10, 20, 30]},
    objective='sharpe_ratio',
    n_jobs=-1
)
results = optimizer.optimize()
best_params = optimizer.get_best_params()
```

### 6. **mcpt.py** - 蒙地卡羅排列測試器 ⭐
- 驗證策略統計顯著性
- 避免過度擬合（Data Snooping）
- 兩種模式：Normal MCPT 和 Walk-Forward MCPT
- 並行計算排列
- 計算p值和顯著性

**核心類：** `MonteCarloTester`

**測試原理：**
```
H0: 策略沒有預測能力
p-value = P(排列績效 >= 原始績效)
如果 p < 0.05，拒絕H0，策略有效
```

**主要功能：**
```python
tester = MonteCarloTester(
    strategy_func=my_strategy,
    data=df,
    strategy_params={'period': 20},
    n_permutations=1000,
    mode='normal'
)
results = tester.run()
```

### 7. **walkforward.py** - Walk-Forward分析器 ⭐
- 滾動窗口優化和樣本外測試
- 以K棒數量為單位（train_window, test_window, step）
- 每個窗口重新優化參數
- 計算IS vs OOS績效比較
- 評估策略穩定性

**核心類：** `WalkForwardAnalyzer`

**工作流程：**
```
Window 1: [Train: 0-1000] -> Optimize -> [Test: 1000-1250]
Window 2: [Train: 250-1250] -> Optimize -> [Test: 1250-1500]
Window 3: [Train: 500-1500] -> Optimize -> [Test: 1500-1750]
...
```

**主要功能：**
```python
analyzer = WalkForwardAnalyzer(
    strategy_func=my_strategy,
    data=df,
    param_grid={'period': [10, 20, 30]},
    train_window=1000,
    test_window=250,
    step=250
)
results = analyzer.run()
```

### 8. **visualization.py** - 繪圖功能模塊 ⭐
- Medium風格的專業圖表
- 支持matplotlib和seaborn
- 中文字體配置

**核心函數：**

#### a. `plot_backtest_results()`
- 權益曲線
- 價格和信號標記
- 回撤曲線

#### b. `plot_optimization_results()`
- 參數熱圖（2D參數空間）
- 參數敏感性分析
- Top N參數組合對比

#### c. `plot_mcpt_distribution()`
- 排列分佈直方圖
- p值和顯著性標記
- 原始績效位置
- 統計摘要

#### d. `plot_walkforward_performance()`
- 樣本外績效趨勢
- IS vs OOS散點圖
- 累積收益曲線
- 勝率和獲利因子

---

## 完整工作流程

```python
# 1. 載入數據
df = pd.read_csv('data.csv', index_col=0, parse_dates=True)

# 2. 定義策略
def my_strategy(data, period=20):
    ma = data['close'].rolling(period).mean()
    signals = pd.Series(0, index=data.index)
    signals[data['close'] > ma] = 1
    signals[data['close'] < ma] = -1
    return signals

# 3. 參數優化
optimizer = ParameterOptimizer(
    strategy_func=my_strategy,
    data=df,
    param_grid={'period': [10, 20, 30, 40, 50]},
    objective='sharpe_ratio'
)
results = optimizer.optimize()
best_params = optimizer.get_best_params()

# 4. MCPT測試（驗證過度擬合）
tester = MonteCarloTester(
    strategy_func=my_strategy,
    data=df,
    strategy_params=best_params,
    n_permutations=1000
)
mcpt_results = tester.run()

# 5. Walk-Forward分析
analyzer = WalkForwardAnalyzer(
    strategy_func=my_strategy,
    data=df,
    param_grid={'period': [10, 20, 30, 40, 50]},
    train_window=1000,
    test_window=250
)
wf_results = analyzer.run()

# 6. 樣本外測試
signals = my_strategy(test_data, **best_params)
engine = BacktestEngine(test_data, signals)
oos_results = engine.run()

# 7. 可視化
plot_backtest_results(oos_results['equity_curve'], test_data, signals)
plot_optimization_results(optimizer.results_df, ['period'])
plot_mcpt_distribution(mcpt_results['original_score'], mcpt_results['permutation_scores'], mcpt_results['p_value'])
plot_walkforward_performance(wf_results)
```

---

## 技術特點

### 1. 交易成本處理
所有模塊都考慮完整的交易成本：
- **Transaction Cost**: 固定比例（例如0.06%）
- **Slippage**: 滑價（例如0.01%）
- **總成本**: 每次倉位變化時扣除

### 2. 並行計算
使用`ProcessPoolExecutor`實現真正的並行：
- 參數優化：並行評估所有參數組合
- MCPT測試：並行計算所有排列
- Walk-Forward：並行優化每個窗口

### 3. 自動結果保存
所有分析器都包含`save_results()`方法：
```
results/
├── bybit_btc_1h_20231001_120000/
│   ├── backtest_summary.json
│   ├── equity_curve.csv
│   ├── optimization/
│   │   ├── optimization_summary.json
│   │   └── optimization_results.csv
│   ├── mcpt/
│   │   ├── mcpt_summary.json
│   │   └── permutation_scores.csv
│   └── walkforward/
│       ├── walkforward_summary.json
│       └── walkforward_results.csv
```

### 4. 完整文檔
- 每個函數都有詳細的中文文檔字符串
- 包含參數說明、返回值、使用範例
- 測試代碼在`if __name__ == "__main__"`中

### 5. Type Hints
所有函數都使用類型提示，便於IDE自動補全和類型檢查。

---

## 性能指標

### 回測速度
- 單次回測：<1秒（1000根K線）
- 快速回測：<0.1秒（用於優化循環）

### 優化速度
- 單參數優化（100組合）：~10秒（8核CPU）
- 雙參數優化（400組合）：~40秒（8核CPU）

### MCPT速度
- Normal MCPT（1000排列）：~5分鐘（8核CPU）
- Walk-Forward MCPT（500排列）：~30分鐘（8核CPU，含優化）

### Walk-Forward速度
- 10個窗口，5參數：~2分鐘（8核CPU）

---

## 依賴項

```
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scipy >= 1.7.0
tqdm >= 4.62.0
```

---

## 使用建議

### 1. 參數優化
- 先用粗網格快速搜索
- 再用細網格精確優化
- 注意參數合理性（例如fast < slow）

### 2. MCPT測試
- Normal模式：用於單一參數組合驗證
- Walk-Forward模式：更嚴格，但更耗時
- 至少1000次排列才有統計意義

### 3. Walk-Forward分析
- 訓練期 = 3-5倍測試期
- 至少5-10個窗口
- 檢查OOS/IS比率（>0.7為佳）
- 檢查一致性（>60%為佳）

### 4. 過度擬合檢測
綜合判斷以下指標：
- ✅ MCPT p值 < 0.05
- ✅ OOS/IS夏普比率 > 0.7
- ✅ Walk-Forward一致性 > 60%
- ✅ 參數敏感性低（微調參數績效不崩潰）

---

## 理論基礎

### MCPT原理（White 2000）
通過隨機排列歷史數據，生成多個"假"歷史序列，比較原始策略績效與排列績效的分佈，判斷策略是否具有統計顯著性。

**虛無假設H0**: 策略沒有預測能力，績效來自隨機性

**p值計算**: p = P(排列績效 >= 原始績效)

**拒絕域**: p < α (通常α = 0.05)

### Bar Permutation（MCPT-Main）
將對數價格分解為獨立的跳空和日內組件：
- **跳空組件**: r_o = log(open_t / close_t-1)
- **日內組件**: r_h, r_l, r_c（相對開盤價）

分別排列兩個組件，保留：
1. K線內部OHLC關係
2. 部分時間序列結構
3. 價格波動特性

比簡單排列更保守，生成更現實的序列。

---

## 測試與驗證

每個模塊都包含完整的測試代碼：

```bash
# 測試回測引擎
python quantitative_framework/shared/backtest.py

# 測試排列邏輯
python quantitative_framework/shared/permutation.py

# 測試參數優化
python quantitative_framework/shared/optimizer.py

# 測試MCPT
python quantitative_framework/shared/mcpt.py

# 測試Walk-Forward
python quantitative_framework/shared/walkforward.py

# 測試可視化
python quantitative_framework/shared/visualization.py
```

---

## 常見問題

### Q: 為什麼優化結果與回測不同？
A: 檢查：
1. 數據範圍是否一致
2. 交易成本設置是否一致
3. 信號是否正確對齊（注意shift）

### Q: MCPT測試太慢怎麼辦？
A:
1. 減少排列次數（但至少500次）
2. 使用Normal模式而非Walk-Forward模式
3. 增加`n_jobs`並行進程數
4. 減少數據量或使用`start_index`

### Q: Walk-Forward結果不理想？
A: 可能原因：
1. 參數空間太小（增加參數範圍）
2. 訓練窗口太小（增加train_window）
3. 市場環境變化太大（策略需要適應性）
4. 策略本身不穩定（重新設計）

### Q: 如何處理多時間週期？
A: 調整`periods_per_year`參數：
- 1H: 24 * 365 = 8760
- 4H: 6 * 365 = 2190
- 1D: 252（交易日）
- 15m: 96 * 365 = 35040

---

## 版本歷史

### v1.0.0 (2024-10-01)
- ✅ 完整實作6個核心模塊
- ✅ 並行計算支持
- ✅ 完整文檔和測試
- ✅ 自動結果保存
- ✅ 專業可視化

---

## 參考文獻

1. **White, H. (2000).** "A Reality Check for Data Snooping." *Econometrica*, 68(5), 1097-1126.
   - MCPT測試的理論基礎

2. **Prado, M. L. (2018).** "Advances in Financial Machine Learning." *Wiley*.
   - 第11章: 回測過度擬合問題
   - 第12章: 回測的危險

3. **Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2014).** "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance." *Notices of the AMS*, 61(5), 458-471.
   - 過度擬合的數學分析

4. **Harvey, C. R., Liu, Y., & Zhu, H. (2016).** "... and the Cross-Section of Expected Returns." *Review of Financial Studies*, 29(1), 5-68.
   - 多重假設檢驗問題

---

## 授權

本框架遵循MIT許可證。

---

## 貢獻

歡迎提交Issue和Pull Request！

---

## 聯繫方式

如有問題或建議，請查閱：
1. `USAGE_EXAMPLES.md` - 詳細使用範例
2. 各模塊的文檔字符串
3. 測試代碼

祝您量化交易順利！📈
