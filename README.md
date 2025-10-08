# Quantitative Trading Framework
# 量化交易框架

一個專業的量化交易策略研發框架，整合MCPT-Main、Medium和trading_framework的精華，專注於避免前視偏差、嚴格回測和穩健優化。

---

## 核心特性

### 1. 嚴格的前視偏差控制
- 所有技術指標使用 `shift(1)` 確保僅使用歷史數據
- 參照MCPT-Main的Bar Permutation原理
- 信號在收盤時產生，下一根K棒開盤執行

### 2. 完整的回測引擎
- 精確的交易成本計算（手續費、滑價）
- 全面的績效指標（夏普比率、最大回撤、盈虧比等）
- 自動生成權益曲線和回測報告

### 3. 高效的參數優化
- Medium的並行參數網格搜索
- 多種優化目標（夏普比率、盈虧比、卡瑪比率等）
- 參數敏感性分析和可視化

### 4. 專業的統計檢驗
- MCPT-Main的Bar Permutation檢驗
- 蒙特卡洛排列測試
- Walk-Forward分析支持

### 5. 靈活的數據管理
- 自動數據抓取和驗證
- Parquet格式高效存儲
- 多時間週期支持（1小時、日線等）

### 6. 完善的可視化
- 權益曲線圖表
- 回撤分析圖
- 參數優化熱圖
- 月度收益表

---

## 快速開始

### 1. 安裝依賴

```bash
cd quantitative_framework
pip install -r requirements.txt
```

### 2. 抓取數據

```python
# 抓取BTC 1小時K線數據
from quantitative_framework.data_fetchers.bybit_btc_1h_fetcher import fetch_bybit_btc_1h_data

df = fetch_bybit_btc_1h_data(
    start_date="2024-01-01",
    end_date="2024-12-31",
    save=True,
    verbose=True
)
```

```python
# 抓取BTC 日K線數據
from quantitative_framework.data_fetchers.bybit_btc_1d_fetcher import fetch_bybit_btc_1d_data

df = fetch_bybit_btc_1d_data(
    start_date="2020-01-01",
    end_date="2024-12-31",
    save=True,
    verbose=True
)
```

### 3. 運行策略回測

```bash
# Donchian策略回測
cd strategies/donchian
python test_backtest.py

# MA交叉策略回測
cd strategies/ma_crossover
python test_backtest.py
```

### 4. 參數優化

```bash
# Donchian策略參數優化
cd strategies/donchian
python test_optimization.py

# MA交叉策略參數優化
cd strategies/ma_crossover
python test_optimization.py
```

---

## 目錄結構

```
quantitative_framework/
│
├── README.md                    # 項目主文檔（本文件）
├── CHANGELOG.md                 # 版本變更記錄
├── requirements.txt             # Python依賴包
│
├── data/                        # 數據存儲目錄
│   ├── BTCUSDT_1h_*.csv    # BTC 1小時K線數據
│   └── BTCUSDT_1d_*.csv    # BTC 日K線數據
│
├── data_fetchers/               # 數據抓取模塊
│   ├── __init__.py
│   ├── data_validator.py       # 數據驗證邏輯
│   ├── bybit_btc_1h_fetcher.py # BTC 1小時K線抓取器
│   └── bybit_btc_1d_fetcher.py # BTC 日K線抓取器
│
├── shared/                      # 共用工具模塊
│   ├── __init__.py
│   ├── backtest.py             # 回測引擎
│   ├── metrics.py              # 績效指標計算
│   ├── optimizer.py            # 參數優化器
│   ├── statistical_tests.py   # 統計檢驗（Bar Permutation）
│   ├── walk_forward.py         # Walk-Forward分析
│   └── visualization.py        # 可視化工具
│
└── strategies/                  # 策略目錄
    ├── donchian/               # Donchian策略
    │   ├── __init__.py
    │   ├── strategy.py         # 策略邏輯
    │   ├── test_backtest.py    # 回測測試
    │   ├── test_optimization.py # 優化測試
    │   ├── README.md           # 策略文檔
    │   └── results/            # 測試結果
    │
    └── ma_crossover/           # MA交叉策略
        ├── __init__.py
        ├── strategy.py         # 策略邏輯
        ├── test_backtest.py    # 回測測試
        ├── test_optimization.py # 優化測試
        ├── README.md           # 策略文檔
        └── results/            # 測試結果
```

---

## 使用指南

### 如何創建新數據源

通過複製修改現有的數據抓取器即可：

```bash
# 複製現有的數據抓取器
cp data_fetchers/bybit_btc_1h_fetcher.py data_fetchers/bybit_eth_1h_fetcher.py

# 修改以下內容：
# 1. symbol = "ETHUSDT"
# 2. 函數名：fetch_bybit_eth_1h_data()
# 3. 文件名：ETHUSDT_1h_{start}_{end}.csv
```

### 如何創建新策略

1. **創建策略目錄**：

```bash
mkdir strategies/your_strategy
```

2. **創建策略文件**：

```python
# strategies/your_strategy/strategy.py
import pandas as pd

class YourStrategy:
    def __init__(self, name: str = "YourStrategy"):
        self.name = name

    def generate_signals(self, df: pd.DataFrame, **params) -> pd.Series:
        """
        生成交易信號

        重要：必須避免前視偏差！
        - 所有技術指標使用shift(1)
        - 信號僅基於歷史數據
        """
        # 計算技術指標（使用shift(1)）
        indicator = df['close'].rolling(20).mean().shift(1)

        # 生成信號
        signals = pd.Series(0, index=df.index)
        signals[df['close'] > indicator] = 1
        signals[df['close'] < indicator] = -1

        return signals

    def get_default_parameters(self):
        return {'period': 20}
```

3. **創建測試文件**：

參照 `strategies/donchian/test_backtest.py` 和 `test_optimization.py` 的範例。

### 測試流程

1. **數據準備**: 使用data_fetchers抓取數據
2. **策略開發**: 實現策略邏輯，確保避免前視偏差
3. **基礎回測**: 使用固定參數驗證策略基本功能
4. **參數優化**: 使用網格搜索尋找最佳參數
5. **統計檢驗**: 使用Bar Permutation檢驗策略顯著性
6. **Walk-Forward**: 樣本外測試，驗證穩健性

---

## 核心設計

### 1. 避免前視偏差的方法

**問題**: 使用未來數據進行歷史決策會導致回測結果過於樂觀。

**解決方案**:

```python
# ❌ 錯誤：使用當前K棒的指標值
ma = df['close'].rolling(20).mean()
signals[df['close'] > ma] = 1  # 當前close和當前ma比較

# ✅ 正確：使用前一根K棒的指標值
ma = df['close'].rolling(20).mean().shift(1)
signals[df['close'] > ma] = 1  # 當前close和昨天的ma比較
```

**Donchian策略示例**:

```python
# MCPT-Main嚴格邏輯
upper = df['close'].rolling(lookback - 1).max().shift(1)
lower = df['close'].rolling(lookback - 1).min().shift(1)

# 當前收盤價突破昨日通道
signals[df['close'] > upper] = 1
signals[df['close'] < lower] = -1
```

### 2. MCPT-Main的Bar Permutation原理

**目的**: 檢驗策略收益是否顯著優於隨機交易。

**原理**:
1. 保持K棒順序不變
2. 隨機打亂交易信號的順序
3. 重複1000次，生成收益分佈
4. 計算原始策略在分佈中的p值

**使用方法**:

```python
from quantitative_framework.shared.statistical_tests import bar_permutation_test

p_value = bar_permutation_test(
    data=df,
    signals=signals,
    n_permutations=1000,
    transaction_cost=0.001
)

print(f"p-value: {p_value}")  # p < 0.05表示策略顯著
```

### 3. Medium的並行優化

**優勢**: 使用多核並行加速參數搜索。

**使用方法**:

```python
from quantitative_framework.shared.optimizer import ParameterOptimizer

param_grid = {
    'lookback': list(range(10, 100, 5))
}

optimizer = ParameterOptimizer(
    strategy_func=strategy.generate_signals,
    data=df,
    param_grid=param_grid,
    objective='sharpe_ratio',
    n_jobs=4  # 使用4核並行
)

results = optimizer.optimize(verbose=True)
best_params = optimizer.get_best_params()
```

### 4. Walk-Forward分析的重要性

**問題**: 參數優化可能過擬合歷史數據。

**解決方案**: Walk-Forward滾動窗口測試。

**原理**:
1. 將數據分為多個時間窗口
2. 在訓練窗口優化參數
3. 在測試窗口驗證績效
4. 滾動窗口，重複測試

**使用方法**:

```python
from quantitative_framework.shared.walk_forward import WalkForwardAnalyzer

analyzer = WalkForwardAnalyzer(
    strategy_func=strategy.generate_signals,
    data=df,
    param_grid=param_grid,
    train_size=0.7,
    test_size=0.3,
    n_splits=5
)

results = analyzer.run()
analyzer.plot_results()
```

---

## 示例：Donchian策略完整工作流程

### 1. 數據準備

```python
from quantitative_framework.data_fetchers.bybit_btc_1h_fetcher import fetch_bybit_btc_1h_data

# 抓取2024年BTC 1小時數據
df = fetch_bybit_btc_1h_data(
    start_date="2024-01-01",
    end_date="2024-12-31",
    save=True
)
```

### 2. 策略回測

```python
from strategies.donchian.strategy import DonchianStrategy
from quantitative_framework.shared.backtest import BacktestEngine

# 創建策略並生成信號
strategy = DonchianStrategy()
signals = strategy.generate_signals(df, lookback=20)

# 執行回測
engine = BacktestEngine(
    data=df,
    signals=signals,
    transaction_cost=0.001,
    slippage=0.0005,
    initial_capital=100000
)

results = engine.run()
engine.print_summary()
```

### 3. 參數優化

```python
from quantitative_framework.shared.optimizer import ParameterOptimizer

param_grid = {
    'lookback': list(range(10, 100, 5))
}

optimizer = ParameterOptimizer(
    strategy_func=strategy.generate_signals,
    data=df,
    param_grid=param_grid,
    objective='profit_factor',
    n_jobs=4
)

results_df = optimizer.optimize(verbose=True)
best_params = optimizer.get_best_params()
print(f"最佳lookback: {best_params['lookback']}")
```

### 4. 統計檢驗

```python
from quantitative_framework.shared.statistical_tests import bar_permutation_test

# 使用最佳參數重新生成信號
signals = strategy.generate_signals(df, **best_params)

# Bar Permutation檢驗
p_value = bar_permutation_test(
    data=df,
    signals=signals,
    n_permutations=1000,
    transaction_cost=0.001
)

print(f"p-value: {p_value:.4f}")
if p_value < 0.05:
    print("策略顯著優於隨機交易！")
else:
    print("策略可能是運氣所致，需要重新評估。")
```

### 5. Walk-Forward分析

```python
from quantitative_framework.shared.walk_forward import WalkForwardAnalyzer

analyzer = WalkForwardAnalyzer(
    strategy_func=strategy.generate_signals,
    data=df,
    param_grid=param_grid,
    train_size=0.7,
    test_size=0.3,
    n_splits=5
)

wf_results = analyzer.run()
analyzer.plot_results(save_path='donchian_walkforward.png')
```

---

## 技術細節

### 交易成本處理

框架精確模擬實盤交易成本：

```python
# 手續費（雙邊）
transaction_cost = 0.001  # 0.1% = 開倉0.05% + 平倉0.05%

# 滑價
slippage = 0.0005  # 0.05% = 每次交易的滑點成本

# 實際成本
total_cost = (transaction_cost + slippage) * abs(position_change)
```

### 績效指標計算

```python
from quantitative_framework.shared.metrics import calculate_metrics

metrics = calculate_metrics(
    equity_curve=equity_curve,
    returns=returns,
    periods_per_year=8760  # 1小時K線
)

# 可用指標：
# - total_return: 總收益率
# - sharpe_ratio: 夏普比率
# - max_drawdown: 最大回撤
# - profit_factor: 盈虧比
# - win_rate: 勝率
# - total_trades: 總交易次數
```

### 並行計算優化

```python
# 使用joblib進行並行參數優化
from joblib import Parallel, delayed

def test_params(params):
    signals = strategy.generate_signals(df, **params)
    engine = BacktestEngine(df, signals)
    results = engine.run()
    return results['sharpe_ratio']

# 並行執行
results = Parallel(n_jobs=4)(
    delayed(test_params)(p) for p in param_combinations
)
```

---

## 常見問題

### Q1: 為什麼要使用shift(1)？

**A**: 避免前視偏差。在實盤交易中，你只能在K棒收盤後獲取該K棒的數據。使用shift(1)確保策略僅使用前一根K棒的指標值進行決策。

### Q2: 如何選擇優化目標？

**A**:
- **Sharpe Ratio**: 風險調整收益，適合追求穩定的策略
- **Profit Factor**: 盈虧比，適合追求高盈利的策略
- **Calmar Ratio**: 收益/最大回撤，適合重視風險控制的策略

### Q3: Walk-Forward和普通回測的區別？

**A**:
- **普通回測**: 在全部歷史數據上優化和測試，容易過擬合
- **Walk-Forward**: 模擬實盤情況，在歷史數據上優化，在未來數據上測試，更能反映實盤表現

### Q4: Bar Permutation檢驗的意義？

**A**: 判斷策略收益是否來自真實的市場規律，還是僅僅是數據挖掘的運氣。p<0.05表示策略顯著優於隨機交易。

### Q5: 如何處理多時間週期？

**A**:
```python
# 高頻數據
df_1h = fetch_bybit_btc_1h_data(...)  # periods_per_year=8760

# 低頻數據
df_1d = fetch_bybit_btc_1d_data(...)  # periods_per_year=365
```

記得在BacktestEngine中設置正確的`periods_per_year`以計算年化指標。

---

## 參考文獻

1. **Pardo, R.** (2008). *The Evaluation and Optimization of Trading Strategies*. Wiley.
2. **Aronson, D.** (2006). *Evidence-Based Technical Analysis*. Wiley.
3. **Chan, E.** (2009). *Quantitative Trading: How to Build Your Own Algorithmic Trading Business*. Wiley.
4. **Lopez de Prado, M.** (2018). *Advances in Financial Machine Learning*. Wiley.

---

## 版本信息

**當前版本**: v1.0.0

查看 [CHANGELOG.md](CHANGELOG.md) 了解詳細的版本變更記錄。

---

## 貢獻指南

歡迎貢獻代碼、報告問題或提出改進建議！

### 貢獻方式

1. Fork本項目
2. 創建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟Pull Request

---

## 免責聲明

⚠️ **重要聲明**:

本框架僅供學習和研究使用。任何基於本框架的交易決策和資金損失由使用者自行承擔。

- 歷史績效不代表未來收益
- 回測結果與實盤表現可能存在顯著差異
- 量化交易存在高風險，請謹慎投資
- 使用前請充分了解相關風險

---

## 許可證

本項目採用 MIT 許可證。詳見 LICENSE 文件。

---

## 聯繫方式

如有問題或建議，歡迎聯繫！

**Happy Trading!** 🚀
