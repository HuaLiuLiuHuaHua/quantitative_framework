# Donchian Strategy - 完整測試套件

## 策略概述

Donchian Channel Breakout Strategy（唐奇安通道突破策略）是最早的趨勢跟蹤系統之一，由Richard Donchian在1960年代開發。

### 策略邏輯

- **上軌（Upper Band）**: N日最高價
- **下軌（Lower Band）**: N日最低價
- **做多信號**: 當前收盤價突破上軌
- **做空信號**: 當前收盤價跌破下軌

### MCPT-Main嚴格邏輯（避免前視偏差）

```python
# 使用lookback-1確保不包含當前K棒
upper = df['close'].rolling(lookback - 1).max().shift(1)
lower = df['close'].rolling(lookback - 1).min().shift(1)
```

## 文件結構

```
donchian/
├── strategy.py                  # Donchian策略類
├── test_backtest.py            # 1. 基礎回測
├── test_optimization.py        # 2. 參數優化
├── test_mcpt.py                # 3. MCPT驗證
├── test_walkforward.py         # 4. Walk-Forward分析
├── test_walkforward_mcpt.py    # 5. WF-MCPT終極驗證
├── results/                    # 測試結果輸出目錄
└── README.md                   # 本文件
```

## 使用方法

### 環境要求

確保已安裝所有依賴：
```bash
pip install pandas numpy matplotlib seaborn scipy tqdm
```

### 測試順序（建議按順序執行）

#### 1. 基礎回測 (test_backtest.py)

**目的**: 驗證策略基本功能

```bash
cd quantitative_framework/strategies/donchian
python test_backtest.py
```

**輸出**:
- 回測績效報告
- 權益曲線圖
- results/BTC_1h_{date}/backtest_summary.json

**參數配置**:
```python
LOOKBACK = 20                    # Donchian通道回溯期
START_DATE = "2024-01-01"        # 開始日期
END_DATE = "2024-12-31"          # 結束日期
TRANSACTION_COST = 0.001         # 交易成本 0.1%
SLIPPAGE = 0.0005                # 滑價 0.05%
```

---

#### 2. 參數優化 (test_optimization.py)

**目的**: 尋找最佳lookback參數

```bash
python test_optimization.py
```

**輸出**:
- 優化結果表（CSV）
- 參數敏感性圖表
- results/{date}/optimization/best_params.json

**參數配置**:
```python
LOOKBACK_RANGE = (10, 100, 5)    # lookback從10到100，步長5
OBJECTIVE = 'profit_factor'      # 優化目標
N_JOBS = 4                       # 並行核心數
```

**優化目標選項**:
- `sharpe_ratio`: 夏普比率（風險調整後收益）
- `profit_factor`: 獲利因子（總獲利/總虧損）
- `calmar_ratio`: 卡瑪比率（年化收益/最大回撤）
- `total_return`: 總收益率

---

#### 3. MCPT驗證 (test_mcpt.py)

**目的**: 驗證策略是否存在過度擬合

```bash
python test_mcpt.py
```

**輸出**:
- p-value（統計顯著性）
- MCPT分佈圖
- results/{date}/mcpt/mcpt_summary.json

**參數配置**:
```python
LOOKBACK = 20                    # 使用從優化獲得的最佳參數
N_PERMUTATIONS = 1000            # 排列次數（建議1000+）
MODE = 'normal'                  # 測試模式
METRIC = 'sharpe_ratio'          # 評估指標
```

**結果解讀**:
- **p < 0.05**: 策略顯著優於隨機，可能具有真實預測能力
- **p >= 0.05**: 策略未顯著優於隨機，可能存在過度擬合

---

#### 4. Walk-Forward分析 (test_walkforward.py)

**目的**: 模擬實盤滾動優化，評估樣本外績效

```bash
python test_walkforward.py
```

**輸出**:
- 樣本外績效報告
- IS vs OOS對比圖
- results/{date}/walkforward/walkforward_results.csv

**參數配置**:
```python
TRAIN_WINDOW = 365               # 訓練窗口: 365根K棒（約15天）
TEST_WINDOW = 30                 # 測試窗口: 30根K棒（約1.25天）
STEP = 30                        # 滾動步長: 30根K棒
```

**關鍵指標**:
- **平均OOS夏普**: 樣本外平均夏普比率
- **OOS/IS夏普比率**: 接近1表示沒有過度擬合
- **一致性**: 正收益窗口比例

---

#### 5. Walk-Forward MCPT終極驗證 (test_walkforward_mcpt.py)

**目的**: 結合WF和MCPT的最嚴格測試

```bash
python test_walkforward_mcpt.py
```

**警告**: 這個測試非常耗時（可能數小時），建議在配置較好的機器上運行。

**輸出**:
- Walk-Forward MCPT p-value
- 終極驗證報告
- results/{date}/mcpt/walkforward_mcpt_summary.json

**參數配置**:
```python
N_PERMUTATIONS = 200             # 排列次數（較少因為每次要優化）
LOOKBACK_RANGE = (10, 100, 10)   # 減少參數空間以加快測試
```

**終極結論標準**:
- **顯著且OOS/IS>0.8**: 策略可能具有真實預測能力
- **顯著但OOS/IS<0.5**: 可能存在一定程度的過度擬合
- **不顯著**: 策略可能主要依賴隨機性

---

## 數據來源

所有測試都使用Bybit BTC/USDT 1小時K線數據：

- **自動抓取**: 如果data/目錄沒有數據，自動從Bybit API抓取
- **自動緩存**: 抓取的數據會保存到data/目錄供後續使用
- **日期範圍**: 默認2024-01-01至2024-12-31（可配置）

## 結果文件結構

```
results/
└── BTC_1h_{timestamp}/
    ├── backtest_summary.json       # 回測摘要
    ├── equity_curve.csv            # 權益曲線
    ├── backtest_chart.png          # 回測圖表
    ├── optimization/
    │   ├── optimization_summary.json
    │   ├── optimization_results.csv
    │   ├── best_params.json
    │   └── optimization_chart.png
    ├── mcpt/
    │   ├── mcpt_summary.json
    │   ├── permutation_scores.csv
    │   └── mcpt_distribution.png
    └── walkforward/
        ├── walkforward_summary.json
        ├── walkforward_results.csv
        ├── window_params.json
        └── walkforward_chart.png
```

## 核心類和方法

### DonchianStrategy

```python
from strategies.donchian.strategy import DonchianStrategy

# 創建策略實例
strategy = DonchianStrategy()

# 生成交易信號
signals = strategy.generate_signals(df, lookback=20)

# 獲取默認參數
params = strategy.get_default_parameters()  # {'lookback': 20}

# 獲取參數網格
grid = strategy.get_parameter_grid()  # {'lookback': (10, 100, 5)}
```

### 信號說明

- **1**: 做多（突破上軌）
- **-1**: 做空（跌破下軌）
- **0**: 空倉（區間內）

## 性能指標說明

### 基礎指標

- **總收益率**: 策略總收益百分比
- **年化收益率**: 按年計算的收益率
- **最大回撤**: 從峰值到谷底的最大跌幅

### 風險調整指標

- **夏普比率**: (年化收益 - 無風險利率) / 年化波動率
  - > 1: 良好
  - > 2: 優秀
  - > 3: 卓越

- **卡瑪比率**: 年化收益 / 最大回撤
  - > 0.5: 良好
  - > 1: 優秀

### 交易指標

- **勝率**: 獲利交易 / 總交易
- **獲利因子**: 總獲利 / 總虧損
  - > 1: 盈利
  - > 1.5: 良好
  - > 2: 優秀

- **平均獲利/虧損**: 單筆交易的平均損益

## 常見問題

### Q: 為什麼要按順序執行測試？

A:
1. **test_backtest.py**: 驗證策略基本功能是否正常
2. **test_optimization.py**: 找到最佳參數
3. **test_mcpt.py**: 使用最佳參數驗證顯著性
4. **test_walkforward.py**: 驗證參數在樣本外的穩定性
5. **test_walkforward_mcpt.py**: 終極驗證

### Q: 如何選擇優化目標？

A:
- **追求穩定收益**: 使用`sharpe_ratio`（夏普比率）
- **追求高獲利**: 使用`profit_factor`（獲利因子）
- **控制回撤**: 使用`calmar_ratio`（卡瑪比率）

### Q: MCPT測試需要多長時間？

A:
- **Normal MCPT**: 1000次排列約5-15分鐘（取決於CPU）
- **Walk-Forward MCPT**: 200次排列可能需要1-4小時

### Q: 如何加快Walk-Forward MCPT測試？

A:
1. 減少`N_PERMUTATIONS`（例如100）
2. 增大`LOOKBACK_RANGE`的步長（例如(10, 100, 20)）
3. 增加`N_JOBS`使用更多CPU核心
4. 減小訓練窗口或測試窗口

### Q: 數據不足怎麼辦？

A:
- 確保數據範圍至少涵蓋: `TRAIN_WINDOW + TEST_WINDOW + (n_windows * STEP)`
- 建議至少1000根K棒以上
- 可以調整窗口大小或抓取更長時間範圍的數據

## 技術細節

### 前視偏差避免

所有測試都使用MCPT-Main的嚴格邏輯：

```python
# 錯誤做法（前視偏差）
upper = df['close'].rolling(lookback).max()  # 包含當前K棒！

# 正確做法（無前視偏差）
upper = df['close'].rolling(lookback - 1).max().shift(1)
```

### 交易成本模型

```python
total_cost = (transaction_cost + slippage) * |position_change|
net_return = strategy_return - total_cost
```

默認成本:
- **transaction_cost**: 0.1% (交易所手續費)
- **slippage**: 0.05% (滑價損失)
- **總成本**: 0.15% 每次交易

## 參考資料

### 策略理論

- Donchian, R. (1960s). "Donchian's 4-Week Rule"
- Turtle Trading System

### 驗證方法

- White, H. (2000). "A Reality Check for Data Snooping." Econometrica
- Pardo, R. (2008). "The Evaluation and Optimization of Trading Strategies"

## 作者與版權

- **策略實現**: Based on MCPT-Main methodology
- **框架**: quantitative_framework
- **數據源**: Bybit API

## 更新日誌

### v1.0.0 (2025-01-01)
- 初始版本
- 完整的6個測試文件
- MCPT-Main嚴格邏輯
- 支持並行計算
- 自動結果保存和可視化

---

## 快速開始示例

```bash
# 1. 基礎回測（驗證策略）
python test_backtest.py

# 2. 參數優化（找最佳參數）
python test_optimization.py

# 3. MCPT驗證（檢查過擬合）
# 修改test_mcpt.py中的LOOKBACK為優化獲得的最佳值
python test_mcpt.py

# 4. Walk-Forward分析（樣本外測試）
python test_walkforward.py

# 5. 終極驗證（可選，耗時較長）
python test_walkforward_mcpt.py
```

祝您回測順利！
