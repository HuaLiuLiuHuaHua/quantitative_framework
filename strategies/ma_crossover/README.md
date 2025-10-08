# MA Crossover Strategy - 移動平均線交叉策略

## 策略概述

移動平均線交叉策略（MA Crossover Strategy）是最經典的趨勢跟蹤策略之一。該策略通過短期移動平均線和長期移動平均線的交叉來識別趨勢轉折點。

### 核心原理

- **金叉（Golden Cross）**: 短期均線從下方上穿長期均線 → 做多信號
- **死叉（Death Cross）**: 短期均線從上方下穿長期均線 → 做空信號

### 理論基礎

移動平均線能夠平滑價格波動，反映市場趨勢：
- **短期均線**：反映近期價格動向，對價格變化更敏感
- **長期均線**：代表整體趨勢，更加穩定
- **交叉信號**：當短期趨勢突破長期趨勢時，可能預示著趨勢轉折

---

## 參數說明

| 參數 | 說明 | 默認值 | 優化範圍 |
|------|------|--------|----------|
| `short_period` | 短期均線週期（K棒數量） | 10 | 5-30 |
| `long_period` | 長期均線週期（K棒數量） | 30 | 20-100 |

---

## 前視偏差控制

本策略使用**嚴格的前視偏差控制邏輯**，確保所有計算僅使用歷史數據：

```python
# 計算移動平均線（使用shift(1)避免前視偏差）
short_ma = df['close'].rolling(window=short_period).mean().shift(1)
long_ma = df['close'].rolling(window=long_period).mean().shift(1)

# 計算前一根K棒的均線位置（用於判斷交叉）
short_ma_prev = short_ma.shift(1)
long_ma_prev = long_ma.shift(1)

# 金叉：短期均線從下方上穿長期均線
golden_cross = (short_ma_prev <= long_ma_prev) & (short_ma > long_ma)

# 死叉：短期均線從上方下穿長期均線
death_cross = (short_ma_prev >= long_ma_prev) & (short_ma < long_ma)
```

### 關鍵設計點

1. **shift(1)**: 確保使用前一根K棒完成時的均線值
2. **交叉判斷**: 比較前一根和當前K棒的均線位置關係
3. **信號產生**: 在收盤時產生信號，下一根K棒開盤執行

---

## 使用方法

### 1. 基礎回測（固定參數）

運行 `test_backtest.py` 使用默認參數（short=10, long=30）進行回測：

```bash
cd strategies/ma_crossover
python test_backtest.py
```

**輸出結果**：
- 績效指標（總收益、夏普比率、最大回撤等）
- 權益曲線圖表
- 詳細回測報告（保存在 `results/` 目錄）

### 2. 參數優化

運行 `test_optimization.py` 進行參數優化：

```bash
cd strategies/ma_crossover
python test_optimization.py
```

**優化配置**：
- short_period: 5 到 30，步長 5
- long_period: 20 到 100，步長 10
- 優化目標: profit_factor（可改為 sharpe_ratio, calmar_ratio）

**輸出結果**：
- 所有參數組合的績效表
- 最佳參數組合
- 參數敏感性分析圖表

---

## 策略特性

### 優勢

1. **邏輯簡單**: 易於理解和實現
2. **趨勢跟蹤**: 能夠捕捉中長期趨勢
3. **風險控制**: 自動止損（反向交叉時平倉）
4. **適應性強**: 可應用於各種市場和時間週期

### 劣勢

1. **震盪市場**: 在橫盤震盪時會產生頻繁假信號
2. **滯後性**: 均線本質上是滯後指標，可能錯過部分趨勢初期
3. **參數敏感**: 不同市場和週期需要不同的參數設置
4. **交易成本**: 頻繁交易可能導致較高的手續費

### 適用場景

- **趨勢明顯的市場**: 牛市或熊市
- **中長期交易**: 日線或更長週期
- **波動率較大的標的**: 如加密貨幣、商品

---

## 績效評估

### 關鍵指標

- **Total Return**: 總收益率
- **Sharpe Ratio**: 夏普比率（風險調整收益）
- **Max Drawdown**: 最大回撤
- **Profit Factor**: 盈虧比（總盈利/總虧損）
- **Win Rate**: 勝率
- **Total Trades**: 總交易次數

### 參數選擇建議

1. **短期均線（short_period）**:
   - 5-10: 對價格變化非常敏感，適合短線交易
   - 10-20: 平衡靈敏度和穩定性
   - 20-30: 更平滑，減少假信號

2. **長期均線（long_period）**:
   - 20-50: 適合中期趨勢
   - 50-100: 適合長期趨勢
   - 100+: 極長期趨勢，交易次數少

3. **週期比例**:
   - 建議 long_period / short_period ≥ 2
   - 常用組合: (10, 30), (5, 20), (20, 60)

---

## 程式碼示例

### 直接調用策略

```python
from strategies.ma_crossover.strategy import MACrossoverStrategy
import pandas as pd

# 載入數據
df = pd.read_parquet('data/BTCUSDT_1h.parquet')

# 創建策略實例
strategy = MACrossoverStrategy()

# 生成交易信號
signals = strategy.generate_signals(df, short_period=10, long_period=30)

# 查看信號統計
print(f"做多信號: {(signals == 1).sum()}")
print(f"做空信號: {(signals == -1).sum()}")
```

### 結合回測引擎

```python
from quantitative_framework.shared.backtest import BacktestEngine

# 執行回測
engine = BacktestEngine(
    data=df,
    signals=signals,
    transaction_cost=0.001,  # 0.1%
    slippage=0.0005,         # 0.05%
    initial_capital=100000
)

results = engine.run()
engine.print_summary()
```

---

## 改進方向

### 1. 過濾條件

添加額外的條件來減少假信號：
- **成交量確認**: 交叉時成交量放大
- **趨勢強度**: 僅在ADX > 25時交易
- **價格突破**: 結合支撐/阻力位

### 2. 動態參數

根據市場狀態調整參數：
- **波動率自適應**: 高波動時縮短週期
- **趨勢強度自適應**: 強趨勢時延長週期

### 3. 風險管理

- **固定止損**: 設置百分比止損
- **追蹤止損**: 保護已有利潤
- **倉位管理**: 根據信號強度調整倉位

### 4. 多時間週期

- **高頻確認**: 短週期確認長週期信號
- **長週期過濾**: 僅在大趨勢方向交易

---

## 文件結構

```
ma_crossover/
├── __init__.py              # 策略包初始化
├── strategy.py              # 策略核心邏輯
├── test_backtest.py         # 基礎回測測試
├── test_optimization.py     # 參數優化測試
├── README.md                # 策略說明文檔（本文件）
└── results/                 # 測試結果輸出目錄
    ├── BTC_1h_20250101_120000/
    │   ├── backtest_results.json
    │   ├── backtest_chart.png
    │   └── equity_curve.csv
    └── optimization_20250101_120000/
        ├── optimization_results.csv
        └── optimization_chart.png
```

---

## 參考文獻

1. Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*
2. Pring, M. J. (2002). *Technical Analysis Explained*
3. Appel, G. (2005). *Technical Analysis: Power Tools for Active Investors*

---

## 注意事項

⚠️ **重要提醒**:

1. **歷史績效不代表未來收益**: 過去的回測結果不能保證未來表現
2. **參數過擬合風險**: 避免過度優化參數以適應歷史數據
3. **交易成本影響**: 實盤交易時需考慮手續費、滑價等成本
4. **市場環境變化**: 策略在不同市場環境下表現可能差異巨大
5. **風險控制**: 使用前務必進行充分的樣本外測試和風險評估

---

## 版本歷史

- **v1.0.0** (2025-01-01): 初始版本
  - 實現基礎MA交叉策略
  - 嚴格前視偏差控制
  - 完整回測和優化測試

---

## 聯繫方式

如有問題或建議，歡迎提出！
