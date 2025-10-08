"""
LSTM SOL Strategy
LSTM SOL 策略模組

這是一個基於技術指標的簡化版策略，源自原始的 LSTM Walk-Forward 策略。
由於缺少訓練好的 LSTM 模型，我們使用純技術指標組合替代機器學習預測。

策略特點：
- 使用 12 個技術指標：RSI, MACD, BB, OBV, ATR, SMA
- 混合式倉位管理 (Fixed Ratio, Risk-Based, Volatility-Based, Hybrid)
- 動態 ATR 止損/止盈
- 盈虧平衡止損和追蹤止損
- 趨勢和動量過濾
- 風險回報比過濾

適用數據：
- SOL/USDT 1小時 K線
- 回溯窗口：72小時
"""

__version__ = "1.0.0"
__author__ = "Quantitative Framework"
