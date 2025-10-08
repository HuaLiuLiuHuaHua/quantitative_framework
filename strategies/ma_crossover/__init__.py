"""
MA Crossover Strategy
移動平均線交叉策略

策略說明：
- 短期均線上穿長期均線時做多
- 短期均線下穿長期均線時做空
- 使用嚴格的前視偏差控制
"""

from .strategy import MACrossoverStrategy

__all__ = ['MACrossoverStrategy']
