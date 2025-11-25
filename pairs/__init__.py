"""
Kalman Filter Pairs Trading Factor

使用卡爾曼濾波器進行配對交易的因子模組
"""

from .factor import KalmanPairsFactor
from .position_strategy import KalmanPairsPositionStrategy

__all__ = ['KalmanPairsFactor', 'KalmanPairsPositionStrategy']
