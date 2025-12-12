"""
決策樹組合因子

用決策樹組合多個基礎因子 (MA交叉、RSI等)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import pickle
from typing import Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from strategies.ma_cross.strategy import MACrossStrategy


class TreeCombinedFactor:
    """決策樹組合因子 - 組合多個基礎因子"""

    factor_type = 'time_series'  # 時間序列因子

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.base_strategies = {}

    def train(self, data: pd.DataFrame):
        """
        訓練決策樹因子

        Args:
            data: OHLCV DataFrame (columns: open, high, low, close, volume)
        """
        print(f"訓練決策樹組合因子 (max_depth: {self.max_depth})")

        # 步驟1: 計算基礎因子值
        print("  計算基礎因子...")
        base_factors = self._calculate_base_factors(data)
        print(f"    基礎因子形狀: {base_factors.shape}")

        # 步驟2: 構建訓練標籤 (未來N期收益)
        print("  構建訓練標籤...")
        future_return = data['close'].pct_change(5).shift(-5)
        labels = (future_return > 0.02).astype(int)  # 正收益為1,否則為0

        # 移除NaN行
        mask = base_factors.notna().all(axis=1) & labels.notna()
        X_train = base_factors[mask].values
        y_train = labels[mask].values

        print(f"    訓練樣本: {len(X_train)}")

        # 步驟3: 訓練決策樹
        print("  訓練決策樹...")
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        self.model.fit(X_train, y_train)

        # 步驟4: 計算特徵重要性
        importance = pd.DataFrame(
            self.model.feature_importances_,
            index=self.feature_names,
            columns=['importance']
        ).sort_values('importance', ascending=False)

        print("\n  特徵重要性:")
        for idx, imp in importance.iterrows():
            print(f"    {idx}: {imp['importance']:.4f}")

        # 保存
        self._save()

        print(f"\n✅ 訓練完成!")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        計算決策樹因子值 (推理)

        Args:
            data: OHLCV DataFrame

        Returns:
            因子值Series (已shift(1)防止lookahead)
        """
        if self.model is None:
            self.load()

        # 計算基礎因子
        base_factors = self._calculate_base_factors(data)

        # 決策樹預測 (概率)
        try:
            probabilities = self.model.predict_proba(base_factors.fillna(0))[:, 1]
        except:
            # 如果有問題,返回全0
            probabilities = np.zeros(len(data))

        # 轉換為因子值 (-0.5 ~ 0.5)
        factor_values = probabilities - 0.5

        # shift(1) 防止 lookahead bias
        return pd.Series(factor_values, index=data.index).shift(1)

    def _calculate_base_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """計算基礎因子"""
        if not self.feature_names:
            # 初始化基礎策略
            self.base_strategies['ma_cross'] = MACrossStrategy()
            self.feature_names = ['ma_cross']

        factors_dict = {}

        # MA交叉因子
        ma_cross_signals = self.base_strategies['ma_cross'].generate_signals(
            data,
            short_window=20,
            long_window=50
        )
        factors_dict['ma_cross'] = ma_cross_signals.astype(float)

        # 可擴展: 添加更多基礎因子
        # factors_dict['rsi'] = ...
        # factors_dict['bbands'] = ...

        return pd.DataFrame(factors_dict, index=data.index)

    def _save(self):
        """保存到 strategies/tree_combined/models/"""
        save_dir = Path(__file__).parent / 'models'
        save_dir.mkdir(exist_ok=True)

        # 保存模型
        import joblib
        joblib.dump(self.model, save_dir / 'model.pkl')

        # 保存配置
        config = {
            'max_depth': self.max_depth,
            'feature_names': self.feature_names
        }
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f)

        print(f"  模型保存到: {save_dir}")

    def load(self):
        """從 strategies/tree_combined/models/ 加載"""
        load_dir = Path(__file__).parent / 'models'

        # 加載配置
        with open(load_dir / 'config.json', 'r') as f:
            config = json.load(f)

        self.max_depth = config['max_depth']
        self.feature_names = config['feature_names']

        # 加載模型
        import joblib
        self.model = joblib.load(load_dir / 'model.pkl')

        print(f"✅ 加載決策樹模型: {load_dir}")
