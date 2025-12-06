"""
XGBoost 因子組合 - 機器學習自動融合多個因子

用 XGBoost 學習如何：
1. 組合多個因子 (手動因子 + 深度學習因子)
2. 自動移除無效因子 (特徵重要性為 0)
3. 生成最強的複合因子
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import xgboost as xgb
import pandas as pd
import numpy as np
import json
import pickle
from typing import Dict, List

from strategies.lstm_feature.factor import LSTMFeatureFactor


class XGBoostCombinedFactor:
    """
    XGBoost 因子組合 - 自動融合與移除因子

    支持組合多個因子：
    - 手動設計的因子 (MA、RSI、MACD、布林帶等)
    - 深度學習挖掘的因子 (LSTM)
    - 任何其他自定義因子

    XGBoost 自動學習：
    - 每個因子的權重
    - 如何非線性組合因子
    - 自動忽略無效因子 (特徵重要性設為 0)
    """

    factor_type = 'ml_combined'

    def __init__(
        self,
        max_depth: int = 5,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        device: str = 'cpu'
    ):
        """
        初始化 XGBoost 因子組合

        Args:
            max_depth: XGBoost 樹的最大深度 (預設 5)
            n_estimators: 樹的數量 (預設 100)
            learning_rate: 學習率 (預設 0.1)
            device: 計算設備 ('cpu' 或 'gpu', 預設 'cpu')
        """
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.device = device

        # XGBoost 模型
        tree_method = 'gpu_hist' if device == 'gpu' else 'hist'
        self.model = xgb.XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            objective='binary:logistic',
            tree_method=tree_method,
            random_state=42,
            verbosity=0
        )

        # 特徵信息
        self.feature_names = None
        self.feature_importance = None
        self.is_trained = False

    def _calculate_base_factors(
        self,
        data: pd.DataFrame,
        use_lstm: bool = True
    ) -> pd.DataFrame:
        """
        計算基礎因子 (手動 + 深度學習)

        Args:
            data: OHLCV DataFrame
            use_lstm: 是否包含 LSTM 因子

        Returns:
            DataFrame，其中每一列是一個因子值
        """
        factors = {}

        # ===== 手動設計的技術指標因子 =====

        # 1. 移動平均線相關
        factors['ma_5_cross_20'] = self._ma_cross(data['close'], 5, 20)
        factors['ma_10_cross_50'] = self._ma_cross(data['close'], 10, 50)

        # 2. 相對強弱指數 (RSI)
        factors['rsi_14'] = self._rsi(data['close'], period=14)

        # 3. MACD
        macd_line, signal_line, histogram = self._macd(data['close'])
        factors['macd_histogram'] = histogram
        factors['macd_signal_cross'] = (macd_line - signal_line).rolling(2).mean()

        # 4. 布林帶
        factors['bb_signal'] = self._bollinger_bands(data['close'], window=20, num_std=2)

        # 5. 成交量相關
        factors['volume_ma_ratio'] = self._volume_ma_ratio(data['volume'], period=20)

        # 6. ATR (波動率)
        factors['atr_normalized'] = self._atr(data, period=14) / data['close']

        # ===== 深度學習因子 =====
        if use_lstm:
            try:
                lstm_factor = LSTMFeatureFactor()
                lstm_factor.load()
                factors['lstm_feature'] = lstm_factor.calculate(data)
            except FileNotFoundError:
                raise RuntimeError(
                    "LSTM 模型未找到。請先訓練 LSTM 因子:\n"
                    "  python strategies/lstm_feature/train.py"
                )

        # 轉換為 DataFrame
        factors_df = pd.DataFrame(factors, index=data.index)

        # 前向填充處理 NaN
        factors_df = factors_df.ffill().fillna(0)

        return factors_df

    def _ma_cross(self, close: pd.Series, fast: int, slow: int) -> pd.Series:
        """移動平均線交叉信號 (-1, 0, 1)"""
        ma_fast = close.rolling(fast).mean().shift(1)
        ma_slow = close.rolling(slow).mean().shift(1)
        # 1: 快線在慢線上方, -1: 快線在慢線下方, 0: 其他
        return np.sign(ma_fast - ma_slow)

    def _rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """相對強弱指數 (0-100, 標準化為 0-1)"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return (rsi / 100.0).shift(1)  # 標準化到 0-1, shift(1) 防止 lookahead bias

    def _macd(self, close: pd.Series) -> tuple:
        """MACD 指標"""
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line.shift(1), signal_line.shift(1), histogram.shift(1)

    def _bollinger_bands(
        self,
        close: pd.Series,
        window: int = 20,
        num_std: float = 2
    ) -> pd.Series:
        """布林帶：價格相對於中線的位置 (-1 到 1)"""
        sma = close.rolling(window).mean().shift(1)
        std = close.rolling(window).std().shift(1)
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        # 標準化到 -1 到 1
        bb_position = (close.shift(1) - sma) / (upper - lower + 1e-8) * 2
        return bb_position.clip(-1, 1)

    def _volume_ma_ratio(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """成交量相對於移動平均的比例"""
        volume_ma = volume.rolling(period).mean().shift(1)
        ratio = volume.shift(1) / (volume_ma + 1e-8)
        # 標準化
        return np.log(ratio + 1).clip(-2, 2) / 2

    def _atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """平均真實波幅"""
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().shift(1)

        return atr

    def train(
        self,
        data: pd.DataFrame,
        use_lstm: bool = True,
        future_return_period: int = 5,
        min_return_threshold: float = 0.02
    ):
        """
        訓練 XGBoost 因子組合

        Args:
            data: 訓練數據 DataFrame (OHLCV)
            use_lstm: 是否使用 LSTM 因子
            future_return_period: 未來收益計算期數 (預設 5)
            min_return_threshold: 定義"上漲"的最小收益率 (預設 2%)
        """
        # 輸入驗證
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"數據缺少必需列: {required_columns - set(data.columns)}")

        if len(data) < 100:
            raise ValueError(f"需要至少 100 根 K 線,實際: {len(data)}")

        print(f"訓練 XGBoost 因子組合")
        print(f"  設備: {self.device}")
        print(f"  樹深度: {self.max_depth}")
        print(f"  樹數量: {self.n_estimators}")

        # Step 1: 計算基礎因子
        print(f"\n計算基礎因子...")
        base_factors = self._calculate_base_factors(data, use_lstm=use_lstm)
        print(f"  因子數量: {len(base_factors.columns)}")
        print(f"  因子列表: {list(base_factors.columns)}")

        # Step 2: 構建標籤 (未來 future_return_period 期是否上漲)
        print(f"\n構建訓練標籤...")
        future_close = data['close'].shift(-future_return_period)
        future_return = (future_close - data['close']) / data['close']
        labels = (future_return > min_return_threshold).astype(int)

        # 移除 NaN
        valid_idx = ~(base_factors.isna().any(axis=1) | labels.isna())
        X = base_factors[valid_idx]
        y = labels[valid_idx]

        print(f"  樣本數: {len(X)}")
        print(f"  正樣本: {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
        print(f"  負樣本: {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")

        # Step 3: 訓練 XGBoost
        print(f"\n訓練 XGBoost...")
        self.model.fit(
            X, y,
            verbose=False,
            eval_metric='logloss'
        )

        # Step 4: 特徵重要性分析
        print(f"\n特徵重要性分析:")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(feature_importance.to_string(index=False))

        # 找出重要特徵 (重要性 > 0)
        important_features = feature_importance[feature_importance['importance'] > 0]
        print(f"\n[OK] 有效因子: {len(important_features)}/{len(feature_importance)}")

        for idx, row in important_features.iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")

        # Step 5: 保存模型和特徵信息
        self.feature_names = X.columns.tolist()
        self.feature_importance = feature_importance
        self.is_trained = True
        self._save()

    def calculate(self, data: pd.DataFrame, use_lstm: bool = True) -> pd.Series:
        """
        計算因子值 (推理)

        Args:
            data: OHLCV DataFrame

        Returns:
            因子值 Series (預測概率 0-1, 已 shift(1) 防止 lookahead bias)
        """
        if self.model is None or not self.is_trained:
            self.load()

        # 計算基礎因子
        base_factors = self._calculate_base_factors(data, use_lstm=use_lstm)

        # 只使用訓練時的特徵
        base_factors = base_factors[self.feature_names]

        # 向量化推理 (比逐個樣本快 100 倍)
        factor_values = self.model.predict_proba(base_factors)[:, 1]

        # 轉換為 Series
        result = pd.Series(factor_values, index=data.index)

        # shift(1) 防止 lookahead bias
        return result.shift(1)

    def _save(self):
        """保存模型到 models/ 目錄"""
        save_dir = Path(__file__).parent / 'models'
        save_dir.mkdir(exist_ok=True)

        # 保存模型
        self.model.save_model(str(save_dir / 'xgboost_combined.json'))

        # 保存特徵信息
        config = {
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance.to_dict('list') if self.feature_importance is not None else None,
            'max_depth': self.max_depth,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
        }

        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"  [OK] 模型已保存到 {save_dir}")

    def load(self):
        """從 models/ 目錄加載模型"""
        load_dir = Path(__file__).parent / 'models'

        # 加載配置
        with open(load_dir / 'config.json', 'r') as f:
            config = json.load(f)

        self.feature_names = config['feature_names']
        if config['feature_importance']:
            self.feature_importance = pd.DataFrame(config['feature_importance'])

        # 加載模型
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(load_dir / 'xgboost_combined.json'))

        print(f"[OK] 加載模型: {load_dir}")
        self.is_trained = True
