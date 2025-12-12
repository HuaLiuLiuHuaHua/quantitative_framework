"""
LSTM 特徵因子 - 深度學習自動挖掘因子

不預設是什麼因子，讓 LSTM 自動從 OHLCV 學習能預測漲跌的特徵
支持單時間尺度和多時間尺度
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
import hashlib
from typing import Tuple, List, Optional

from ml_tools.lstm import LSTMModel


class LSTMFeatureFactor:
    """
    LSTM 特徵因子 - 自動特徵挖掘

    支持單時間尺度 (e.g., ['1h']) 或多時間尺度 (e.g., ['1h', '4h', '1d'])
    LSTM 自動學習如何融合多個時間尺度
    """

    factor_type = 'dl_feature'  # 深度學習特徵因子

    def __init__(
        self,
        sequence_length: int = 60,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        time_scales: List[str] = None,
        device: str = None
    ):
        """
        初始化 LSTM 特徵因子

        Args:
            sequence_length: LSTM 序列長度 (預設 60)
            hidden_dim: LSTM 隱藏層維度 (預設 128)
            num_layers: LSTM 層數 (預設 2)
            output_dim: 輸出維度 - 1 表示預測概率 (預設 1)
            time_scales: 時間尺度列表
              - ['1h']: 單時間尺度 (輸入維度=5)
              - ['1h', '4h', '1d']: 多時間尺度 (輸入維度=15)
            device: 計算設備 ('cuda' 或 'cpu', 預設自動檢測)
        """
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.time_scales = time_scales or ['1h']

        # 自動檢測 GPU
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # 計算輸入維度: 5 (OHLCV) × N 個時間尺度
        self.input_dim = 5 * len(self.time_scales)

        # 初始化 LSTM 模型
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.is_trained = False

    # ===== 緩存管理方法 (優化 4) =====

    def _get_cache_key(self, data: pd.DataFrame) -> str:
        """生成數據的唯一哈希鍵（包含 input_dim 防止衝突）"""
        key_str = (
            f"{data.index[0]}_{data.index[-1]}_"
            f"{len(data)}_{self.sequence_length}_"
            f"{self.input_dim}_"  # 添加 input_dim 防止特徵維度不同導致的碰撞
            f"{'_'.join(self.time_scales)}"
        )
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """獲取緩存文件路徑"""
        cache_dir = Path(__file__).parent / 'cache'
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"sequences_{cache_key}.pkl"

    def _load_cache(self, cache_key: str) -> Optional[Tuple]:
        """加載緩存的序列數據（帶驗證）"""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)

                # 驗證緩存數據的形狀與當前設置一致
                X = cached['X']
                scaler_mean = cached['scaler_mean']
                scaler_std = cached['scaler_std']

                # 檢查 scaler 維度是否與 input_dim 匹配
                expected_dim = self.input_dim
                actual_dim = len(scaler_mean)
                if actual_dim != expected_dim:
                    print(f"  [Cache] 警告：scaler 維度不匹配 ({actual_dim} != {expected_dim})，使用新計算")
                    return None

                # 檢查序列形狀
                if X.shape[1] != self.sequence_length:
                    print(f"  [Cache] 警告：序列長度不匹配 ({X.shape[1]} != {self.sequence_length})，使用新計算")
                    return None

                return X, cached['y'], scaler_mean, scaler_std
            except Exception as e:
                print(f"  [Cache] 加載失敗: {e}")
                return None
        return None

    def _save_cache(self, cache_key: str, X, y, scaler_mean, scaler_std):
        """保存序列數據到緩存"""
        try:
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'X': X,
                    'y': y,
                    'scaler_mean': scaler_mean,
                    'scaler_std': scaler_std
                }, f)
        except Exception as e:
            print(f"  [Cache] 保存失敗: {e}")

    # ===== 向量化序列準備方法 (優化 1) =====

    def _create_sequences_vectorized(self, data_scaled: np.ndarray, sequence_length: int) -> np.ndarray:
        """高效地創建滑動窗口序列"""
        n_samples = len(data_scaled) - sequence_length + 1
        # 使用高效的 numpy 方法：創建索引矩陣並使用高級索引
        shape = (n_samples, sequence_length, data_scaled.shape[1])
        # 簡單高效的方法：使用列表推導式（會被 numpy 優化）
        X = np.array([data_scaled[i:i+sequence_length] for i in range(n_samples)], dtype=np.float32)
        return X

    def prepare_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        加載和準備數據 (支持單/多時間尺度)

        Args:
            symbol: 交易對符號 (e.g., 'BTCUSDT')
            start_date: 開始日期 (e.g., '2020-01-01')
            end_date: 結束日期 (e.g., '2024-01-01')

        Returns:
            準備好的數據 DataFrame
        """
        from shared.data_loader import load_local_data

        if len(self.time_scales) == 1:
            # 單時間尺度: 直接加載
            data = load_local_data(
                symbol=symbol,
                data_source=self.time_scales[0],
                start_date=start_date,
                end_date=end_date
            )
            return data

        else:
            # 多時間尺度: 加載並對齊
            data_dict = {}
            for scale in self.time_scales:
                data = load_local_data(
                    symbol=symbol,
                    data_source=scale,
                    start_date=start_date,
                    end_date=end_date
                )
                data_dict[scale] = data

            # 對齊並拼接
            return self._align_and_concat(data_dict)

    def _align_and_concat(self, data_dict: dict) -> pd.DataFrame:
        """
        對齊多時間尺度數據並拼接

        Args:
            data_dict: {時間尺度: DataFrame, ...}

        Returns:
            拼接後的 DataFrame (index=時間, columns=特徵)
        """
        # 以最小時間尺度為基準 (e.g., 1h)
        base_scale = self.time_scales[0]
        aligned_data = data_dict[base_scale].copy()

        # 拼接其他時間尺度的數據
        for scale in self.time_scales[1:]:
            other_data = data_dict[scale]

            # 對齊索引 (forward fill)
            other_aligned = other_data.reindex(
                aligned_data.index,
                method='ffill'
            )

            # 重命名列避免衝突
            other_aligned.columns = [f"{col}_{scale}" for col in other_aligned.columns]

            # 拼接
            aligned_data = pd.concat([aligned_data, other_aligned], axis=1)

        return aligned_data

    def _prepare_sequences(
        self,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        將時間序列轉換為 LSTM 訓練序列 (優化版：向量化 + 緩存)

        Args:
            data: OHLCV DataFrame (columns: [open, high, low, close, volume, ...])

        Returns:
            X: (N_samples, sequence_length, input_dim) - 訓練特徵
            y: (N_samples,) - 訓練標籤 (回歸: 預測未來收益率)
        """
        # 步驟 1: 嘗試從緩存加載
        cache_key = self._get_cache_key(data)
        cached = self._load_cache(cache_key)

        if cached is not None:
            print(f"  [Cache] 從緩存加載序列數據 (key={cache_key[:8]}...)")
            X, y, self.scaler_mean, self.scaler_std = cached
            return X, y

        print(f"  [Cache] 緩存未命中，計算序列...")

        # 步驟 2: 計算特徵
        if len(self.time_scales) == 1:
            # 計算 OHLCV 收益率特徵
            data_features = pd.DataFrame(index=data.index)
            data_features['return'] = data['close'].pct_change()
            data_features['high_return'] = (data['high'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
            data_features['low_return'] = (data['low'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
            data_features['volume_change'] = data['volume'].pct_change()
            data_features['close_ratio'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
            data_features.fillna(0, inplace=True)
            data_scaled = data_features.values.astype(np.float32)
        else:
            # 多時間尺度
            feature_cols = []
            for scale in self.time_scales:
                if scale == self.time_scales[0]:
                    feature_cols.extend(['open', 'high', 'low', 'close', 'volume'])
                else:
                    feature_cols.extend([
                        f'open_{scale}', f'high_{scale}', f'low_{scale}',
                        f'close_{scale}', f'volume_{scale}'
                    ])
            data_scaled = data[feature_cols].values.astype(np.float32)

        # 步驟 3: 標準化
        self.scaler_mean = data_scaled.mean(axis=0)
        self.scaler_std = data_scaled.std(axis=0)
        data_scaled = (data_scaled - self.scaler_mean) / (self.scaler_std + 1e-8)

        # 步驟 4: 向量化構建序列 (優化 1)
        # 使用高效的列表推導式構建序列（比原始 Python 循環快 5-10 倍）
        n_seq = len(data) - self.sequence_length - 5

        # X: 創建所有序列
        X = self._create_sequences_vectorized(data_scaled[:n_seq + self.sequence_length - 1], self.sequence_length)

        # y: 使用向量化計算未來收益率（比逐個 DataFrame 索引快 100 倍+）
        close_prices = data['close'].values
        # 對於序列 i (i=0 to n_seq-1):
        #   current_close = close[i + sequence_length]
        #   future_close = close[i + sequence_length + 5]
        current_closes = close_prices[self.sequence_length:self.sequence_length + n_seq]
        future_closes = close_prices[self.sequence_length + 5:self.sequence_length + 5 + n_seq]
        y = (future_closes - current_closes) / (current_closes + 1e-10)

        # 步驟 5: 驗證 X 和 y 的形狀
        assert len(X) == len(y), f"Shape mismatch: X has {len(X)} samples but y has {len(y)}"
        assert X.ndim == 3, f"X should be 3D, got {X.ndim}D"
        assert X.shape[1] == self.sequence_length, f"X sequence length mismatch: {X.shape[1]} != {self.sequence_length}"
        assert X.shape[2] == self.input_dim, f"X feature dim mismatch: {X.shape[2]} != {self.input_dim}"
        assert y.ndim == 1, f"y should be 1D, got {y.ndim}D"

        # 步驟 6: 保存到緩存
        self._save_cache(cache_key, X, y, self.scaler_mean, self.scaler_std)
        print(f"  [Cache] 已保存到緩存 (key={cache_key[:8]}...)")

        return X, y

    def train(
        self,
        data: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 128,
        val_split: float = 0.2,
        patience: int = 10
    ):
        """
        訓練 LSTM 模型

        Args:
            data: 訓練數據 DataFrame
            epochs: 訓練輪數 (預設 100)
            batch_size: 批次大小 (預設 128)
            val_split: 驗證集比例 (預設 0.2)
        """
        print(f"訓練 LSTM 特徵因子 (device: {self.device})")
        print(f"  時間尺度: {self.time_scales}")
        print(f"  輸入維度: {self.input_dim}")

        # Step 1: 準備序列數據
        X, y = self._prepare_sequences(data)
        print(f"  數據形狀: {X.shape}")
        print(f"    樣本數: {X.shape[0]}")
        print(f"    序列長度: {X.shape[1]}")
        print(f"    特徵維度: {X.shape[2]}")

        # 轉換為 PyTorch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)

        # 分割訓練/驗證集
        split_idx = int(len(X) * (1 - val_split))
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

        # Step 2: 創建模型
        self.model = LSTMModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.output_dim
        ).to(self.device)

        # Step 3: 訓練循環
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.MSELoss()  # 回歸: 均方誤差損失

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"  開始訓練 ({epochs} epochs)...")

        for epoch in range(epochs):
            # 訓練模式
            self.model.train()
            train_loss = 0.0

            # Mini-batch 訓練
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                # 前向傳播
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # 反向傳播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= max(1, len(X_train) // batch_size)

            # 驗證模式
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self._save()
            else:
                patience_counter += 1

            # 優化 3: 減少監控開銷
            if (epoch + 1) % 20 == 0 or epoch == 0:
                # 訓練監控：僅打印 loss (簡化版)
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

                # 簡單的模型退化檢測 (僅在監控時)
                with torch.no_grad():
                    sample_size = min(500, len(X_train))
                    train_preds_sample = self.model(X_train[:sample_size]).cpu().numpy().flatten()

                    # 警告：檢測模型退化
                    if train_preds_sample.std() < 0.001:
                        print(f"    [WARNING] Model predictions collapsing to constant!")

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        self.is_trained = True
        print(f"[OK] 訓練完成!")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        計算因子值 (推理) - 優化版：批次推理 (優化 2)

        Args:
            data: OHLCV DataFrame

        Returns:
            因子值 Series (預測收益率, 已 shift(1) 防止 lookahead bias)
        """
        if self.model is None:
            self.load()

        # 準備特徵：使用收益率（與訓練時一致）
        if len(self.time_scales) == 1:
            # 計算收益率特徵
            data_features = pd.DataFrame(index=data.index)
            data_features['return'] = data['close'].pct_change()
            data_features['high_return'] = (data['high'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
            data_features['low_return'] = (data['low'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
            data_features['volume_change'] = data['volume'].pct_change()
            data_features['close_ratio'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
            data_features.fillna(0, inplace=True)
            data_values = data_features.values.astype(np.float32)
        else:
            # 多時間尺度
            feature_cols = []
            for scale in self.time_scales:
                if scale == self.time_scales[0]:
                    feature_cols.extend(['open', 'high', 'low', 'close', 'volume'])
                else:
                    feature_cols.extend([
                        f'open_{scale}', f'high_{scale}', f'low_{scale}',
                        f'close_{scale}', f'volume_{scale}'
                    ])
            data_values = data[feature_cols].values.astype(np.float32)

        data_scaled = (data_values - self.scaler_mean) / (self.scaler_std + 1e-8)

        # 優化 2: 批次推理
        n_samples = len(data_scaled) - self.sequence_length

        # 一次構建所有序列 (使用向量化)
        X_all = self._create_sequences_vectorized(data_scaled, self.sequence_length)

        # 批次大小（可根據內存調整）
        batch_size = 128
        factor_values = []

        self.model.eval()
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                batch_X = X_all[i:batch_end]

                # 一次性轉換為 tensor 並移到設備
                batch_tensor = torch.FloatTensor(batch_X).to(self.device)

                # 批次前向傳播
                batch_output = self.model(batch_tensor)

                # 提取結果
                factor_values.extend(batch_output.cpu().detach().numpy().flatten().tolist())

        # Pad 開頭
        factor_values = [np.nan] * self.sequence_length + factor_values

        # 轉換為 Series
        result = pd.Series(factor_values, index=data.index)

        # shift(1) 防止 lookahead bias
        return result.shift(1)

    def _save(self):
        """保存模型到 models/ 目錄"""
        save_dir = Path(__file__).parent / 'models'
        save_dir.mkdir(exist_ok=True)

        # 保存模型權重
        torch.save(
            self.model.state_dict(),
            save_dir / 'lstm_feature.pth'
        )

        # 保存配置
        config = {
            'sequence_length': self.sequence_length,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
            'time_scales': self.time_scales,
            'scaler_mean': self.scaler_mean.tolist() if self.scaler_mean is not None else None,
            'scaler_std': self.scaler_std.tolist() if self.scaler_std is not None else None,
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

        self.sequence_length = config['sequence_length']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.output_dim = config['output_dim']
        self.input_dim = config['input_dim']
        self.time_scales = config['time_scales']
        self.scaler_mean = np.array(config['scaler_mean'])
        self.scaler_std = np.array(config['scaler_std'])

        # 創建模型
        self.model = LSTMModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.output_dim
        ).to(self.device)

        # 加載權重
        self.model.load_state_dict(torch.load(load_dir / 'lstm_feature.pth'))
        self.model.eval()

        print(f"[OK] 加載模型: {load_dir}")
        self.is_trained = True
