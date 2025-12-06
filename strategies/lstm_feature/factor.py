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
from typing import Tuple, List

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
        將時間序列轉換為 LSTM 訓練序列

        Args:
            data: OHLCV DataFrame (columns: [open, high, low, close, volume, ...])

        Returns:
            X: (N_samples, sequence_length, input_dim) - 訓練特徵
            y: (N_samples,) - 訓練標籤 (二分類: 漲/跌)
        """
        # 選擇 OHLCV 特徵
        if len(self.time_scales) == 1:
            feature_cols = ['open', 'high', 'low', 'close', 'volume']
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

        # 標準化
        data_scaled = data[feature_cols].values.astype(np.float32)
        self.scaler_mean = data_scaled.mean(axis=0)
        self.scaler_std = data_scaled.std(axis=0)
        data_scaled = (data_scaled - self.scaler_mean) / (self.scaler_std + 1e-8)

        # 構建序列
        X, y = [], []

        for i in range(len(data) - self.sequence_length - 5):
            # X: 過去 sequence_length 根 K 線
            X.append(data_scaled[i:i+self.sequence_length])

            # y: 未來 5 期 (約 5 小時) 的收益是否為正
            future_close = data.iloc[i+self.sequence_length+5]['close']
            current_close = data.iloc[i+self.sequence_length]['close']
            future_return = (future_close - current_close) / current_close

            y.append(1 if future_return > 0.02 else 0)  # 漲幅 > 2% 記為上漲

        return np.array(X), np.array(y)

    def train(
        self,
        data: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 128,
        val_split: float = 0.2
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        patience = 10
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

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        self.is_trained = True
        print(f"[OK] 訓練完成!")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        計算因子值 (推理)

        Args:
            data: OHLCV DataFrame

        Returns:
            因子值 Series (預測概率 0-1, 已 shift(1) 防止 lookahead bias)
        """
        if self.model is None:
            self.load()

        # 準備特徵
        if len(self.time_scales) == 1:
            feature_cols = ['open', 'high', 'low', 'close', 'volume']
        else:
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

        # 構建序列
        factor_values = []

        self.model.eval()
        with torch.no_grad():
            for i in range(len(data) - self.sequence_length):
                seq = data_scaled[i:i+self.sequence_length]

                # 轉換為 tensor
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)

                # 推理
                output = self.model(seq_tensor)
                prob = torch.sigmoid(output).item()  # 轉換為概率 [0, 1]

                factor_values.append(prob)

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
