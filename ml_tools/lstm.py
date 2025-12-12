"""
LSTM模型定義

純PyTorch實現,不含業務邏輯
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM特徵提取器"""

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2, output_dim=32):
        """
        Args:
            input_dim: 輸入特徵維度 (通常是5,表示OHLCV)
            hidden_dim: LSTM隱藏層維度
            num_layers: LSTM層數
            dropout: Dropout率
            output_dim: 輸出特徵維度 (提取出多少個特徵)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # LSTM層
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # 全連接層 (將LSTM輸出投影到輸出維度)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_dim)

        Returns:
            output: (batch_size, output_dim) - 提取出的特徵
        """
        # LSTM輸出
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 取最後一個時間步的輸出
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # 全連接層
        features = self.fc(last_output)  # (batch_size, output_dim)

        return features
