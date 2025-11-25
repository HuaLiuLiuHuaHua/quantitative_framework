"""
卡爾曼濾波器配對交易因子

使用卡爾曼濾波器動態估計對沖比率,計算價差 z-score
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
import logging

from .cointegration_utils import get_pair_id

logger = logging.getLogger(__name__)


class KalmanPairsFactor:
    """
    單配對卡爾曼濾波器因子

    不同於傳統因子(多資產cross-sectional),此因子只處理一個指定的配對
    使用卡爾曼濾波器動態估計對沖比率和價差均值
    """

    def __init__(self, factor_name: str = "KalmanPairs"):
        self.factor_name = factor_name
        self.parameters = {}

    def calculate(
        self,
        price_data_dict: Dict[str, pd.DataFrame],
        pair: Tuple[str, str],
        delta: float = 0.0001,
        Ve: float = 0.001,
        **kwargs
    ) -> pd.DataFrame:
        """
        計算單個配對的價差 z-score

        Args:
            price_data_dict: {symbol: DataFrame} 價格數據字典
            pair: (asset1, asset2) 配對,例如 ('BTCUSDT', 'ETHUSDT')
            delta: 卡爾曼狀態轉移方差 (控制對沖比率變化速度)
            Ve: 測量誤差方差 (控制觀測噪音)
            **kwargs: 其他參數 (保留用於擴展)

        Returns:
            DataFrame with:
            - index: dates
            - columns: [pair_id] (單列,例如 'BTCUSDT_ETHUSDT')
            - values: z-score (e_t / sqrt_Q_t), shifted by 1 to avoid lookahead bias
        """
        asset1, asset2 = pair

        # 驗證資產存在
        if asset1 not in price_data_dict or asset2 not in price_data_dict:
            raise ValueError(f"配對資產不在數據字典中: {asset1}, {asset2}")

        # 提取收盤價
        df1 = price_data_dict[asset1]
        df2 = price_data_dict[asset2]

        price1 = df1['close']
        price2 = df2['close']

        # 對齊數據 (使用共同的日期索引)
        common_idx = price1.index.intersection(price2.index)
        if len(common_idx) < 100:
            logger.warning(
                f"配對 {asset1}-{asset2} 共同數據點不足 ({len(common_idx)}), "
                "可能影響因子計算質量"
            )

        price1_aligned = price1.loc[common_idx].values
        price2_aligned = price2.loc[common_idx].values

        # 卡爾曼濾波器計算
        beta, alpha, e_t, sqrt_Q_t = self._kalman_filter(
            price1_aligned,
            price2_aligned,
            delta,
            Ve
        )

        # 計算 z-score (避免除以零)
        zscore = np.divide(
            e_t,
            sqrt_Q_t,
            out=np.zeros_like(e_t),
            where=sqrt_Q_t > 1e-10
        )

        # 構建 DataFrame
        pair_id = get_pair_id(asset1, asset2)
        factor_df = pd.DataFrame({
            pair_id: zscore
        }, index=common_idx)

        # Shift(1) 避免前視偏差
        factor_df = factor_df.shift(1)

        # 存儲中間結果 (用於調試和可視化)
        self.parameters['last_beta'] = beta
        self.parameters['last_alpha'] = alpha
        self.parameters['last_e_t'] = e_t
        self.parameters['last_sqrt_Q_t'] = sqrt_Q_t
        self.parameters['last_pair'] = pair
        self.parameters['last_pair_id'] = pair_id

        return factor_df

    def _kalman_filter(
        self,
        y: np.ndarray,
        x: np.ndarray,
        delta: float,
        Ve: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        卡爾曼濾波器核心實現

        狀態空間模型:
        - 測量方程: y_t = β_t * x_t + α_t + ε_t,  ε_t ~ N(0, Ve)
        - 狀態轉移: θ_t = θ_t-1 + ω_t,  ω_t ~ N(0, Ww)

        其中:
        - θ_t = [β_t, α_t]^T  (狀態向量: 對沖比率和截距)
        - Ww = delta * I  (狀態轉移協方差矩陣)

        Args:
            y: 資產1價格序列 (被解釋變量)
            x: 資產2價格序列 (解釋變量)
            delta: 狀態轉移方差
            Ve: 測量誤差方差

        Returns:
            beta: 動態對沖比率 β_t
            alpha: 動態截距 α_t
            e_t: 預測誤差 (實際價差)
            sqrt_Q_t: 預測標準差
        """
        T = len(y)

        # 初始化輸出數組
        beta = np.zeros(T)
        alpha = np.zeros(T)
        e = np.zeros(T)
        sqrt_Q = np.zeros(T)

        # 初始化狀態向量 θ = [β, α]^T
        # 使用簡單 OLS 初始化
        theta = self._initialize_state(y, x)

        # 初始化狀態協方差矩陣 P
        P = np.eye(2) * 1.0  # 初始不確定性較大

        # 狀態轉移協方差矩陣
        Ww = np.eye(2) * delta

        # 卡爾曼濾波遞歸
        for t in range(T):
            # 觀測矩陣 F_t = [x_t, 1]
            F = np.array([x[t], 1.0])

            # === 預測步驟 (Predict) ===
            # 狀態預測: θ_t|t-1 = θ_t-1
            # (隨機漫步模型,狀態預測不變)

            # 一步預測: y_t|t-1 = F_t * θ_t|t-1
            y_pred = F @ theta

            # 預測誤差 (innovation)
            e[t] = y[t] - y_pred

            # 預測協方差: Q_t = F_t * P_t|t-1 * F_t^T + Ve
            Q = F @ P @ F.T + Ve
            sqrt_Q[t] = np.sqrt(max(Q, 1e-10))  # 避免負數或零

            # === 更新步驟 (Update) ===
            # 卡爾曼增益: K_t = P_t|t-1 * F_t^T / Q_t
            K = (P @ F) / Q

            # 狀態更新: θ_t = θ_t|t-1 + K_t * e_t
            theta = theta + K * e[t]

            # 協方差更新 (Joseph form for numerical stability)
            # P_t = (I - K_t * F_t) * P_t|t-1 * (I - K_t * F_t)^T + K_t * Ve * K_t^T + Ww
            I_KF = np.eye(2) - np.outer(K, F)
            P = I_KF @ P @ I_KF.T + np.outer(K, K) * Ve + Ww

            # 記錄當前狀態
            beta[t] = theta[0]
            alpha[t] = theta[1]

        return beta, alpha, e, sqrt_Q

    def _initialize_state(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        初始化卡爾曼濾波器狀態向量

        使用最近60個數據點的 OLS 回歸進行初始化

        Args:
            y: 資產1價格序列
            x: 資產2價格序列

        Returns:
            theta_0: 初始狀態向量 [β_0, α_0]^T
        """
        try:
            # 使用前60個數據點 (或全部如果不足)
            n_init = min(60, len(y))
            y_init = y[:n_init]
            x_init = x[:n_init]

            # OLS 回歸: y = β * x + α
            X = np.column_stack([x_init, np.ones(n_init)])
            theta = np.linalg.lstsq(X, y_init, rcond=None)[0]

            return theta  # [β, α]

        except Exception as e:
            logger.warning(f"狀態初始化失敗,使用默認值: {e}")
            return np.array([1.0, 0.0])  # 默認 β=1, α=0

    def get_parameter_grid(self) -> Dict[str, Any]:
        """
        返回參數優化範圍

        Returns:
            參數網格字典
        """
        return {
            'delta': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            'Ve': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        """
        返回默認參數

        Returns:
            默認參數字典
        """
        return {
            'delta': 0.0001,
            'Ve': 0.001,
        }

    def get_last_hedge_ratio(self) -> Optional[np.ndarray]:
        """
        獲取最後一次計算的對沖比率序列

        Returns:
            對沖比率數組,如果未計算則返回 None
        """
        return self.parameters.get('last_beta')

    def get_last_spread(self) -> Optional[np.ndarray]:
        """
        獲取最後一次計算的價差序列

        Returns:
            價差數組,如果未計算則返回 None
        """
        return self.parameters.get('last_e_t')

    def __call__(
        self,
        data: Dict[str, pd.DataFrame],
        pair: Tuple[str, str],
        **params
    ) -> pd.DataFrame:
        """
        使因子可調用 (callable)

        Args:
            data: 價格數據字典
            pair: 配對
            **params: 參數

        Returns:
            因子 DataFrame
        """
        return self.calculate(data, pair=pair, **params)
