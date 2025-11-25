"""
協整性檢驗工具模組

提供配對交易所需的協整性檢驗、半衰期計算等功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.ar_model import AutoReg
import logging
from statsmodels.tsa.stattools import kpss


logger = logging.getLogger(__name__)


def get_pair_id(asset1: str, asset2: str) -> str:
    """
    標準化配對ID (按字母順序排序)

    Args:
        asset1: 資產1符號
        asset2: 資產2符號

    Returns:
        標準化的配對ID,例如 "BTCUSDT_ETHUSDT"
    """
    symbols = sorted([asset1, asset2])
    return f"{symbols[0]}_{symbols[1]}"


def calculate_correlation(
    price1: pd.Series,
    price2: pd.Series,
    method: str = 'pearson'
) -> float:
    """
    計算兩個價格序列的相關係數

    Args:
        price1: 價格序列1
        price2: 價格序列2
        method: 相關係數方法 ('pearson', 'spearman', 'kendall')

    Returns:
        相關係數 (-1 到 1)
    """
    try:
        return price1.corr(price2, method=method)
    except Exception as e:
        logger.warning(f"計算相關係數失敗: {e}")
        return 0.0


def test_cointegration(
    price1: pd.Series,
    price2: pd.Series
) -> Tuple[bool, float, float]:
    """
    Engle-Granger 協整檢驗

    Args:
        price1: 價格序列1
        price2: 價格序列2

    Returns:
        is_cointegrated: 是否協整 (p-value < 0.05)
        pvalue: 協整檢驗 p-value
        test_statistic: 檢驗統計量
    """
    try:
        score, pvalue, _ = coint(price1, price2)
        is_cointegrated = pvalue < 0.05
        return is_cointegrated, pvalue, score
    except Exception as e:
        logger.warning(f"協整檢驗失敗: {e}")
        return False, 1.0, 0.0


def calculate_halflife(spread: pd.Series, max_lag: int = 1) -> float:
    """
    計算價差的均值回歸半衰期

    使用 AR(1) 模型估計: spread_t = c + φ * spread_{t-1} + ε_t
    半衰期 = -log(2) / log(φ)

    只有當 0 < φ < 1 時,序列才是均值回歸的,半衰期才有意義。

    Args:
        spread: 價差序列
        max_lag: AR 模型最大滯後階數

    Returns:
        半衰期 (天數)
        - 如果序列是均值回歸的: 返回正數半衰期
        - 如果序列不是均值回歸的 (φ ≤ 0 或 φ ≥ 1): 返回 np.inf
        - 如果計算失敗: 返回 np.inf
    """
    try:
        # 移除 NaN
        spread_clean = spread.dropna()

        if len(spread_clean) < 20:
            logger.debug("價差數據不足 (< 20 點)")
            return np.inf

        # 擬合 AR(1) 模型
        model = AutoReg(spread_clean, lags=max_lag, trend='c')
        results = model.fit()

        # 獲取 AR(1) 係數
        phi = results.params[1] if len(results.params) > 1 else 0.0

        # ===== 關鍵修復: 檢查均值回歸條件 =====
        # 只有當 0 < φ < 1 時,序列才是均值回歸的
        if phi <= 0:
            # φ ≤ 0: 振盪序列或白噪音,不是均值回歸
            logger.debug(f"AR(1) 係數 φ={phi:.4f} ≤ 0, 非均值回歸")
            return np.inf

        if phi >= 1.0:
            # φ ≥ 1: 單位根或爆炸過程,不穩定
            logger.debug(f"AR(1) 係數 φ={phi:.4f} ≥ 1, 序列不穩定")
            return np.inf

        # 計算半衰期 (此時保證 0 < φ < 1)
        halflife = -np.log(2) / np.log(phi)

        # 額外驗證: 半衰期應該是正數且有限
        if halflife <= 0 or np.isnan(halflife) or np.isinf(halflife):
            logger.debug(f"計算出的半衰期無效: {halflife}")
            return np.inf

        return halflife

    except Exception as e:
        logger.warning(f"計算半衰期失敗: {e}")
        return np.inf


def calculate_hedge_ratio_ols(
    price1: pd.Series,
    price2: pd.Series
) -> float:
    """
    使用 OLS 線性迴歸計算對沖比率

    模型: price1 = β * price2 + α + ε

    Args:
        price1: 依變數價格序列
        price2: 自變數價格序列

    Returns:
        對沖比率 β
    """
    try:
        from sklearn.linear_model import LinearRegression

        X = price2.values.reshape(-1, 1)
        y = price1.values

        model = LinearRegression()
        model.fit(X, y)

        return model.coef_[0]

    except Exception as e:
        logger.warning(f"計算對沖比率失敗: {e}")
        return 1.0


def find_best_cointegrated_pair(
    price_data_dict: Dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    min_correlation: float = 0.6,
    max_pvalue: float = 0.05,
    min_halflife: int = 5,
    max_halflife: int = 30
) -> Tuple[Optional[Tuple[str, str]], dict]:
    """
    從價格數據字典中找出協整率最高的配對

    分層篩選流程:
    1. 相關性篩選: correlation > min_correlation
    2. 協整性檢驗: p-value < max_pvalue
    3. 半衰期過濾: min_halflife < half_life < max_halflife
    4. 選擇 p-value 最小的配對

    Args:
        price_data_dict: {symbol: DataFrame} 價格數據字典
        start_date: 開始日期
        end_date: 結束日期
        min_correlation: 最小相關係數閾值
        max_pvalue: 最大 p-value 閾值
        min_halflife: 最小半衰期 (天)
        max_halflife: 最大半衰期 (天)

    Returns:
        best_pair: (asset1, asset2) 最佳配對,如果沒有則為 None
        metrics: {
            'pvalue': 協整 p-value,
            'correlation': 相關係數,
            'halflife': 半衰期,
            'hedge_ratio': 對沖比率,
            'test_statistic': 協整檢驗統計量,
            'candidates_count': 通過篩選的候選數量,
            'total_pairs_tested': 測試的總配對數
        }
    """
    symbols = list(price_data_dict.keys())
    n_symbols = len(symbols)

    if n_symbols < 2:
        logger.warning(f"資產數量不足 ({n_symbols}), 無法形成配對")
        return None, {'candidates_count': 0, 'total_pairs_tested': 0}

    logger.info(f"開始配對篩選: {n_symbols} 個資產, {n_symbols*(n_symbols-1)//2} 個可能配對")

    # 篩選結果
    candidates = []
    total_tested = 0

    for i in range(n_symbols):
        for j in range(i + 1, n_symbols):
            asset1 = symbols[i]
            asset2 = symbols[j]
            total_tested += 1

            try:
                # 提取價格數據
                price1 = price_data_dict[asset1].loc[start_date:end_date, 'close']
                price2 = price_data_dict[asset2].loc[start_date:end_date, 'close']

                # 對齊數據
                common_idx = price1.index.intersection(price2.index)
                if len(common_idx) < 100:
                    continue

                price1 = price1.loc[common_idx]
                price2 = price2.loc[common_idx]

                # 第一層: 相關性篩選
                corr = calculate_correlation(price1, price2)
                if abs(corr) < min_correlation:
                    continue

                # 第二層: 協整性檢驗
                is_coint, pvalue, test_stat = test_cointegration(price1, price2)
                if not is_coint or pvalue >= max_pvalue:
                    continue

                # 計算對沖比率和價差
                hedge_ratio = calculate_hedge_ratio_ols(price1, price2)
                spread = price1 - hedge_ratio * price2

                # 第三層: 半衰期過濾
                halflife = calculate_halflife(spread)
                if halflife < min_halflife or halflife > max_halflife:
                    continue

                # 通過所有篩選,加入候選
                candidates.append({
                    'pair': (asset1, asset2),
                    'pvalue': pvalue,
                    'correlation': corr,
                    'halflife': halflife,
                    'hedge_ratio': hedge_ratio,
                    'test_statistic': test_stat
                })

                logger.debug(
                    f"候選配對: {asset1}-{asset2}, "
                    f"p={pvalue:.4f}, corr={corr:.3f}, hl={halflife:.1f}"
                )

            except Exception as e:
                logger.debug(f"配對 {asset1}-{asset2} 檢驗失敗: {e}")
                continue

    logger.info(f"篩選完成: 測試 {total_tested} 個配對, 找到 {len(candidates)} 個候選")

    if not candidates:
        return None, {
            'candidates_count': 0,
            'total_pairs_tested': total_tested,
            'reason': 'no_candidates_passed_filters'
        }

    # 選擇 p-value 最小的配對 (協整性最強)
    best_candidate = min(candidates, key=lambda x: x['pvalue'])
    best_pair = best_candidate['pair']

    metrics = {
        'pvalue': best_candidate['pvalue'],
        'correlation': best_candidate['correlation'],
        'halflife': best_candidate['halflife'],
        'hedge_ratio': best_candidate['hedge_ratio'],
        'test_statistic': best_candidate['test_statistic'],
        'candidates_count': len(candidates),
        'total_pairs_tested': total_tested
    }

    logger.info(
        f"最佳配對: {best_pair[0]}-{best_pair[1]}, "
        f"p-value={metrics['pvalue']:.4f}, "
        f"半衰期={metrics['halflife']:.1f}天"
    )

    return best_pair, metrics


def recheck_cointegration(
    asset1_data: pd.DataFrame,
    asset2_data: pd.DataFrame,
    lookback_days: int = 252,
    max_pvalue: float = 0.05
) -> Tuple[bool, float]:
    """
    重新檢驗配對的協整性 (用於測試期開始時)

    使用最近 lookback_days 天的數據進行檢驗

    Args:
        asset1_data: 資產1的完整數據 (到測試期開始日期)
        asset2_data: 資產2的完整數據
        lookback_days: 回溯天數 (默認252天,約1年)
        max_pvalue: 最大 p-value 閾值

    Returns:
        is_cointegrated: 是否仍協整
        pvalue: 協整檢驗 p-value
    """
    try:
        # 取最近的數據
        price1 = asset1_data['close'].iloc[-lookback_days:]
        price2 = asset2_data['close'].iloc[-lookback_days:]

        # 對齊數據
        common_idx = price1.index.intersection(price2.index)
        if len(common_idx) < 100:
            logger.warning(f"數據點不足 ({len(common_idx)}), 無法重新檢驗協整性")
            return False, 1.0

        price1 = price1.loc[common_idx]
        price2 = price2.loc[common_idx]

        # 協整檢驗
        is_coint, pvalue, _ = test_cointegration(price1, price2)

        return is_coint and pvalue < max_pvalue, pvalue

    except Exception as e:
        logger.error(f"重新檢驗協整性失敗: {e}")
        return False, 1.0


def get_universe_symbols_for_date(
    universe_df: pd.DataFrame,
    target_date: pd.Timestamp,
    top_n: int = 40
) -> List[str]:
    """
    獲取指定日期的動態宇宙資產列表

    Args:
        universe_df: 宇宙DataFrame (從 40.csv 載入)
        target_date: 目標日期
        top_n: 返回前 N 個資產

    Returns:
        資產符號列表 (不包含USDT後綴)
    """
    # 找到最接近且不晚於目標日期的記錄
    universe_df = universe_df.copy()
    universe_df['timestamp'] = pd.to_datetime(universe_df['timestamp'], dayfirst=True)

    valid_dates = universe_df[universe_df['timestamp'] <= target_date]

    if valid_dates.empty:
        # 如果目標日期早於所有記錄,使用第一條
        row = universe_df.iloc[0]
    else:
        # 使用最接近的記錄
        row = valid_dates.iloc[-1]

    # 提取資產符號 (去除 NaN 和空字符串)
    symbols = [
        str(s) for s in row.iloc[1:top_n+1]
        if pd.notna(s) and s != ''
    ]

    return symbols

def test_stationarity_adf(spread: pd.Series) -> Tuple[float, bool, float, float]:
    """
    對價差進行 ADF 檢驗
    H0: 序列非平穩 (有單位根)
    
    Returns:
        p_value: ADF 檢驗的 p-value
        is_stationary: 是否平穩 (p < 0.05)
        adf_statistic: ADF t-statistic
        critical_value_95: 95% 信心水準的臨界值
    """
    try:
        result = adfuller(spread.dropna())
        p_value = result[1]
        adf_statistic = result[0]
        critical_values = result[4]
        critical_value_95 = critical_values.get('5%', -2.86)  # 95% 信心水準
        return p_value, p_value < 0.05, adf_statistic, critical_value_95
    except Exception as e:
        logger.warning(f"ADF 檢驗失敗: {e}")
        return 1.0, False, 0.0, -2.86

def test_stationarity_kpss(spread: pd.Series) -> Tuple[float, bool]:
    """
    對價差進行 KPSS 檢驗
    H0: 序列是平穩的
    """
    try:
        result = kpss(spread.dropna(), regression='c')
        p_value = result[1]
        # p-value > 0.05 表示無法拒絕 H0, 即序列是平穩的
        return p_value, p_value > 0.05
    except Exception as e:
        logger.warning(f"KPSS 檢驗失敗: {e}")
        return 0.0, False

def calculate_hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    """
    計算赫斯特指數 (Hurst Exponent) using Rescaled Range (R/S) analysis.
    H < 0.5: 均值回歸 (Mean-reverting)
    H = 0.5: 隨機漫步 (Geometric Brownian Motion)
    H > 0.5: 趨勢 (Trending)
    """
    series = series.dropna()
    n_min = 10  # Smallest window size
    if len(series) < n_min * 2:
        return np.nan

    # Adjust max_lag to be at most half the length of the series
    max_lag = min(max_lag, len(series) // 2)
    if max_lag < n_min:
        return np.nan

    # Create a range of lag values to iterate over
    lags = range(n_min, max_lag)

    # Calculate the array of the log of R/S
    try:
        # Pre-calculate cumulative sum and mean
        cumsum = np.cumsum(series - np.mean(series))
        
        rs_values = []
        for lag in lags:
            # Reshape for vectorized operations
            reshaped_series = series[:len(series) - (len(series) % lag)].values.reshape(-1, lag)
            
            # Calculate R
            mean_adj = np.mean(reshaped_series, axis=1, keepdims=True)
            cumsum_chunks = np.cumsum(reshaped_series - mean_adj, axis=1)
            R = np.max(cumsum_chunks, axis=1) - np.min(cumsum_chunks, axis=1)
            
            # Calculate S
            S = np.std(reshaped_series, axis=1)
            
            # Avoid division by zero
            S[S == 0] = 1e-10
            
            rs_values.append(np.mean(R / S))

        # Filter out zero or negative values before log
        valid_indices = [i for i, v in enumerate(rs_values) if v > 0]
        if len(valid_indices) < 2:
            return np.nan
            
        valid_lags = np.array(lags)[valid_indices]
        valid_rs = np.array(rs_values)[valid_indices]

        # Fit a line to the log-log plot
        log_lags = np.log(valid_lags)
        log_rs = np.log(valid_rs)
        
        poly = np.polyfit(log_lags, log_rs, 1)
        return poly[0]

    except (np.linalg.LinAlgError, ValueError) as e:
        logger.warning(f"Hurst Exponent calculation failed: {e}")
        return np.nan


def calculate_variance_ratio(series: pd.Series, lag: int = 2) -> float:
    """
    計算方差比 (Variance Ratio Test)
    
    用於檢驗序列是否為隨機漫步。
    公式: VR(k) = Var(r_k) / (k * Var(r_1))
    
    VR < 1: 均值回歸 (適合配對交易)
    VR = 1: 隨機漫步
    VR > 1: 趨勢型 (不適合均值回歸)
    
    Args:
        series: 價格或價差序列
        lag: 時間滯後週期 (默認 2)
    
    Returns:
        方差比值
    """
    try:
        series = series.dropna().values
        
        if len(series) < lag + 10:
            return np.nan
        
        # 計算單週期回報的方差
        returns_1 = np.diff(series)
        var_1 = np.var(returns_1, ddof=1)
        
        if var_1 == 0:
            return np.nan
        
        # 計算 k 週期回報
        returns_k = np.diff(series, n=lag)
        var_k = np.var(returns_k, ddof=1)
        
        # 計算方差比
        vr = var_k / (lag * var_1)
        
        return vr
    
    except Exception as e:
        logger.warning(f"方差比計算失敗: {e}")
        return np.nan


def johansen_test(price1: pd.Series, price2: pd.Series) -> Tuple[float, float, bool]:
    """
    約翰森協整檢驗 (Johansen Cointegration Test)
    
    檢驗兩個序列之間是否存在長期均衡關係。
    
    Returns:
        trace_statistic: Trace 統計量
        critical_value_95: 95% 信心水準的臨界值
        is_cointegrated: 是否協整 (Trace > Critical)
    """
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        
        # 準備數據
        price1_clean = price1.dropna().values
        price2_clean = price2.dropna().values
        
        # 對齊長度
        min_len = min(len(price1_clean), len(price2_clean))
        price1_clean = price1_clean[-min_len:]
        price2_clean = price2_clean[-min_len:]
        
        if min_len < 50:
            logger.warning("數據不足以進行約翰森檢驗")
            return np.nan, np.nan, False
        
        # 組合成 2xN 的數組
        data = np.column_stack([price1_clean, price2_clean])
        
        # 進行約翰森檢驗 (det_order=0 表示無確定性項)
        result = coint_johansen(data, det_order=0, k_ar_diff=1)
        
        # 獲取 trace 統計量 (第一個特徵值對應的 trace statistic)
        trace_statistic = result.lr1[0]  # lr1 是 Trace statistic
        
        # 95% 信心水準的臨界值
        critical_value_95 = result.cvt[0, 1]  # cvt[:, 1] 是 95% 臨界值
        
        # 判斷是否協整
        is_cointegrated = trace_statistic > critical_value_95
        
        return trace_statistic, critical_value_95, is_cointegrated
    
    except Exception as e:
        logger.warning(f"約翰森檢驗失敗: {e}")
        return np.nan, np.nan, False
