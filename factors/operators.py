"""
Factor Operators Library
因子運算子庫（從phandas移植並適配單資產時間序列）

提供豐富的技術因子運算子：
- 時間序列運算子 (ts_*)
- 數學運算子 (log, sqrt, power等)
- 統計運算子 (rank, zscore, quantile等)
- 技術指標運算子 (高階封裝)

所有運算子都返回 pd.Series，可直接用於策略開發。
"""

import pandas as pd
import numpy as np
from scipy.stats import rankdata, norm
from typing import Union, Optional


# ==================== 時間序列運算子 ====================

def ts_delay(series: pd.Series, window: int) -> pd.Series:
    """
    時間延遲：返回N期前的值

    Args:
        series: 輸入序列
        window: 延遲期數

    Returns:
        pd.Series: 延遲後的序列
    """
    return series.shift(window)


def ts_delta(series: pd.Series, window: int) -> pd.Series:
    """
    時間變化：當前值 - N期前值

    Args:
        series: 輸入序列
        window: 期數

    Returns:
        pd.Series: 變化值序列
    """
    return series.diff(window)


def ts_pct_change(series: pd.Series, window: int) -> pd.Series:
    """
    時間百分比變化：(當前值 - N期前值) / N期前值

    Args:
        series: 輸入序列
        window: 期數

    Returns:
        pd.Series: 百分比變化序列
    """
    return series.pct_change(window)


def ts_sum(series: pd.Series, window: int) -> pd.Series:
    """
    時間序列求和：過去N期的總和

    Args:
        series: 輸入序列
        window: 窗口大小

    Returns:
        pd.Series: 滾動求和序列
    """
    return series.rolling(window).sum()


def ts_mean(series: pd.Series, window: int) -> pd.Series:
    """
    時間序列均值：過去N期的平均值（簡單移動平均）

    Args:
        series: 輸入序列
        window: 窗口大小

    Returns:
        pd.Series: 滾動均值序列
    """
    return series.rolling(window).mean()


def ts_std_dev(series: pd.Series, window: int) -> pd.Series:
    """
    時間序列標準差：過去N期的標準差

    Args:
        series: 輸入序列
        window: 窗口大小

    Returns:
        pd.Series: 滾動標準差序列
    """
    return series.rolling(window).std()


def ts_min(series: pd.Series, window: int) -> pd.Series:
    """
    時間序列最小值：過去N期的最小值

    Args:
        series: 輸入序列
        window: 窗口大小

    Returns:
        pd.Series: 滾動最小值序列
    """
    return series.rolling(window).min()


def ts_max(series: pd.Series, window: int) -> pd.Series:
    """
    時間序列最大值：過去N期的最大值

    Args:
        series: 輸入序列
        window: 窗口大小

    Returns:
        pd.Series: 滾動最大值序列
    """
    return series.rolling(window).max()


def ts_rank(series: pd.Series, window: int) -> pd.Series:
    """
    時間序列排名：當前值在過去N期中的排名（百分位）

    Args:
        series: 輸入序列
        window: 窗口大小

    Returns:
        pd.Series: 滾動排名序列（0-1之間）
    """
    def rank_last(x):
        if x.notna().sum() < window:
            return np.nan
        ranks = rankdata(x.to_numpy(), method='min')
        return ranks[-1] / len(ranks)

    return series.rolling(window).apply(rank_last, raw=False)


def ts_argmax(series: pd.Series, window: int) -> pd.Series:
    """
    時間序列最大值位置：過去N期最大值距離當前的期數

    Args:
        series: 輸入序列
        window: 窗口大小

    Returns:
        pd.Series: 最大值相對位置序列
    """
    def safe_argmax(s):
        if s.isna().all():
            return np.nan
        return (len(s) - 1) - s.argmax()

    return series.rolling(window).apply(safe_argmax, raw=False)


def ts_argmin(series: pd.Series, window: int) -> pd.Series:
    """
    時間序列最小值位置：過去N期最小值距離當前的期數

    Args:
        series: 輸入序列
        window: 窗口大小

    Returns:
        pd.Series: 最小值相對位置序列
    """
    def safe_argmin(s):
        if s.isna().all():
            return np.nan
        return (len(s) - 1) - s.argmin()

    return series.rolling(window).apply(safe_argmin, raw=False)


def ts_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    時間序列Z-Score：(當前值 - 均值) / 標準差

    Args:
        series: 輸入序列
        window: 窗口大小

    Returns:
        pd.Series: Z-Score序列
    """
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def ts_scale(series: pd.Series, window: int) -> pd.Series:
    """
    時間序列Min-Max標準化：(當前值 - 最小值) / (最大值 - 最小值)

    Args:
        series: 輸入序列
        window: 窗口大小

    Returns:
        pd.Series: 標準化序列（0-1之間）
    """
    min_val = series.rolling(window).min()
    max_val = series.rolling(window).max()
    return (series - min_val) / (max_val - min_val)


def ts_quantile(series: pd.Series, window: int, quantile: float = 0.5) -> pd.Series:
    """
    時間序列分位數：過去N期的指定分位數

    Args:
        series: 輸入序列
        window: 窗口大小
        quantile: 分位數（0-1之間）

    Returns:
        pd.Series: 滾動分位數序列
    """
    return series.rolling(window).quantile(quantile)


def ts_correlation(series1: pd.Series, series2: pd.Series, window: int) -> pd.Series:
    """
    時間序列相關性：兩個序列過去N期的相關系數

    Args:
        series1: 輸入序列1
        series2: 輸入序列2
        window: 窗口大小

    Returns:
        pd.Series: 滾動相關系數序列
    """
    return series1.rolling(window).corr(series2)


def ts_covariance(series1: pd.Series, series2: pd.Series, window: int) -> pd.Series:
    """
    時間序列協方差：兩個序列過去N期的協方差

    Args:
        series1: 輸入序列1
        series2: 輸入序列2
        window: 窗口大小

    Returns:
        pd.Series: 滾動協方差序列
    """
    return series1.rolling(window).cov(series2)


def ts_decay_linear(series: pd.Series, window: int) -> pd.Series:
    """
    線性衰減加權平均：近期數據權重更高

    Args:
        series: 輸入序列
        window: 窗口大小

    Returns:
        pd.Series: 線性衰減加權平均序列
    """
    def linear_weighted_avg(x):
        if x.isna().all():
            return np.nan
        weights = np.arange(1, len(x) + 1)
        return (x * weights).sum() / weights.sum()

    return series.rolling(window).apply(linear_weighted_avg, raw=False)


def ts_ema(series: pd.Series, span: int) -> pd.Series:
    """
    指數移動平均 (EMA)

    Args:
        series: 輸入序列
        span: 跨度（相當於窗口大小）

    Returns:
        pd.Series: EMA序列
    """
    return series.ewm(span=span, adjust=False).mean()


# ==================== 數學運算子 ====================

def log(series: pd.Series) -> pd.Series:
    """自然對數"""
    return np.log(series)


def log1p(series: pd.Series) -> pd.Series:
    """log(1 + x)，避免log(0)"""
    return np.log1p(series)


def sqrt(series: pd.Series) -> pd.Series:
    """平方根"""
    return np.sqrt(series)


def power(series: pd.Series, exponent: float) -> pd.Series:
    """幂次方"""
    return series ** exponent


def abs_val(series: pd.Series) -> pd.Series:
    """絕對值"""
    return series.abs()


def sign(series: pd.Series) -> pd.Series:
    """符號函數：正數返回1，負數返回-1，零返回0"""
    return np.sign(series)


def clip(series: pd.Series, lower: Optional[float] = None, upper: Optional[float] = None) -> pd.Series:
    """
    截斷函數：限制值在指定範圍內

    Args:
        series: 輸入序列
        lower: 下界
        upper: 上界

    Returns:
        pd.Series: 截斷後的序列
    """
    return series.clip(lower=lower, upper=upper)


# ==================== 統計運算子 ====================

def standardize(series: pd.Series) -> pd.Series:
    """
    全局標準化：(x - mean) / std

    Returns:
        pd.Series: 標準化序列
    """
    return (series - series.mean()) / series.std()


def normalize(series: pd.Series) -> pd.Series:
    """
    全局Min-Max標準化：(x - min) / (max - min)

    Returns:
        pd.Series: 歸一化序列（0-1之間）
    """
    return (series - series.min()) / (series.max() - series.min())


def rank_pct(series: pd.Series) -> pd.Series:
    """
    全局百分位排名：返回每個值在整個序列中的百分位

    Returns:
        pd.Series: 百分位排名序列（0-1之間）
    """
    return series.rank(pct=True)


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Winsorize處理：將極端值替換為指定分位數的值

    Args:
        series: 輸入序列
        lower: 下分位數
        upper: 上分位數

    Returns:
        pd.Series: 處理後的序列
    """
    lower_val = series.quantile(lower)
    upper_val = series.quantile(upper)
    return series.clip(lower=lower_val, upper=upper_val)


# ==================== 技術指標運算子 ====================

def sma(series: pd.Series, window: int) -> pd.Series:
    """簡單移動平均（Simple Moving Average）"""
    return ts_mean(series, window)


def ema(series: pd.Series, span: int) -> pd.Series:
    """指數移動平均（Exponential Moving Average）"""
    return ts_ema(series, span)


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    相對強弱指標（Relative Strength Index）

    Args:
        series: 價格序列
        window: 窗口大小（默認14）

    Returns:
        pd.Series: RSI序列（0-100之間）
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD指標

    Args:
        series: 價格序列
        fast: 快線EMA週期
        slow: 慢線EMA週期
        signal: 信號線EMA週期

    Returns:
        DataFrame: 包含 macd, signal, histogram 三列
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    })


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    布林帶指標

    Args:
        series: 價格序列
        window: 窗口大小
        num_std: 標準差倍數

    Returns:
        DataFrame: 包含 upper, middle, lower 三列
    """
    middle = sma(series, window)
    std = ts_std_dev(series, window)
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    return pd.DataFrame({
        'upper': upper,
        'middle': middle,
        'lower': lower
    })


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    平均真實波幅（Average True Range）

    Args:
        high: 最高價序列
        low: 最低價序列
        close: 收盤價序列
        window: 窗口大小

    Returns:
        pd.Series: ATR序列
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def momentum(series: pd.Series, window: int = 10) -> pd.Series:
    """
    動量指標：當前價格 - N期前價格

    Args:
        series: 價格序列
        window: 期數

    Returns:
        pd.Series: 動量序列
    """
    return ts_delta(series, window)


def rate_of_change(series: pd.Series, window: int = 10) -> pd.Series:
    """
    變化率指標（ROC）：(當前價格 - N期前價格) / N期前價格 * 100

    Args:
        series: 價格序列
        window: 期數

    Returns:
        pd.Series: ROC序列
    """
    return ts_pct_change(series, window) * 100


# ==================== 組合因子構建器 ====================

def create_mean_reversion_factor(series: pd.Series, window: int = 20) -> pd.Series:
    """
    均值回歸因子：(均值 - 當前值) / 標準差

    當因子為正時，價格低於均值，可能上漲
    當因子為負時，價格高於均值，可能下跌

    Args:
        series: 價格序列
        window: 窗口大小

    Returns:
        pd.Series: 均值回歸因子
    """
    mean = ts_mean(series, window)
    std = ts_std_dev(series, window)
    return (mean - series) / std


def create_momentum_factor(series: pd.Series, short: int = 10, long: int = 50) -> pd.Series:
    """
    動量因子：短期均線 - 長期均線

    當因子為正時，短期趨勢向上
    當因子為負時，短期趨勢向下

    Args:
        series: 價格序列
        short: 短期窗口
        long: 長期窗口

    Returns:
        pd.Series: 動量因子
    """
    ma_short = sma(series, short)
    ma_long = sma(series, long)
    return ma_short - ma_long


def create_volatility_factor(series: pd.Series, window: int = 20) -> pd.Series:
    """
    波動率因子：標準差 / 均值（變異係數）

    值越大表示波動率越高

    Args:
        series: 價格序列
        window: 窗口大小

    Returns:
        pd.Series: 波動率因子
    """
    mean = ts_mean(series, window)
    std = ts_std_dev(series, window)
    return std / mean


def create_trend_strength_factor(series: pd.Series, window: int = 20) -> pd.Series:
    """
    趨勢強度因子：線性回歸斜率 * R²

    值越大表示趨勢越強

    Args:
        series: 價格序列
        window: 窗口大小

    Returns:
        pd.Series: 趨勢強度因子
    """
    def calculate_trend(x):
        if x.isna().any() or len(x) < 2:
            return np.nan
        y = x.values
        X = np.arange(len(y))
        # 線性回歸
        coef = np.polyfit(X, y, 1)
        slope = coef[0]
        # R²
        y_pred = np.polyval(coef, X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return slope * r_squared

    return series.rolling(window).apply(calculate_trend, raw=False)


# ==================== 輔助函數 ====================

def fill_na(series: pd.Series, method: str = 'ffill') -> pd.Series:
    """
    填充缺失值

    Args:
        series: 輸入序列
        method: 填充方法（'ffill', 'bfill', 'mean', 'zero'）

    Returns:
        pd.Series: 填充後的序列
    """
    if method == 'ffill':
        return series.fillna(method='ffill')
    elif method == 'bfill':
        return series.fillna(method='bfill')
    elif method == 'mean':
        return series.fillna(series.mean())
    elif method == 'zero':
        return series.fillna(0)
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    print("因子運算子庫測試\n")

    # 創建模擬數據
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='1H')
    returns = np.random.normal(0.0001, 0.02, 1000)
    prices = 30000 * (1 + returns).cumprod()
    series = pd.Series(prices, index=dates)

    # 測試時間序列運算子
    print("=== 時間序列運算子測試 ===")
    print(f"原始價格（最後5個）：\n{series.tail()}\n")
    print(f"20期移動平均（最後5個）：\n{ts_mean(series, 20).tail()}\n")
    print(f"20期Z-Score（最後5個）：\n{ts_zscore(series, 20).tail()}\n")

    # 測試技術指標
    print("=== 技術指標測試 ===")
    rsi_values = rsi(series, 14)
    print(f"14期RSI（最後5個）：\n{rsi_values.tail()}\n")

    macd_values = macd(series)
    print(f"MACD（最後5個）：\n{macd_values.tail()}\n")

    # 測試組合因子
    print("=== 組合因子測試 ===")
    mr_factor = create_mean_reversion_factor(series, 20)
    print(f"均值回歸因子（最後5個）：\n{mr_factor.tail()}\n")

    momentum_factor = create_momentum_factor(series, 10, 50)
    print(f"動量因子（最後5個）：\n{momentum_factor.tail()}\n")

    print("測試完成！")
