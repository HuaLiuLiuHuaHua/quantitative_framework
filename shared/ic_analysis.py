"""
Information Coefficient (IC) Analysis Module
IC分析模組（從phandas移植並增強）

用於評估因子預測能力：
- Daily IC (Spearman/Pearson/Kendall)
- IC Information Ratio (ICIR)
- Rolling IC
- IC Decay分析
- IC自相關
- Newey-West調整
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Optional, Callable, Dict
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ICAnalyzer:
    """
    IC分析器

    評估因子對未來收益的預測能力。
    """

    def __init__(
        self,
        factor_func: Callable,
        data: pd.DataFrame,
        factor_params: Optional[Dict] = None,
        forward_periods: int = 1,
        method: Literal['spearman', 'pearson', 'kendall'] = 'spearman'
    ):
        """
        初始化IC分析器

        Args:
            factor_func: 因子計算函數（返回因子值序列）
            data: OHLCV數據
            factor_params: 因子參數（可選）
            forward_periods: 未來收益期數
            method: 相關性計算方法
        """
        self.factor_func = factor_func
        self.data = data
        self.factor_params = factor_params or {}
        self.forward_periods = forward_periods
        self.method = method

        self.factor_values = None
        self.forward_returns = None
        self.daily_ic = None
        self.ic_summary = None

    def compute_factor(self) -> pd.Series:
        """計算因子值"""
        if self.factor_values is None:
            self.factor_values = self.factor_func(self.data, **self.factor_params)

        return self.factor_values

    def compute_forward_returns(self) -> pd.Series:
        """計算未來收益率"""
        if self.forward_returns is None:
            returns = self.data['close'].pct_change(self.forward_periods).shift(-self.forward_periods)
            self.forward_returns = returns

        return self.forward_returns

    def calculate_daily_ic(self) -> pd.Series:
        """
        計算每日IC（因子與未來收益的相關性）

        Returns:
            pd.Series: 每日IC時間序列
        """
        factor = self.compute_factor()
        forward_ret = self.compute_forward_returns()

        # 對齊數據
        df = pd.DataFrame({
            'factor': factor,
            'forward_return': forward_ret
        }).dropna()

        if len(df) == 0:
            raise ValueError("沒有足夠的有效數據計算IC")

        # 由於是時間序列數據（單資產），IC是全局計算
        # 但我們可以使用滾動窗口來評估時間變化的預測能力
        ic_list = []

        # 使用滾動窗口計算局部IC
        window = min(100, len(df) // 10)  # 自適應窗口大小

        for i in range(window, len(df)):
            window_data = df.iloc[i-window:i]

            if len(window_data) < 10:
                ic_list.append(np.nan)
                continue

            try:
                if self.method == 'spearman':
                    result = stats.spearmanr(window_data['factor'], window_data['forward_return'])
                    corr = result.correlation
                elif self.method == 'pearson':
                    corr, _ = stats.pearsonr(window_data['factor'], window_data['forward_return'])
                elif self.method == 'kendall':
                    corr, _ = stats.kendalltau(window_data['factor'], window_data['forward_return'])
                else:
                    corr = np.nan

                ic_list.append(corr)
            except Exception:
                ic_list.append(np.nan)

        # 創建IC時間序列（與原始數據對齊）
        ic_series = pd.Series([np.nan] * window + ic_list, index=df.index)
        self.daily_ic = ic_series.dropna()

        return self.daily_ic

    def calculate_summary(self, newey_west: bool = True) -> Dict:
        """
        計算IC統計摘要

        Args:
            newey_west: 是否應用Newey-West調整

        Returns:
            Dict: IC統計指標
        """
        if self.daily_ic is None:
            self.calculate_daily_ic()

        ic_clean = self.daily_ic.dropna()

        if len(ic_clean) == 0:
            return self._get_empty_summary()

        mean_ic = ic_clean.mean()
        std_ic = ic_clean.std()
        icir = mean_ic / std_ic if std_ic > 0 else np.nan

        # Newey-West調整
        if newey_west and len(ic_clean) > 10:
            lag = min(int(4 * (len(ic_clean) / 100) ** (2/9)), len(ic_clean) // 4)
            nw_std = self._newey_west_std(ic_clean.values, lag)
            icir_nw = mean_ic / nw_std if nw_std > 0 else np.nan
        else:
            icir_nw = icir

        hit_rate = (ic_clean > 0).mean()
        ic_pos_mean = ic_clean[ic_clean > 0].mean() if (ic_clean > 0).any() else 0
        ic_neg_mean = ic_clean[ic_clean < 0].mean() if (ic_clean < 0).any() else 0

        self.ic_summary = {
            'mean_ic': mean_ic,
            'std_ic': std_ic,
            'icir': icir,
            'icir_nw': icir_nw,
            'hit_rate': hit_rate,
            'ic_pos_mean': ic_pos_mean,
            'ic_neg_mean': ic_neg_mean,
            'n_periods': len(ic_clean),
            'ic_min': ic_clean.min(),
            'ic_max': ic_clean.max()
        }

        return self.ic_summary

    def calculate_rolling_ic(self, window: int = 63) -> pd.Series:
        """
        計算滾動IC

        Args:
            window: 滾動窗口大小

        Returns:
            pd.Series: 滾動IC時間序列
        """
        if self.daily_ic is None:
            self.calculate_daily_ic()

        rolling_ic = self.daily_ic.rolling(window, min_periods=max(window // 2, 20)).mean()
        return rolling_ic

    def calculate_ic_decay(self, lags: list = None) -> pd.DataFrame:
        """
        計算IC衰減（不同預測期的IC）

        Args:
            lags: 預測期列表

        Returns:
            DataFrame: 各預測期的IC統計
        """
        if lags is None:
            lags = [1, 2, 3, 5, 10, 20]

        results = []

        original_periods = self.forward_periods

        for lag in lags:
            self.forward_periods = lag
            self.forward_returns = None  # 重置
            self.daily_ic = None  # 重置

            try:
                self.calculate_daily_ic()
                summary = self.calculate_summary(newey_west=True)

                results.append({
                    'lag': lag,
                    'mean_ic': summary['mean_ic'],
                    'icir': summary['icir'],
                    'icir_nw': summary['icir_nw'],
                    'hit_rate': summary['hit_rate']
                })
            except Exception as e:
                print(f"計算lag={lag}時出錯: {e}")
                results.append({
                    'lag': lag,
                    'mean_ic': np.nan,
                    'icir': np.nan,
                    'icir_nw': np.nan,
                    'hit_rate': np.nan
                })

        # 恢復原始設置
        self.forward_periods = original_periods
        self.forward_returns = None
        self.daily_ic = None

        return pd.DataFrame(results).set_index('lag')

    def plot_ic_analysis(self, save_path: Optional[Path] = None):
        """
        繪製IC分析圖表

        Args:
            save_path: 保存路徑（可選）
        """
        if self.daily_ic is None:
            self.calculate_daily_ic()

        if self.ic_summary is None:
            self.calculate_summary()

        ic_clean = self.daily_ic.dropna()

        if len(ic_clean) == 0:
            print("沒有有效的IC數據可繪圖")
            return

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 1])

        # 計算移動平均
        ma20 = ic_clean.rolling(20, min_periods=10).mean()
        ma63 = ic_clean.rolling(63, min_periods=30).mean()

        # 主圖：IC時間序列
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(ic_clean.index, ic_clean.values, linewidth=0.8, alpha=0.4,
                color='#94a3b8', label='Daily IC')
        ax1.plot(ma20.index, ma20.values, linewidth=1.5, alpha=0.9,
                color='#3b82f6', label='MA20')
        ax1.plot(ma63.index, ma63.values, linewidth=1.8, alpha=0.95,
                color='#dc2626', label='MA63')
        ax1.axhline(y=0, color='#6b7280', linestyle='--', linewidth=0.8, alpha=0.5)
        ax1.fill_between(ic_clean.index, 0, ic_clean.values,
                        where=(ic_clean.values > 0), alpha=0.08, color='#3b82f6')
        ax1.fill_between(ic_clean.index, 0, ic_clean.values,
                        where=(ic_clean.values < 0), alpha=0.08, color='#ef4444')
        ax1.set_title(f'IC Analysis (Forward Period={self.forward_periods})', fontsize=13, fontweight='500', pad=15)
        ax1.set_xlabel('Date', fontsize=10)
        ax1.set_ylabel('IC', fontsize=10)
        ax1.legend(loc='upper right', fontsize=9, framealpha=0.95)
        ax1.grid(True, alpha=0.2)

        # 添加統計信息
        stats_text = (
            f"Mean IC: {self.ic_summary['mean_ic']:.4f}\n"
            f"ICIR (NW): {self.ic_summary['icir_nw']:.3f}\n"
            f"Hit Rate: {self.ic_summary['hit_rate']:.1%}\n"
            f"Periods: {self.ic_summary['n_periods']}"
        )
        ax1.text(0.015, 0.97, stats_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                         edgecolor='#d1d5db', alpha=0.95, linewidth=1))

        # 子圖1：IC分佈
        ax2 = fig.add_subplot(gs[1, 0])
        values = ic_clean.values
        ax2.hist(values, bins=30, density=True, color='#c7d2fe', alpha=0.9,
                edgecolor='white', linewidth=0.4)
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(values[~np.isnan(values)])
            xs = np.linspace(np.nanmin(values), np.nanmax(values), 200)
            ax2.plot(xs, kde(xs), color='#3b82f6', linewidth=1.6, label='KDE')
        except Exception:
            pass
        ax2.axvline(x=0, color='#6b7280', linestyle='--', linewidth=0.8)
        ax2.axvline(x=ic_clean.mean(), color='#dc2626', linestyle='-', linewidth=1.2)
        ax2.set_title('IC Distribution', fontsize=11, fontweight='400')
        ax2.set_xlabel('IC', fontsize=9)
        ax2.set_ylabel('Density', fontsize=9)
        ax2.grid(True, alpha=0.2, axis='y')

        # 子圖2：滾動IC
        ax3 = fig.add_subplot(gs[1, 1])
        rolling_ic = self.calculate_rolling_ic(window=63)
        ax3.plot(rolling_ic.index, rolling_ic.values, linewidth=1.5, color='#8b5cf6')
        ax3.axhline(y=0, color='#6b7280', linestyle='--', linewidth=0.8, alpha=0.5)
        ax3.set_title('Rolling IC (63D)', fontsize=11, fontweight='400')
        ax3.set_xlabel('Date', fontsize=9)
        ax3.set_ylabel('IC', fontsize=9)
        ax3.grid(True, alpha=0.2)
        ax3.tick_params(axis='x', rotation=15)

        # 子圖3：IC Decay
        ax4 = fig.add_subplot(gs[1, 2])
        decay_df = self.calculate_ic_decay()
        ax4.plot(decay_df.index, decay_df['mean_ic'], marker='o', linewidth=2,
                markersize=6, color='#2563eb')
        ax4.axhline(y=0, color='#6b7280', linestyle='--', linewidth=0.8, alpha=0.5)

        best_lag = decay_df['icir_nw'].idxmax()
        ax4.text(0.05, 0.95, f'Best: {best_lag}D', transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef08a',
                         edgecolor='#facc15', alpha=0.9, linewidth=1))

        ax4.set_title('IC Decay', fontsize=11, fontweight='400')
        ax4.set_xlabel('Forward Period (Days)', fontsize=9)
        ax4.set_ylabel('Mean IC', fontsize=9)
        ax4.grid(True, alpha=0.2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def print_summary(self):
        """打印IC分析摘要"""
        if self.ic_summary is None:
            self.calculate_summary()

        print("\n" + "=" * 60)
        print("IC分析摘要")
        print("=" * 60)
        print(f"Forward Period:     {self.forward_periods}")
        print(f"Method:             {self.method}")
        print(f"Sample Size:        {self.ic_summary['n_periods']}")
        print("-" * 60)
        print(f"Mean IC:            {self.ic_summary['mean_ic']:.4f}")
        print(f"Std IC:             {self.ic_summary['std_ic']:.4f}")
        print(f"ICIR:               {self.ic_summary['icir']:.3f}")
        print(f"ICIR (NW):          {self.ic_summary['icir_nw']:.3f}")
        print(f"Hit Rate:           {self.ic_summary['hit_rate']:.1%}")
        print("-" * 60)
        print(f"IC Max:             {self.ic_summary['ic_max']:.4f}")
        print(f"IC Min:             {self.ic_summary['ic_min']:.4f}")
        print(f"IC Positive Mean:   {self.ic_summary['ic_pos_mean']:.4f}")
        print(f"IC Negative Mean:   {self.ic_summary['ic_neg_mean']:.4f}")
        print("=" * 60)

        # 評估因子質量
        self._interpret_ic()

    def _interpret_ic(self):
        """解釋IC結果"""
        if self.ic_summary is None:
            return

        mean_ic = abs(self.ic_summary['mean_ic'])
        icir_nw = self.ic_summary['icir_nw']

        print("\n因子質量評估:")

        # IC強度評估
        if mean_ic > 0.05:
            ic_quality = "強"
        elif mean_ic > 0.03:
            ic_quality = "中等"
        elif mean_ic > 0.02:
            ic_quality = "弱"
        else:
            ic_quality = "很弱"

        print(f"  IC強度: {ic_quality} (|IC|={mean_ic:.4f})")

        # ICIR評估
        if icir_nw > 0.5:
            icir_quality = "優秀"
        elif icir_nw > 0.3:
            icir_quality = "良好"
        elif icir_nw > 0.1:
            icir_quality = "可接受"
        else:
            icir_quality = "較差"

        print(f"  ICIR質量: {icir_quality} (ICIR={icir_nw:.3f})")

        # 綜合建議
        if mean_ic > 0.03 and icir_nw > 0.3:
            print("  ✓ 建議：因子預測能力較強，可考慮使用")
        elif mean_ic > 0.02 and icir_nw > 0.1:
            print("  ⚠ 建議：因子預測能力中等，建議與其他因子組合使用")
        else:
            print("  ✗ 建議：因子預測能力較弱，不建議單獨使用")

    def _newey_west_std(self, data: np.ndarray, lag: int) -> float:
        """計算Newey-West調整標準差"""
        n = len(data)
        mean = np.mean(data)

        var = np.sum((data - mean) ** 2) / n

        for k in range(1, lag + 1):
            weight = 1 - k / (lag + 1)
            autocov = np.sum((data[k:] - mean) * (data[:-k] - mean)) / n
            var += 2 * weight * autocov

        return np.sqrt(var)

    def _get_empty_summary(self) -> Dict:
        """返回空的摘要字典"""
        return {
            'mean_ic': np.nan,
            'std_ic': np.nan,
            'icir': np.nan,
            'icir_nw': np.nan,
            'hit_rate': np.nan,
            'ic_pos_mean': np.nan,
            'ic_neg_mean': np.nan,
            'n_periods': 0,
            'ic_min': np.nan,
            'ic_max': np.nan
        }


def quick_ic_analysis(
    factor_func: Callable,
    data: pd.DataFrame,
    factor_params: Optional[Dict] = None,
    forward_periods: int = 1,
    method: str = 'spearman'
) -> ICAnalyzer:
    """
    快速IC分析（一鍵執行）

    Args:
        factor_func: 因子計算函數
        data: OHLCV數據
        factor_params: 因子參數
        forward_periods: 未來收益期數
        method: 相關性方法

    Returns:
        ICAnalyzer: 完成分析的分析器對象

    Example:
        >>> def momentum_factor(df, lookback=20):
        ...     return df['close'].pct_change(lookback)
        >>> analyzer = quick_ic_analysis(momentum_factor, data, {'lookback': 20})
    """
    analyzer = ICAnalyzer(factor_func, data, factor_params, forward_periods, method)

    print("=" * 60)
    print("開始IC分析")
    print("=" * 60)

    # 計算IC
    analyzer.calculate_daily_ic()
    analyzer.calculate_summary(newey_west=True)

    # 打印摘要
    analyzer.print_summary()

    # 繪製圖表
    analyzer.plot_ic_analysis()

    return analyzer


if __name__ == "__main__":
    print("IC分析模組測試\n")

    # 創建模擬數據
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='1H')
    returns = np.random.normal(0.0001, 0.02, 1000)
    prices = 30000 * (1 + returns).cumprod()

    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 1000)),
        'high': prices * (1 + np.abs(np.random.normal(0.002, 0.001, 1000))),
        'low': prices * (1 - np.abs(np.random.normal(0.002, 0.001, 1000))),
        'close': prices,
        'volume': np.random.uniform(100, 1000, 1000)
    }, index=dates)

    # 定義測試因子
    def momentum_factor(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """動量因子：過去N期收益率"""
        return df['close'].pct_change(lookback)

    # 執行快速IC分析
    print("執行快速IC分析...\n")

    analyzer = quick_ic_analysis(
        factor_func=momentum_factor,
        data=data,
        factor_params={'lookback': 20},
        forward_periods=5,
        method='spearman'
    )

    print("\n測試完成！")
