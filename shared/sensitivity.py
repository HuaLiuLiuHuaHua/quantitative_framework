"""
Sensitivity Analysis Module
敏感性分析模組

用於評估策略參數的穩健性：
- 單參數敏感性分析
- 多參數敏感性分析
- 參數穩定性評分
- 可視化敏感性熱圖
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Callable, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SensitivityAnalyzer:
    """
    敏感性分析器

    評估策略參數變化對績效指標的影響，判斷最佳參數的穩健性。
    """

    def __init__(
        self,
        strategy_func: Callable,
        data: pd.DataFrame,
        base_params: Dict,
        metric: str = 'sharpe_ratio',
        backtest_func: Optional[Callable] = None
    ):
        """
        初始化敏感性分析器

        Args:
            strategy_func: 策略函數（返回信號序列）
            data: OHLCV 數據
            base_params: 基準參數（最佳參數）
            metric: 評估指標（default: 'sharpe_ratio'）
            backtest_func: 自定義回測函數（可選）
        """
        self.strategy_func = strategy_func
        self.data = data
        self.base_params = base_params
        self.metric = metric
        self.backtest_func = backtest_func

        self.single_param_results = {}
        self.multi_param_results = None
        self.stability_scores = {}

    def analyze_single_parameter(
        self,
        param_name: str,
        param_range: List,
        fixed_params: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        單參數敏感性分析

        固定其他參數，只變化一個參數，觀察績效變化。

        Args:
            param_name: 要分析的參數名稱
            param_range: 參數取值範圍
            fixed_params: 固定的其他參數（默認使用base_params）

        Returns:
            DataFrame: 包含參數值和對應績效指標
        """
        if fixed_params is None:
            fixed_params = self.base_params.copy()

        results = []

        for param_value in param_range:
            test_params = fixed_params.copy()
            test_params[param_name] = param_value

            # 執行回測
            metric_value = self._run_backtest(test_params)

            results.append({
                'parameter': param_name,
                'value': param_value,
                self.metric: metric_value
            })

        df = pd.DataFrame(results)
        self.single_param_results[param_name] = df

        return df

    def analyze_two_parameters(
        self,
        param1_name: str,
        param1_range: List,
        param2_name: str,
        param2_range: List,
        fixed_params: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        雙參數敏感性分析

        同時變化兩個參數，生成2D熱圖數據。

        Args:
            param1_name: 第一個參數名稱
            param1_range: 第一個參數範圍
            param2_name: 第二個參數名稱
            param2_range: 第二個參數範圍
            fixed_params: 固定的其他參數

        Returns:
            DataFrame: 交叉參數組合的績效矩陣
        """
        if fixed_params is None:
            fixed_params = self.base_params.copy()

        results = []

        for val1 in param1_range:
            for val2 in param2_range:
                test_params = fixed_params.copy()
                test_params[param1_name] = val1
                test_params[param2_name] = val2

                metric_value = self._run_backtest(test_params)

                results.append({
                    param1_name: val1,
                    param2_name: val2,
                    self.metric: metric_value
                })

        df = pd.DataFrame(results)

        # 生成pivot表格用於熱圖
        pivot_df = df.pivot(index=param2_name, columns=param1_name, values=self.metric)

        self.multi_param_results = {
            'params': (param1_name, param2_name),
            'data': df,
            'pivot': pivot_df
        }

        return df

    def calculate_stability_score(
        self,
        param_name: str,
        neighborhood_pct: float = 0.2
    ) -> Dict:
        """
        計算參數穩定性評分

        評估最佳參數附近的績效穩定性。

        Args:
            param_name: 參數名稱
            neighborhood_pct: 鄰域範圍（基準值的百分比）

        Returns:
            Dict: 穩定性評分和統計信息
        """
        if param_name not in self.single_param_results:
            raise ValueError(f"請先對 {param_name} 執行 analyze_single_parameter()")

        df = self.single_param_results[param_name]
        base_value = self.base_params[param_name]
        base_metric = df[df['value'] == base_value][self.metric].values[0]

        # 定義鄰域範圍
        lower_bound = base_value * (1 - neighborhood_pct)
        upper_bound = base_value * (1 + neighborhood_pct)

        neighborhood = df[
            (df['value'] >= lower_bound) &
            (df['value'] <= upper_bound)
        ]

        if len(neighborhood) < 2:
            return {
                'stability_score': 0.0,
                'neighborhood_mean': np.nan,
                'neighborhood_std': np.nan,
                'degradation_pct': np.nan
            }

        # 計算鄰域內的統計數據
        neighborhood_mean = neighborhood[self.metric].mean()
        neighborhood_std = neighborhood[self.metric].std()

        # 穩定性評分：鄰域均值 / 鄰域標準差（越高越穩定）
        if neighborhood_std > 0:
            stability_score = neighborhood_mean / neighborhood_std
        else:
            stability_score = np.inf

        # 計算相對於最佳值的性能下降百分比
        if base_metric != 0:
            degradation_pct = (base_metric - neighborhood_mean) / abs(base_metric) * 100
        else:
            degradation_pct = 0.0

        score_dict = {
            'stability_score': stability_score,
            'neighborhood_mean': neighborhood_mean,
            'neighborhood_std': neighborhood_std,
            'neighborhood_size': len(neighborhood),
            'base_metric': base_metric,
            'degradation_pct': degradation_pct
        }

        self.stability_scores[param_name] = score_dict

        return score_dict

    def is_parameter_robust(
        self,
        param_name: str,
        degradation_threshold: float = 10.0,
        min_stability_score: float = 1.0
    ) -> Tuple[bool, str]:
        """
        判斷參數是否穩健

        Args:
            param_name: 參數名稱
            degradation_threshold: 允許的最大性能下降百分比
            min_stability_score: 最小穩定性評分要求

        Returns:
            (is_robust, reason): 布林值和判斷理由
        """
        if param_name not in self.stability_scores:
            self.calculate_stability_score(param_name)

        score_dict = self.stability_scores[param_name]

        # 檢查性能下降是否在可接受範圍內
        if score_dict['degradation_pct'] > degradation_threshold:
            return False, f"鄰域性能下降過大 ({score_dict['degradation_pct']:.1f}% > {degradation_threshold}%)"

        # 檢查穩定性評分
        if score_dict['stability_score'] < min_stability_score:
            return False, f"穩定性評分過低 ({score_dict['stability_score']:.2f} < {min_stability_score})"

        return True, "參數穩健"

    def plot_single_parameter(
        self,
        param_name: str,
        save_path: Optional[Path] = None
    ):
        """
        繪製單參數敏感性曲線

        Args:
            param_name: 參數名稱
            save_path: 保存路徑（可選）
        """
        if param_name not in self.single_param_results:
            raise ValueError(f"請先對 {param_name} 執行 analyze_single_parameter()")

        df = self.single_param_results[param_name]
        base_value = self.base_params[param_name]

        fig, ax = plt.subplots(figsize=(10, 6))

        # 繪製敏感性曲線
        ax.plot(df['value'], df[self.metric], 'o-', linewidth=2, markersize=6, color='#3b82f6')

        # 標記最佳參數
        base_metric = df[df['value'] == base_value][self.metric].values
        if len(base_metric) > 0:
            ax.plot(base_value, base_metric[0], 'o', markersize=12,
                   color='#ef4444', label=f'最佳參數: {base_value}', zorder=5)

        # 標記鄰域
        if param_name in self.stability_scores:
            score_dict = self.stability_scores[param_name]
            neighborhood_pct = 0.2
            lower = base_value * (1 - neighborhood_pct)
            upper = base_value * (1 + neighborhood_pct)
            ax.axvspan(lower, upper, alpha=0.2, color='green', label='穩定性鄰域 (±20%)')

        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel(self.metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'參數敏感性分析: {param_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 添加穩定性信息
        if param_name in self.stability_scores:
            score_dict = self.stability_scores[param_name]
            info_text = (
                f"穩定性評分: {score_dict['stability_score']:.2f}\n"
                f"鄰域均值: {score_dict['neighborhood_mean']:.4f}\n"
                f"鄰域標準差: {score_dict['neighborhood_std']:.4f}\n"
                f"性能下降: {score_dict['degradation_pct']:.1f}%"
            )
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_two_parameters(
        self,
        save_path: Optional[Path] = None
    ):
        """
        繪製雙參數敏感性熱圖

        Args:
            save_path: 保存路徑（可選）
        """
        if self.multi_param_results is None:
            raise ValueError("請先執行 analyze_two_parameters()")

        param1_name, param2_name = self.multi_param_results['params']
        pivot_df = self.multi_param_results['pivot']

        fig, ax = plt.subplots(figsize=(10, 8))

        # 繪製熱圖
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn',
                   center=pivot_df.mean().mean(), ax=ax, cbar_kws={'label': self.metric})

        # 標記最佳參數組合
        base_val1 = self.base_params[param1_name]
        base_val2 = self.base_params[param2_name]

        try:
            row_idx = list(pivot_df.index).index(base_val2)
            col_idx = list(pivot_df.columns).index(base_val1)
            ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1,
                                      fill=False, edgecolor='blue', lw=3))
        except ValueError:
            pass

        ax.set_xlabel(param1_name, fontsize=12)
        ax.set_ylabel(param2_name, fontsize=12)
        ax.set_title(f'雙參數敏感性分析: {param1_name} vs {param2_name}',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def generate_report(self) -> pd.DataFrame:
        """
        生成完整的敏感性分析報告

        Returns:
            DataFrame: 所有參數的穩定性評分匯總
        """
        if not self.stability_scores:
            raise ValueError("請先對參數執行 calculate_stability_score()")

        report_data = []

        for param_name, score_dict in self.stability_scores.items():
            is_robust, reason = self.is_parameter_robust(param_name)

            report_data.append({
                '參數名稱': param_name,
                '基準值': self.base_params[param_name],
                '基準績效': score_dict['base_metric'],
                '鄰域均值': score_dict['neighborhood_mean'],
                '鄰域標準差': score_dict['neighborhood_std'],
                '穩定性評分': score_dict['stability_score'],
                '性能下降%': score_dict['degradation_pct'],
                '是否穩健': '✓' if is_robust else '✗',
                '判斷理由': reason
            })

        df = pd.DataFrame(report_data)
        return df

    def _run_backtest(self, params: Dict) -> float:
        """
        執行回測並返回指定指標

        Args:
            params: 策略參數

        Returns:
            float: 績效指標值
        """
        try:
            if self.backtest_func is not None:
                # 使用自定義回測函數
                results = self.backtest_func(self.data, params)
                return results['metrics'][self.metric]
            else:
                # 使用默認回測流程
                from .backtest import BacktestEngine

                signals = self.strategy_func(self.data, **params)
                engine = BacktestEngine(
                    data=self.data,
                    signals=signals,
                    transaction_cost=0.0006,
                    slippage=0.0001
                )
                results = engine.run()
                return results['metrics'][self.metric]

        except Exception as e:
            print(f"回測失敗 (params={params}): {e}")
            return np.nan


def quick_sensitivity_analysis(
    strategy_func: Callable,
    data: pd.DataFrame,
    base_params: Dict,
    param_ranges: Dict[str, List],
    metric: str = 'sharpe_ratio'
) -> SensitivityAnalyzer:
    """
    快速敏感性分析（一鍵執行所有參數）

    Args:
        strategy_func: 策略函數
        data: OHLCV數據
        base_params: 最佳參數
        param_ranges: 各參數的測試範圍，格式 {'param_name': [values]}
        metric: 評估指標

    Returns:
        SensitivityAnalyzer: 完成分析的分析器對象

    Example:
        >>> analyzer = quick_sensitivity_analysis(
        ...     strategy_func=my_strategy.generate_signals,
        ...     data=df,
        ...     base_params={'lookback': 20, 'threshold': 0.5},
        ...     param_ranges={
        ...         'lookback': list(range(10, 50, 5)),
        ...         'threshold': [0.3, 0.4, 0.5, 0.6, 0.7]
        ...     }
        ... )
        >>> report = analyzer.generate_report()
        >>> print(report)
    """
    analyzer = SensitivityAnalyzer(strategy_func, data, base_params, metric)

    print("=" * 60)
    print("開始敏感性分析")
    print("=" * 60)

    for param_name, param_range in param_ranges.items():
        print(f"\n正在分析參數: {param_name} (範圍: {param_range[0]} - {param_range[-1]})")

        # 執行單參數分析
        analyzer.analyze_single_parameter(param_name, param_range)

        # 計算穩定性評分
        score_dict = analyzer.calculate_stability_score(param_name)
        is_robust, reason = analyzer.is_parameter_robust(param_name)

        print(f"  穩定性評分: {score_dict['stability_score']:.2f}")
        print(f"  判斷結果: {reason}")

        # 繪製敏感性曲線
        analyzer.plot_single_parameter(param_name)

    # 生成報告
    print("\n" + "=" * 60)
    print("敏感性分析報告")
    print("=" * 60)
    report = analyzer.generate_report()
    print(report.to_string(index=False))

    return analyzer


if __name__ == "__main__":
    print("敏感性分析模組測試\n")

    # 創建模擬數據
    import numpy as np
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

    # 模擬策略函數
    def dummy_strategy(df: pd.DataFrame, lookback: int = 20, threshold: float = 0.5) -> pd.Series:
        ma = df['close'].rolling(lookback).mean().shift(1)
        signals = pd.Series(0, index=df.index)
        signals[df['close'] > ma * (1 + threshold/100)] = 1
        signals[df['close'] < ma * (1 - threshold/100)] = -1
        return signals

    # 執行快速敏感性分析
    print("執行快速敏感性分析...\n")

    analyzer = quick_sensitivity_analysis(
        strategy_func=dummy_strategy,
        data=data,
        base_params={'lookback': 20, 'threshold': 0.5},
        param_ranges={
            'lookback': list(range(10, 40, 5)),
            'threshold': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        },
        metric='sharpe_ratio'
    )

    print("\n測試完成！")


# ==================== 自定義穩健性分析 (Cross-Sectional Factor Specific) ====================

def get_parameter_neighborhood(
    param_name: str,
    param_value: int,
    param_grid: Dict,
    step_size: int = 2
) -> List[int]:
    """
    獲取參數鄰域 (上下2個步長)

    Args:
        param_name: 參數名稱
        param_value: 參數中心值
        param_grid: 參數範圍字典
        step_size: 鄰域步長數量

    Returns:
        List[int]: 鄰域參數值列表
    """
    param_range = param_grid.get(param_name, [])
    if np.size(param_range) == 0:
        return []

    param_list = sorted(param_range)
    try:
        center_idx = param_list.index(param_value)
    except ValueError:
        # 如果中心值不在列表中,找最接近的
        center_idx = min(range(len(param_list)), key=lambda i: abs(param_list[i] - param_value))

    # 獲取鄰域索引
    neighbors = []
    for offset in range(-step_size, step_size + 1):
        idx = center_idx + offset
        if 0 <= idx < len(param_list):
            neighbors.append(param_list[idx])

    return neighbors


def calculate_custom_robustness_score(
    results_df: pd.DataFrame,
    best_params: Dict,
    param_grid: Dict,
    metric: str = 'sharpe_ratio',
    sharpe_threshold: float = 1.25
) -> Dict:
    """
    計算自定義穩健性調整後分數

    公式:
    穩健性調整後分數 = 原始夏普比率 × (1 - (中心點績效 - 鄰域平均績效) / 中心點績效) × (1 - 鄰域績效的標準差)

    Args:
        results_df: 優化結果DataFrame
        best_params: 最優參數組合
        param_grid: 參數範圍字典
        metric: 評估指標 (default: 'sharpe_ratio')
        sharpe_threshold: 夏普比率閾值 (default: 1.25)

    Returns:
        Dict: 包含穩健性分析結果
    """
    # 1. 獲取中心點績效
    center_mask = pd.Series(True, index=results_df.index)
    for param_name, param_value in best_params.items():
        if param_name in results_df.columns:
            center_mask &= (results_df[param_name] == param_value)

    center_performance = results_df[center_mask][metric].values
    if len(center_performance) == 0:
        return {
            'is_robust': False,
            'robustness_score': 0.0,
            'reason': '無法找到中心點績效'
        }

    center_sharpe = float(center_performance[0])

    # 2. 檢查夏普比率是否大於閾值
    if center_sharpe <= sharpe_threshold:
        return {
            'is_robust': False,
            'robustness_score': 0.0,
            'center_sharpe': center_sharpe,
            'reason': f'夏普比率 ({center_sharpe:.4f}) 未達到閾值 ({sharpe_threshold})'
        }

    # 3. 獲取鄰域績效
    # OPTIMIZATION: Vectorized neighborhood filtering
    neighbor_performances = []
    param_names = list(best_params.keys())

    # 預先計算固定參數的基礎 mask（只計算一次）
    base_conditions = []
    for pname in param_names:
        if pname in results_df.columns and pname in param_grid:
            base_conditions.append(pname)

    for param_name, param_value in best_params.items():
        if param_name not in param_grid:
            continue

        neighbors = get_parameter_neighborhood(param_name, param_value, param_grid, step_size=2)
        # 排除中心點
        neighbors = [n for n in neighbors if n != param_value]

        if not neighbors:
            continue

        # 為當前 param_name 構建固定條件的 mask（只包含其他參數）
        fixed_mask = pd.Series(True, index=results_df.index)
        for pname, pval in best_params.items():
            if pname in results_df.columns and pname != param_name:
                fixed_mask &= (results_df[pname] == pval)

        # 批量查找所有鄰域值（使用 .isin() 向量化）
        neighbor_mask = fixed_mask & results_df[param_name].isin(neighbors)
        neighbor_perf = results_df.loc[neighbor_mask, metric].values

        neighbor_performances.extend(neighbor_perf.tolist())

    if len(neighbor_performances) == 0:
        return {
            'is_robust': False,
            'robustness_score': 0.0,
            'center_sharpe': center_sharpe,
            'reason': '無法找到鄰域績效數據'
        }

    neighbor_performances = np.array(neighbor_performances)
    neighbor_mean = np.mean(neighbor_performances)
    neighbor_std = np.std(neighbor_performances)

    # 4. 計算穩健性調整後分數
    # 公式: 原始夏普 × (1 - (中心點 - 鄰域平均) / 中心點) × (1 - 鄰域標準差)
    if center_sharpe == 0:
        degradation_factor = 0
    else:
        degradation_factor = 1 - ((center_sharpe - neighbor_mean) / abs(center_sharpe))

    volatility_factor = max(1 - neighbor_std, 0)  # 確保不為負

    robustness_score = center_sharpe * degradation_factor * volatility_factor

    return {
        'is_robust': True,
        'robustness_score': float(robustness_score),
        'center_sharpe': float(center_sharpe),
        'neighbor_mean': float(neighbor_mean),
        'neighbor_std': float(neighbor_std),
        'neighbor_count': len(neighbor_performances),
        'degradation_factor': float(degradation_factor),
        'volatility_factor': float(volatility_factor),
        'reason': f'穩健參數 (夏普={center_sharpe:.4f}, 穩健性分數={robustness_score:.4f})'
    }


def filter_robust_params(
    candidates_df: pd.DataFrame,
    all_results_df: pd.DataFrame,
    param_grid: Dict,
    metric: str = 'sharpe_ratio',
    sharpe_threshold: float = 1.25,
    turnover_min: float = 0.01,
    turnover_max: float = 0.7,
    top_n: int = 10
) -> Tuple[pd.DataFrame, Dict]:
    """
    篩選穩健參數組合

    流程:
    1. (已在外部完成) 篩選出候選參數組合 (candidates_df)
    2. (可選) 根據換手率進一步篩選
    3. 對每個候選組合，在 all_results_df 中查找鄰居並計算穩健性分數
    4. 選擇穩健性分數最高的組合

    Args:
        candidates_df: 待分析的候選參數 DataFrame
        all_results_df: 包含所有優化結果的 DataFrame，用於查找鄰居
        param_grid: 參數範圍字典
        metric: 評估指標
        sharpe_threshold: 夏普比率閾值 (用於穩健性分數計算內部)
        turnover_min: 最小換手率
        turnover_max: 最大換手率
        top_n: 返回前N個穩健組合

    Returns:
        (robust_df, best_robust_params): 穩健參數DataFrame和最佳穩健參數
    """
    if candidates_df.empty:
        return pd.DataFrame(), {}

    # 換手率篩選（僅在有turnover列時執行）
    if 'turnover' in candidates_df.columns:
        robust_df = candidates_df[
            (candidates_df['turnover'] >= turnover_min) &
            (candidates_df['turnover'] <= turnover_max)
        ].copy()
    else:
        robust_df = candidates_df.copy()

    if robust_df.empty:
        return pd.DataFrame(), {}

    # 對每個組合計算穩健性分數
    robustness_results = []
    param_names = [col for col in robust_df.columns if col in param_grid]

    for idx, row in robust_df.iterrows():
        params = {pname: row[pname] for pname in param_names}

        robustness_result = calculate_custom_robustness_score(
            results_df=all_results_df,
            best_params=params,
            param_grid=param_grid,
            metric=metric,
            sharpe_threshold=sharpe_threshold
        )

        result_row = {**params, **robustness_result}
        robustness_results.append(result_row)

    robust_df = pd.DataFrame(robustness_results)

    # 按穩健性分數排序並選取前N個
    robust_df = robust_df.sort_values('robustness_score', ascending=False).head(top_n)

    # 選擇最佳穩健參數
    if not robust_df.empty:
        best_robust_row = robust_df.iloc[0]
        best_robust_params = {pname: best_robust_row[pname] for pname in param_names if pname in best_robust_row}
    else:
        best_robust_params = {}

    return robust_df, best_robust_params
