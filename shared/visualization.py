"""
Visualization Module
可視化模塊 - Medium風格的繪圖功能

功能：
1. plot_backtest_results: 回測結果可視化
   - 權益曲線
   - 價格和信號標記
   - 回撤曲線

2. plot_optimization_results: 參數優化結果可視化
   - 參數熱圖
   - 參數敏感性分析
   - 指標對比圖

3. plot_mcpt_distribution: MCPT結果可視化
   - 排列分佈直方圖
   - p值標記
   - 原始績效位置

4. plot_walkforward_performance: Walk-Forward結果可視化
   - 樣本外績效趨勢
   - IS vs OOS對比
   - 參數穩定性分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import warnings

# 設置繪圖風格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# 解決中文顯示問題
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_backtest_results(
    equity_curve: pd.Series,
    data: pd.DataFrame,
    signals: pd.Series,
    title: str = "回測結果",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10),
    show_buy_and_hold: bool = True
):
    """
    繪製回測結果

    包含3個子圖：
    1. 權益曲線（含長期持有對比）
    2. 價格和交易信號
    3. 回撤曲線（含長期持有對比）

    Args:
        equity_curve: 權益曲線Series
        data: OHLCV數據
        signals: 交易信號Series
        title: 圖表標題
        save_path: 保存路徑（可選）
        figsize: 圖表大小
        show_buy_and_hold: 是否顯示長期持有基準（默認True）

    Example:
        >>> plot_backtest_results(
        ...     equity_curve=results['equity_curve'],
        ...     data=df,
        ...     signals=signals,
        ...     save_path=Path("results/backtest_plot.png")
        ... )

    Raises:
        ValueError: If input data is empty or invalid
        KeyError: If required columns are missing
    """
    # ===== CRITICAL ISSUE FIX #3: Empty data validation =====
    if equity_curve is None or len(equity_curve) == 0:
        raise ValueError("equity_curve cannot be None or empty")

    if data is None or len(data) == 0:
        raise ValueError("data DataFrame cannot be None or empty")

    if signals is None or len(signals) == 0:
        raise ValueError("signals Series cannot be None or empty")

    # ===== HIGH SEVERITY ISSUE FIX #4: Missing 'close' column validation =====
    if 'close' not in data.columns:
        raise KeyError("data DataFrame must contain 'close' column")

    # ===== CRITICAL ISSUE FIX #1: NaN validation in input data =====
    if equity_curve.isna().any():
        raise ValueError("equity_curve contains NaN values - please clean data before plotting")

    if data['close'].isna().all():
        raise ValueError("data['close'] column is entirely NaN")

    # ===== CRITICAL ISSUE FIX #2: Division by zero checks =====
    initial_equity = equity_curve.iloc[0]
    if initial_equity == 0 or np.isnan(initial_equity) or np.isinf(initial_equity):
        raise ValueError(f"Invalid initial equity value: {initial_equity}")

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 計算長期持有（Buy & Hold）基準
    if show_buy_and_hold:
        initial_capital = equity_curve.iloc[0]
        # 對齊數據索引
        aligned_data = data.reindex(equity_curve.index, method='ffill')

        # ===== CRITICAL ISSUE FIX #1: Index alignment with NaN validation =====
        # Check if reindexing introduced NaN values
        if aligned_data['close'].isna().all():
            warnings.warn("Aligned data contains all NaN values - disabling buy & hold comparison")
            show_buy_and_hold = False
        else:
            # Forward fill any NaN values introduced by reindexing
            aligned_data['close'] = aligned_data['close'].fillna(method='ffill')

            # If still have NaN at the beginning, backward fill
            if aligned_data['close'].isna().any():
                aligned_data['close'] = aligned_data['close'].fillna(method='bfill')

            # ===== CRITICAL ISSUE FIX #2: Division by zero check for initial close price =====
            initial_close = aligned_data['close'].iloc[0]
            if initial_close == 0 or np.isnan(initial_close) or np.isinf(initial_close):
                warnings.warn(f"Invalid initial close price: {initial_close} - disabling buy & hold comparison")
                show_buy_and_hold = False
            else:
                buy_hold_returns = aligned_data['close'] / initial_close
                buy_hold_equity = initial_capital * buy_hold_returns

                # ===== HIGH SEVERITY ISSUE FIX #5: NaN propagation validation =====
                # Verify buy & hold equity doesn't contain NaN
                if buy_hold_equity.isna().any():
                    warnings.warn("Buy & hold equity calculation resulted in NaN values - disabling comparison")
                    show_buy_and_hold = False

    # 子圖1: 權益曲線
    axes[0].plot(equity_curve.index, equity_curve.values,
                 linewidth=2, color='#2E86AB', label='策略權益')

    if show_buy_and_hold:
        axes[0].plot(buy_hold_equity.index, buy_hold_equity.values,
                     linewidth=2, color='orange', label='長期持有', alpha=0.7, linestyle='--')

    axes[0].axhline(y=equity_curve.iloc[0], color='gray',
                    linestyle='--', alpha=0.5, label='初始資金')
    axes[0].set_ylabel('權益 ($)', fontsize=12)
    axes[0].set_title('權益曲線對比', fontsize=12, pad=10)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # 添加最終收益標註
    # ===== CRITICAL ISSUE FIX #2: Division by zero check for final return =====
    initial_eq = equity_curve.iloc[0]
    final_eq = equity_curve.iloc[-1]
    if initial_eq == 0 or np.isnan(initial_eq) or np.isinf(initial_eq):
        final_return = 0.0
        warnings.warn("Cannot calculate final return with invalid initial equity")
    else:
        final_return = (final_eq / initial_eq - 1) * 100

    if show_buy_and_hold:
        bh_initial_eq = buy_hold_equity.iloc[0]
        bh_final_eq = buy_hold_equity.iloc[-1]
        if bh_initial_eq == 0 or np.isnan(bh_initial_eq) or np.isinf(bh_initial_eq):
            bh_final_return = 0.0
            warnings.warn("Cannot calculate buy & hold final return with invalid initial equity")
        else:
            bh_final_return = (bh_final_eq / bh_initial_eq - 1) * 100
        text_content = f'策略收益率: {final_return:.2f}%\n長期持有: {bh_final_return:.2f}%'
    else:
        text_content = f'總收益率: {final_return:.2f}%'

    axes[0].text(
        0.02, 0.95,
        text_content,
        transform=axes[0].transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    # 子圖2: 價格和交易信號
    axes[1].plot(data.index, data['close'].values,
                 linewidth=1.5, color='#333', label='收盤價', alpha=0.7)

    # 標記做多信號
    long_signals = signals[signals == 1]
    if len(long_signals) > 0:
        axes[1].scatter(long_signals.index,
                       data.loc[long_signals.index, 'close'],
                       color='green', marker='^', s=100,
                       label='做多', alpha=0.7, edgecolors='darkgreen')

    # 標記做空信號
    short_signals = signals[signals == -1]
    if len(short_signals) > 0:
        axes[1].scatter(short_signals.index,
                       data.loc[short_signals.index, 'close'],
                       color='red', marker='v', s=100,
                       label='做空', alpha=0.7, edgecolors='darkred')

    axes[1].set_ylabel('價格 ($)', fontsize=12)
    axes[1].set_title('價格與交易信號', fontsize=12, pad=10)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    # 子圖3: 回撤曲線
    # ===== CRITICAL ISSUE FIX #2: Division by zero check in drawdown calculation =====
    initial_value = equity_curve.iloc[0]
    if initial_value == 0 or np.isnan(initial_value):
        warnings.warn("Cannot calculate drawdown with invalid initial value")
        cumulative_returns = pd.Series(1, index=equity_curve.index)
        drawdown = pd.Series(0, index=equity_curve.index)
    else:
        cumulative_returns = equity_curve / initial_value
        rolling_max = cumulative_returns.expanding().max()

        # ===== HIGH SEVERITY ISSUE FIX #5: NaN propagation in drawdown calculation =====
        # Handle division by zero in rolling_max
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdown = (cumulative_returns - rolling_max) / rolling_max

        # Replace any inf or NaN values with 0
        drawdown = drawdown.replace([np.inf, -np.inf], 0)
        drawdown = drawdown.fillna(0)

    axes[2].fill_between(drawdown.index, drawdown.values * 100, 0,
                         color='#2E86AB', alpha=0.3, label='策略回撤')
    axes[2].plot(drawdown.index, drawdown.values * 100,
                color='#2E86AB', linewidth=1.5)

    max_dd = drawdown.min() * 100
    axes[2].axhline(y=max_dd, color='#2E86AB', linestyle='--',
                   alpha=0.5, label=f'策略最大回撤: {max_dd:.2f}%')

    # 添加長期持有回撤對比
    if show_buy_and_hold:
        # ===== CRITICAL ISSUE FIX #2: Division by zero check for buy & hold drawdown =====
        bh_initial_value = buy_hold_equity.iloc[0]
        if bh_initial_value == 0 or np.isnan(bh_initial_value):
            warnings.warn("Cannot calculate buy & hold drawdown with invalid initial value")
        else:
            bh_cumulative_returns = buy_hold_equity / bh_initial_value
            bh_rolling_max = bh_cumulative_returns.expanding().max()

            # ===== HIGH SEVERITY ISSUE FIX #5: NaN propagation in drawdown calculation =====
            with np.errstate(divide='ignore', invalid='ignore'):
                bh_drawdown = (bh_cumulative_returns - bh_rolling_max) / bh_rolling_max

            # Replace any inf or NaN values with 0
            bh_drawdown = bh_drawdown.replace([np.inf, -np.inf], 0)
            bh_drawdown = bh_drawdown.fillna(0)

            axes[2].fill_between(bh_drawdown.index, bh_drawdown.values * 100, 0,
                                 color='orange', alpha=0.2, label='長期持有回撤')
            axes[2].plot(bh_drawdown.index, bh_drawdown.values * 100,
                        color='orange', linewidth=1.5, linestyle='--', alpha=0.7)

            bh_max_dd = bh_drawdown.min() * 100
            axes[2].axhline(y=bh_max_dd, color='orange', linestyle=':',
                           alpha=0.5, label=f'持有最大回撤: {bh_max_dd:.2f}%')

    axes[2].set_ylabel('回撤 (%)', fontsize=12)
    axes[2].set_xlabel('時間', fontsize=12)
    axes[2].set_title('回撤曲線對比', fontsize=12, pad=10)
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"圖表已保存: {save_path}")

    plt.show()


def plot_optimization_results(
    results_df: pd.DataFrame,
    param_names: List[str],
    metric: str = 'objective_value',
    top_n: int = 10,
    title: str = "參數優化結果",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    繪製參數優化結果

    包含多個子圖：
    1. Top N參數組合對比
    2. 參數敏感性分析（每個參數）
    3. 參數熱圖（如果是2D參數空間）

    Args:
        results_df: 優化結果DataFrame
        param_names: 參數名稱列表
        metric: 評估指標名稱
        top_n: 顯示前N個最佳結果
        title: 圖表標題
        save_path: 保存路徑
        figsize: 圖表大小

    Example:
        >>> plot_optimization_results(
        ...     results_df=optimizer.results_df,
        ...     param_names=['fast_period', 'slow_period'],
        ...     metric='sharpe_ratio'
        ... )
    """
    n_params = len(param_names)

    if n_params == 1:
        # 單參數優化：簡單折線圖
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        param = param_names[0]

        # 子圖1: 參數 vs 指標
        sorted_df = results_df.sort_values(param)
        axes[0].plot(sorted_df[param], sorted_df[metric],
                    marker='o', linewidth=2, markersize=8)
        axes[0].set_xlabel(param, fontsize=12)
        axes[0].set_ylabel(metric, fontsize=12)
        axes[0].set_title(f'{param} vs {metric}', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # 標記最佳點
        best_row = results_df.loc[results_df[metric].idxmax()]
        axes[0].scatter([best_row[param]], [best_row[metric]],
                       color='red', s=200, zorder=5,
                       label=f'最佳: {param}={best_row[param]:.0f}')
        axes[0].legend()

        # 子圖2: Top N對比
        top_df = results_df.nlargest(top_n, metric)
        axes[1].barh(range(len(top_df)), top_df[metric].values)
        axes[1].set_yticks(range(len(top_df)))
        axes[1].set_yticklabels([f"{row[param]:.0f}" for _, row in top_df.iterrows()])
        axes[1].set_xlabel(metric, fontsize=12)
        axes[1].set_ylabel(param, fontsize=12)
        axes[1].set_title(f'Top {top_n} 參數組合', fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='x')

    elif n_params == 2:
        # 雙參數優化：熱圖 + 散點圖
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # 創建3個子圖
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])  # 熱圖
        ax2 = fig.add_subplot(gs[1, 0])  # 參數1敏感性
        ax3 = fig.add_subplot(gs[1, 1])  # 參數2敏感性

        param1, param2 = param_names

        # 子圖1: 參數熱圖
        pivot_table = results_df.pivot_table(
            values=metric,
            index=param2,
            columns=param1,
            aggfunc='mean'
        )

        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn',
                   ax=ax1, cbar_kws={'label': metric})
        ax1.set_title(f'{metric} 熱圖', fontsize=12)
        ax1.set_xlabel(param1, fontsize=11)
        ax1.set_ylabel(param2, fontsize=11)

        # 子圖2: 參數1敏感性
        param1_sensitivity = results_df.groupby(param1)[metric].agg(['mean', 'std'])
        ax2.plot(param1_sensitivity.index, param1_sensitivity['mean'],
                marker='o', linewidth=2, markersize=8, label='平均')
        ax2.fill_between(
            param1_sensitivity.index,
            param1_sensitivity['mean'] - param1_sensitivity['std'],
            param1_sensitivity['mean'] + param1_sensitivity['std'],
            alpha=0.3, label='±1 std'
        )
        ax2.set_xlabel(param1, fontsize=11)
        ax2.set_ylabel(metric, fontsize=11)
        ax2.set_title(f'{param1} 敏感性分析', fontsize=11)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 子圖3: 參數2敏感性
        param2_sensitivity = results_df.groupby(param2)[metric].agg(['mean', 'std'])
        ax3.plot(param2_sensitivity.index, param2_sensitivity['mean'],
                marker='o', linewidth=2, markersize=8, label='平均')
        ax3.fill_between(
            param2_sensitivity.index,
            param2_sensitivity['mean'] - param2_sensitivity['std'],
            param2_sensitivity['mean'] + param2_sensitivity['std'],
            alpha=0.3, label='±1 std'
        )
        ax3.set_xlabel(param2, fontsize=11)
        ax3.set_ylabel(metric, fontsize=11)
        ax3.set_title(f'{param2} 敏感性分析', fontsize=11)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    else:
        # 多參數優化：平行座標圖 + Top N
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # 子圖1: Top N 參數組合
        top_df = results_df.nlargest(top_n, metric)
        x_pos = np.arange(len(top_df))

        axes[0].bar(x_pos, top_df[metric].values, color='steelblue')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([f"#{i+1}" for i in x_pos])
        axes[0].set_xlabel('排名', fontsize=12)
        axes[0].set_ylabel(metric, fontsize=12)
        axes[0].set_title(f'Top {top_n} 參數組合', fontsize=12)
        axes[0].grid(True, alpha=0.3, axis='y')

        # 子圖2: 參數對比
        top_df_normalized = top_df[param_names].copy()
        for param in param_names:
            pmin = top_df_normalized[param].min()
            pmax = top_df_normalized[param].max()
            if pmax > pmin:
                top_df_normalized[param] = (top_df_normalized[param] - pmin) / (pmax - pmin)

        for i, (_, row) in enumerate(top_df_normalized.iterrows()):
            axes[1].plot(param_names, row.values, marker='o',
                        label=f'#{i+1}', alpha=0.7, linewidth=2)

        axes[1].set_xlabel('參數', fontsize=12)
        axes[1].set_ylabel('標準化值', fontsize=12)
        axes[1].set_title('Top參數組合對比（標準化）', fontsize=12)
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"圖表已保存: {save_path}")

    plt.show()


def plot_mcpt_distribution(
    original_score: float,
    permutation_scores: np.ndarray,
    p_value: float,
    metric_name: str = "績效指標",
    title: str = "MCPT排列測試結果",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    繪製MCPT排列測試結果

    包含：
    1. 排列分佈直方圖
    2. 原始績效位置
    3. p值標記
    4. 統計信息

    Args:
        original_score: 原始策略績效
        permutation_scores: 所有排列的績效數組
        p_value: p值
        metric_name: 指標名稱
        title: 圖表標題
        save_path: 保存路徑
        figsize: 圖表大小

    Example:
        >>> plot_mcpt_distribution(
        ...     original_score=results['original_score'],
        ...     permutation_scores=results['permutation_scores'],
        ...     p_value=results['p_value'],
        ...     metric_name='夏普比率'
        ... )
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 繪製排列分佈直方圖
    n, bins, patches = ax.hist(
        permutation_scores,
        bins=50,
        density=True,
        alpha=0.7,
        color='skyblue',
        edgecolor='black',
        label='排列分佈'
    )

    # 添加核密度估計曲線
    from scipy import stats
    kde = stats.gaussian_kde(permutation_scores)
    x_range = np.linspace(permutation_scores.min(), permutation_scores.max(), 200)
    ax.plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE')

    # 標記原始績效
    ax.axvline(original_score, color='red', linestyle='--',
              linewidth=3, label=f'原始策略: {original_score:.4f}')

    # 標記平均值和中位數
    mean_score = np.mean(permutation_scores)
    median_score = np.median(permutation_scores)
    ax.axvline(mean_score, color='green', linestyle=':',
              linewidth=2, alpha=0.7, label=f'排列平均: {mean_score:.4f}')
    ax.axvline(median_score, color='orange', linestyle=':',
              linewidth=2, alpha=0.7, label=f'排列中位數: {median_score:.4f}')

    # 設置標籤
    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title(f'排列測試分佈 (p-value = {p_value:.4f})', fontsize=13, pad=10)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # 添加統計信息文本框
    is_significant = p_value < 0.05
    percentile = np.mean(permutation_scores < original_score) * 100

    stats_text = (
        f"統計摘要:\n"
        f"排列次數: {len(permutation_scores)}\n"
        f"排列平均: {mean_score:.4f}\n"
        f"排列標準差: {np.std(permutation_scores):.4f}\n"
        f"原始績效百分位: {percentile:.2f}%\n"
        f"p值: {p_value:.4f}\n"
        f"顯著性(α=0.05): {'是' if is_significant else '否'}"
    )

    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='lightyellow' if is_significant else 'lightcoral',
            alpha=0.8
        )
    )

    # 添加結論
    conclusion = (
        "結論: 策略顯著優於隨機" if is_significant
        else "結論: 策略未顯著優於隨機\n可能存在過度擬合"
    )

    ax.text(
        0.98, 0.02, conclusion,
        transform=ax.transAxes,
        fontsize=11,
        fontweight='bold',
        horizontalalignment='right',
        verticalalignment='bottom',
        bbox=dict(
            boxstyle='round',
            facecolor='lightgreen' if is_significant else 'salmon',
            alpha=0.8
        )
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"圖表已保存: {save_path}")

    plt.show()


def plot_mcpt_distribution_medium_style(
    original_score: float,
    permutation_scores: np.ndarray,
    p_value: float,
    metric_name: str = "Profit Factor",
    title: str = "MCPT Distribution",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    繪製MCPT排列測試結果 (Medium框架風格)

    Medium風格特徵：
    - 深色背景 (dark_background)
    - 簡潔設計，無過多裝飾
    - 英文標籤
    - 藍色直方圖 + 紅色真實策略線
    - 無網格線
    - 黑色背景保存

    Args:
        original_score: 原始策略績效
        permutation_scores: 所有排列的績效數組
        p_value: p值
        metric_name: 指標名稱（英文）
        title: 圖表標題（英文）
        save_path: 保存路徑
        figsize: 圖表大小

    Example:
        >>> plot_mcpt_distribution_medium_style(
        ...     original_score=2.5,
        ...     permutation_scores=np.array([1.2, 1.5, ...]),
        ...     p_value=0.023,
        ...     metric_name='Profit Factor',
        ...     save_path=Path("mcpt_distribution.png")
        ... )

    Raises:
        ValueError: If input validation fails
        TypeError: If input types are incorrect
    """
    # ===== CRITICAL ISSUE FIX #1: Input Validation =====
    # Validate permutation_scores
    if permutation_scores is None:
        raise ValueError("permutation_scores cannot be None")

    if not isinstance(permutation_scores, np.ndarray):
        raise TypeError(f"permutation_scores must be numpy.ndarray, got {type(permutation_scores)}")

    if len(permutation_scores) == 0:
        raise ValueError("permutation_scores cannot be empty")

    # Check for NaN/Inf values
    if np.isnan(permutation_scores).any():
        raise ValueError("permutation_scores contains NaN values")

    if np.isinf(permutation_scores).any():
        raise ValueError("permutation_scores contains Inf values")

    # Validate p_value range
    if not isinstance(p_value, (int, float)):
        raise TypeError(f"p_value must be numeric, got {type(p_value)}")

    if not (0 <= p_value <= 1):
        raise ValueError(f"p_value must be in range [0, 1], got {p_value}")

    # Validate original_score
    if not isinstance(original_score, (int, float)):
        raise TypeError(f"original_score must be numeric, got {type(original_score)}")

    if np.isnan(original_score) or np.isinf(original_score):
        raise ValueError(f"original_score contains invalid value: {original_score}")

    # ===== HIGH SEVERITY ISSUE FIX #5: Histogram Bin Validation =====
    # Calculate dynamic bins based on sample size and unique values
    n_samples = len(permutation_scores)
    n_unique = len(np.unique(permutation_scores))

    # Handle edge case: all values are the same
    if n_unique == 1:
        bins = 1
        warnings.warn("All permutation scores are identical - using single bin")
    else:
        # Use Sturges' rule as baseline, but adjust for sample size
        bins = min(int(np.log2(n_samples) + 1), n_unique, 50)
        bins = max(bins, 10)  # Ensure at least 10 bins for reasonable distribution

    # ===== CRITICAL ISSUE FIX #2: Style Conflict - Use context manager =====
    # Use context manager to prevent global style pollution
    with plt.style.context('dark_background'):
        try:
            fig, ax = plt.subplots(figsize=figsize)

            # 繪製排列分佈直方圖（藍色）
            ax.hist(
                permutation_scores,
                bins=bins,
                color='blue',
                alpha=0.7,
                label='Permutations',
                edgecolor='none'
            )

            # 標記真實策略績效（紅色垂直線）
            ax.axvline(
                original_score,
                color='red',
                linewidth=2,
                label='Real Strategy'
            )

            # 設置標籤和標題
            ax.set_xlabel(metric_name, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'{title}. P-Value: {p_value:.4f}', fontsize=13, pad=10)

            # 圖例
            ax.legend(loc='best', fontsize=10)

            # 無網格線（Medium風格）
            ax.grid(False)

            plt.tight_layout()

            # ===== HIGH SEVERITY ISSUE FIX #6: File Permission Error Handling =====
            if save_path:
                try:
                    # Create parent directory if not exists
                    save_path = Path(save_path)
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    # 使用黑色背景保存（Medium風格）
                    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
                    print(f"Chart saved: {save_path}")

                except PermissionError as e:
                    warnings.warn(f"Permission denied when saving to {save_path}: {e}")
                    print(f"Error: Cannot save chart due to permission error: {e}")
                except OSError as e:
                    warnings.warn(f"OS error when saving to {save_path}: {e}")
                    print(f"Error: Cannot save chart due to OS error: {e}")
                except Exception as e:
                    warnings.warn(f"Unexpected error when saving to {save_path}: {e}")
                    print(f"Error: Unexpected error while saving chart: {e}")

        finally:
            # Ensure plt.close() is called even if plotting fails
            plt.close()


def plot_optimization_results_medium_style(
    results_df: pd.DataFrame,
    param_names: List[str],
    output_dir: Path,
    metrics: List[str] = ['final_value', 'sharpe_ratio', 'profit_factor']
):
    """
    繪製參數優化結果 (Medium框架風格)

    為每個參數生成獨立的圖片，每張圖包含3個子圖：
    1. 參數 vs Final Value (Max + Mean)
    2. 參數 vs Sharpe Ratio (Max + Mean)
    3. 參數 vs Profit Factor (Max + Mean)

    Args:
        results_df: 優化結果DataFrame
        param_names: 參數名稱列表
        output_dir: 保存目錄
        metrics: 要顯示的指標列表

    Example:
        >>> plot_optimization_results_medium_style(
        ...     results_df=optimizer.results_df,
        ...     param_names=['lookback'],
        ...     output_dir=Path("results/optimization")
        ... )
    """
    # Critical Bug Fix 1: Add empty DataFrame validation
    if results_df is None or results_df.empty:
        raise ValueError("results_df cannot be None or empty")

    # Medium Bug Fix 6: Add data type validation
    if not isinstance(results_df, pd.DataFrame):
        raise TypeError(f"results_df must be a pandas DataFrame, got {type(results_df)}")

    if not isinstance(param_names, list) or len(param_names) == 0:
        raise ValueError("param_names must be a non-empty list")

    output_dir.mkdir(parents=True, exist_ok=True)

    for param in param_names:
        if param not in results_df.columns:
            continue

        # Critical Bug Fix 3: Filter metrics first before creating subplots
        available_metrics = [m for m in metrics if m in results_df.columns]

        if len(available_metrics) == 0:
            print(f"警告: 參數 {param} 沒有可用的指標，跳過繪圖")
            continue

        # Critical Bug Fix 2: Dynamic subplot creation based on available metrics
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))

        # Handle single metric case (axes is not an array)
        if n_metrics == 1:
            axes = [axes]

        fig.suptitle(f'{param.replace("_", " ").title()} vs Performance Metrics', fontsize=16)

        fig_created = False
        try:
            for i, metric in enumerate(available_metrics):
                ax = axes[i]

                # Group by parameter and calculate max, mean, min for each metric
                grouped = results_df.groupby(param)[metric].agg(['max', 'mean', 'min']).reset_index()

                # Plot max values (best performance for each parameter)
                ax.plot(grouped[param], grouped['max'], 'o-', linewidth=2, markersize=8,
                       label='Best Performance', color='blue')

                # Plot mean values
                ax.plot(grouped[param], grouped['mean'], 's--', linewidth=1, markersize=6,
                       label='Average Performance', color='gray', alpha=0.7)

                ax.set_xlabel(param.replace('_', ' ').title())
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(metric.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)

                # Medium Bug Fix 4: Add NaN/Inf value handling when finding global best
                valid_max = grouped['max'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid_max) > 0:
                    global_best_idx = valid_max.idxmax()
                    best_x = grouped.loc[global_best_idx, param]
                    best_y = grouped.loc[global_best_idx, 'max']
                    ax.scatter(best_x, best_y, color='red', s=150, zorder=5,
                              label=f'Global Best: {best_x}', marker='*')
                ax.legend()

            plt.tight_layout()
            fig_created = True

            # Medium Bug Fix 7: Add file I/O error handling with try-catch
            output_path = output_dir / f'{param}_comparison.png'
            try:
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            except (IOError, OSError) as e:
                print(f"Error: Cannot save chart to {output_path}: {e}")
            except Exception as e:
                print(f"Error: Unexpected error while saving chart: {e}")
        finally:
            # Minor Bug Fix 8: Add resource leak protection with finally block
            if fig_created:
                plt.close()


def plot_walkforward_performance(
    results_df: pd.DataFrame,
    title: str = "Walk-Forward分析結果",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10),
    combined_equity_curve: Optional[pd.Series] = None,
    buy_hold_equity_curve: Optional[pd.Series] = None
):
    """
    繪製Walk-Forward分析的權益與回撤曲線

    包含2個子圖：
    1. 策略權益曲線 vs 長期持有
    2. 策略回撤曲線 vs 長期持有回撤

    Args:
        results_df: Walk-Forward結果DataFrame (用於統計數據)
        title: 圖表標題
        save_path: 保存路徑
        figsize: 圖表大小
        combined_equity_curve: 合併的OOS權益曲線
        buy_hold_equity_curve: 長期持有權益曲線
    """
    # We must have equity curve data to proceed
    if combined_equity_curve is None or combined_equity_curve.empty:
        print("警告: 缺少權益曲線數據 (combined_equity_curve)，無法繪製圖表。")
        return

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Subplot 1: Equity Curve Comparison
    ax1 = axes[0]
    ax1.plot(combined_equity_curve.index, combined_equity_curve.values,
                 linewidth=2, color='#2E86AB', label='策略權益')

    if buy_hold_equity_curve is not None and not buy_hold_equity_curve.empty:
        ax1.plot(buy_hold_equity_curve.index, buy_hold_equity_curve.values,
                     linewidth=2, color='orange', label='長期持有', alpha=0.7, linestyle='--')

    ax1.axhline(y=combined_equity_curve.iloc[0], color='gray',
                    linestyle='--', alpha=0.5, label='初始資金')
    ax1.set_ylabel('權益 ($)', fontsize=12)
    ax1.set_title('Walk-Forward 權益曲線比較', fontsize=12, pad=10)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Add final return text
    final_return = (combined_equity_curve.iloc[-1] / combined_equity_curve.iloc[0] - 1) * 100
    text_content = f'策略最終收益率: {final_return:.2f}%'
    if buy_hold_equity_curve is not None and not buy_hold_equity_curve.empty:
        bh_final_return = (buy_hold_equity_curve.iloc[-1] / buy_hold_equity_curve.iloc[0] - 1) * 100
        text_content += f'\n長期持有收益率: {bh_final_return:.2f}%'

    ax1.text(
        0.02, 0.95, text_content,
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    # Subplot 2: Drawdown Curve Comparison
    ax2 = axes[1]
    
    # Calculate strategy drawdown
    if combined_equity_curve.iloc[0] == 0 or np.isnan(combined_equity_curve.iloc[0]):
        drawdown = pd.Series(0, index=combined_equity_curve.index)
    else:
        cumulative_returns = combined_equity_curve / combined_equity_curve.iloc[0]
        rolling_max = cumulative_returns.expanding().max()
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdown = (cumulative_returns - rolling_max) / rolling_max
        drawdown = drawdown.replace([np.inf, -np.inf], 0).fillna(0)

    ax2.fill_between(drawdown.index, drawdown.values * 100, 0,
                         color='#2E86AB', alpha=0.3, label='策略回撤')
    ax2.plot(drawdown.index, drawdown.values * 100,
                color='#2E86AB', linewidth=1.5)
    max_dd = drawdown.min() * 100
    ax2.axhline(y=max_dd, color='#2E86AB', linestyle='--',
                   alpha=0.5, label=f'策略最大回撤: {max_dd:.2f}%')

    # Calculate Buy & Hold drawdown
    if buy_hold_equity_curve is not None and not buy_hold_equity_curve.empty:
        if buy_hold_equity_curve.iloc[0] == 0 or np.isnan(buy_hold_equity_curve.iloc[0]):
            bh_drawdown = pd.Series(0, index=buy_hold_equity_curve.index)
        else:
            bh_cumulative_returns = buy_hold_equity_curve / buy_hold_equity_curve.iloc[0]
            bh_rolling_max = bh_cumulative_returns.expanding().max()
            with np.errstate(divide='ignore', invalid='ignore'):
                bh_drawdown = (bh_cumulative_returns - bh_rolling_max) / bh_rolling_max
            bh_drawdown = bh_drawdown.replace([np.inf, -np.inf], 0).fillna(0)

        ax2.fill_between(bh_drawdown.index, bh_drawdown.values * 100, 0,
                             color='orange', alpha=0.2, label='長期持有回撤')
        ax2.plot(bh_drawdown.index, bh_drawdown.values * 100,
                    color='orange', linewidth=1.5, linestyle='--', alpha=0.7)
        bh_max_dd = bh_drawdown.min() * 100
        ax2.axhline(y=bh_max_dd, color='orange', linestyle=':',
                       alpha=0.5, label=f'持有最大回撤: {bh_max_dd:.2f}%')

    ax2.set_ylabel('回撤 (%)', fontsize=12)
    ax2.set_xlabel('時間', fontsize=12)
    ax2.set_title('Walk-Forward 回撤曲線比較', fontsize=12, pad=10)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"圖表已保存: {save_path}")

    plt.show()


# ==================== 輔助函數 ====================

def save_all_plots(
    backtest_results: Optional[Dict] = None,
    optimization_results: Optional[Dict] = None,
    mcpt_results: Optional[Dict] = None,
    walkforward_results: Optional[Dict] = None,
    output_dir: Path = Path("results/plots")
):
    """
    一次性保存所有圖表

    Args:
        backtest_results: 回測結果字典
        optimization_results: 優化結果字典
        mcpt_results: MCPT結果字典
        walkforward_results: Walk-Forward結果字典
        output_dir: 輸出目錄
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if backtest_results:
        plot_backtest_results(
            equity_curve=backtest_results['equity_curve'],
            data=backtest_results['data'],
            signals=backtest_results['signals'],
            save_path=output_dir / "backtest.png"
        )

    if optimization_results:
        plot_optimization_results(
            results_df=optimization_results['results_df'],
            param_names=optimization_results['param_names'],
            save_path=output_dir / "optimization.png"
        )

    if mcpt_results:
        plot_mcpt_distribution(
            original_score=mcpt_results['original_score'],
            permutation_scores=np.array(mcpt_results['permutation_scores']),
            p_value=mcpt_results['p_value'],
            save_path=output_dir / "mcpt.png"
        )

    if walkforward_results:
        plot_walkforward_performance(
            results_df=walkforward_results['results_df'],
            save_path=output_dir / "walkforward.png"
        )

    print(f"\n所有圖表已保存到: {output_dir}")


if __name__ == "__main__":
    # 測試可視化模塊
    print("可視化模塊測試\n")

    # 創建測試數據
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=500, freq='1H')

    returns = np.random.normal(0.0005, 0.02, 500)
    prices = 30000 * (1 + returns).cumprod()

    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 500)),
        'high': prices * (1 + np.abs(np.random.normal(0.002, 0.001, 500))),
        'low': prices * (1 - np.abs(np.random.normal(0.002, 0.001, 500))),
        'close': prices,
        'volume': np.random.uniform(100, 1000, 500)
    }, index=dates)

    # 測試1: 回測結果可視化
    print("=" * 70)
    print("測試1: 回測結果可視化")
    print("=" * 70)

    # 創建模擬信號和權益曲線
    signals = pd.Series(0, index=data.index)
    ma = data['close'].rolling(20).mean()
    signals[data['close'] > ma] = 1
    signals[data['close'] < ma] = -1

    strategy_returns = signals.shift(1) * data['close'].pct_change()
    equity_curve = 100000 * (1 + strategy_returns.fillna(0)).cumprod()

    plot_backtest_results(
        equity_curve=equity_curve,
        data=data,
        signals=signals,
        title="測試策略回測結果"
    )

    # 測試2: 優化結果可視化（2D參數空間）
    print("\n" + "=" * 70)
    print("測試2: 參數優化結果可視化")
    print("=" * 70)

    # 創建模擬優化結果
    opt_results = []
    for fast in [5, 10, 15, 20]:
        for slow in [20, 30, 40, 50]:
            opt_results.append({
                'fast_period': fast,
                'slow_period': slow,
                'sharpe_ratio': np.random.normal(1.0, 0.5)
            })

    opt_df = pd.DataFrame(opt_results)

    plot_optimization_results(
        results_df=opt_df,
        param_names=['fast_period', 'slow_period'],
        metric='sharpe_ratio',
        title="測試參數優化"
    )

    # 測試3: MCPT結果可視化
    print("\n" + "=" * 70)
    print("測試3: MCPT結果可視化")
    print("=" * 70)

    original = 1.5
    permutations = np.random.normal(0.8, 0.4, 1000)
    p_val = np.mean(permutations >= original)

    plot_mcpt_distribution(
        original_score=original,
        permutation_scores=permutations,
        p_value=p_val,
        metric_name="夏普比率",
        title="測試MCPT分佈"
    )

    # 測試4: Walk-Forward結果可視化
    print("\n" + "=" * 70)
    print("測試4: Walk-Forward結果可視化")
    print("=" * 70)

    # 創建模擬Walk-Forward結果
    wf_results = []
    for i in range(10):
        wf_results.append({
            'window_id': i,
            'is_sharpe_ratio': np.random.normal(1.2, 0.3),
            'oos_sharpe_ratio': np.random.normal(0.9, 0.4),
            'oos_total_return': np.random.normal(0.05, 0.03),
            'oos_win_rate': np.random.normal(0.52, 0.05),
            'oos_profit_factor': np.random.normal(1.3, 0.4)
        })

    wf_df = pd.DataFrame(wf_results)

    plot_walkforward_performance(
        results_df=wf_df,
        title="測試Walk-Forward分析"
    )

    print("\n所有可視化測試完成!")
