"""
Quantitative Framework - Shared Module
共享核心功能模塊

包含以下子模塊：
- base_strategy: 策略基類
- metrics: 績效指標計算
- backtest: 回測引擎
- permutation: K線排列邏輯（MCPT-Main）
- optimizer: 並行參數優化
- mcpt: 蒙地卡羅排列測試
- walkforward: Walk-Forward分析
- visualization: 繪圖功能
"""

__version__ = "1.0.0"

# 導入核心類和函數
from .base_strategy import BaseStrategy
from .metrics import (
    calculate_all_metrics,
    calculate_sharpe_ratio,
    calculate_profit_factor,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    print_metrics
)
from .backtest import BacktestEngine, quick_backtest
from .permutation import advanced_permutation, simple_permutation, validate_ohlc
from .optimizer import ParameterOptimizer, quick_optimize
from .mcpt import MonteCarloTester
from .walkforward import WalkForwardAnalyzer
from .visualization import (
    plot_backtest_results,
    plot_optimization_results,
    plot_mcpt_distribution,
    plot_walkforward_performance,
    save_all_plots
)

__all__ = [
    # 基類
    'BaseStrategy',

    # 指標計算
    'calculate_all_metrics',
    'calculate_sharpe_ratio',
    'calculate_profit_factor',
    'calculate_calmar_ratio',
    'calculate_max_drawdown',
    'print_metrics',

    # 回測
    'BacktestEngine',
    'quick_backtest',

    # 排列
    'advanced_permutation',
    'simple_permutation',
    'validate_ohlc',

    # 優化
    'ParameterOptimizer',
    'quick_optimize',

    # MCPT
    'MonteCarloTester',

    # Walk-Forward
    'WalkForwardAnalyzer',

    # 可視化
    'plot_backtest_results',
    'plot_optimization_results',
    'plot_mcpt_distribution',
    'plot_walkforward_performance',
    'save_all_plots',
]
