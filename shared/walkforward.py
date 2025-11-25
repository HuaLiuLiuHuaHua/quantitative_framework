"""
Walk-Forward Analysis
Walk-Forward分析器 - 模擬實盤的滾動優化和樣本外測試
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Callable, List, Optional
import json
from tqdm import tqdm
import os

from .optimizer import ParameterOptimizer, convert_numpy_types
from .backtest import quick_backtest

class WalkForwardAnalyzer:
    """
    Walk-Forward分析器
    """

    def __init__(
        self,
        strategy_func: Callable,
        data: pd.DataFrame,
        param_grid: Dict[str, List],
        train_window: int = 1000,
        test_window: int = 250,
        step: Optional[int] = None,
        objective: str = 'sharpe_ratio',
        optimization_method: str = 'grid', # 新增：優化方法
        n_iter: int = 100, # 新增：隨機搜索迭代次數
        transaction_cost: float = 0.0006,
        slippage: float = 0.0001,
        n_jobs: int = -1,
        periods_per_year: int = 252
    ):
        self.strategy_func = strategy_func
        self.data = data.copy()
        self.param_grid = param_grid
        self.train_window = train_window
        self.test_window = test_window
        self.step = step if step is not None else test_window
        self.objective = objective
        self.optimization_method = optimization_method
        self.n_iter = n_iter
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.n_jobs = n_jobs
        self.periods_per_year = periods_per_year

        if train_window + test_window > len(data):
            raise ValueError(f"訓練窗口({train_window}) + 測試窗口({test_window}) 大於數據長度({len(data)})")

        self.windows = []
        self.results_df = None
        self.combined_metrics = None
        self.combined_equity_curve = None
        self.buy_hold_equity_curve = None

    def _generate_windows(self) -> List[Dict]:
        windows = []
        current_pos = 0
        while current_pos + self.train_window + self.test_window <= len(self.data):
            train_start = current_pos
            train_end = current_pos + self.train_window
            test_start = train_end
            test_end = test_start + self.test_window
            window = {
                'window_id': len(windows),
                'train_start_date': self.data.index[train_start],
                'train_end_date': self.data.index[train_end - 1],
                'test_start_date': self.data.index[test_start],
                'test_end_date': self.data.index[test_end - 1]
            }
            windows.append(window)
            current_pos += self.step
        return windows

    def _process_window(self, window_info: Dict) -> Dict:
        train_data = self.data.loc[window_info['train_start_date']:window_info['train_end_date']]
        test_data = self.data.loc[window_info['test_start_date']:window_info['test_end_date']]

        optimizer = ParameterOptimizer(
            strategy_func=self.strategy_func,
            data=train_data,
            param_grid=self.param_grid,
            objective=self.objective,
            method=self.optimization_method, # 傳遞優化方法
            n_iter=self.n_iter, # 傳遞迭代次數
            transaction_cost=self.transaction_cost,
            slippage=self.slippage,
            n_jobs=self.n_jobs,
            periods_per_year=self.periods_per_year
        )

        optimizer.optimize(verbose=False)
        best_params = optimizer.get_best_params()

        if not best_params:
            return {**window_info, 'error': 'Optimization failed to find best parameters.'}

        _, train_metrics = quick_backtest(train_data, self.strategy_func(train_data, **best_params), self.transaction_cost, self.slippage, self.periods_per_year)
        test_returns, test_metrics = quick_backtest(test_data, self.strategy_func(test_data, **best_params), self.transaction_cost, self.slippage, self.periods_per_year)

        initial_capital = 10000.0
        test_equity_curve = initial_capital * (1 + test_returns).cumprod()

        result = {
            **window_info,
            'optimal_params': best_params,
            'is_sharpe_ratio': train_metrics.get('sharpe_ratio'),
            'oos_sharpe_ratio': test_metrics.get('sharpe_ratio'),
            'oos_total_return': test_metrics.get('total_return'),
            'oos_max_drawdown': test_metrics.get('max_drawdown'),
            'oos_profit_factor': test_metrics.get('profit_factor'),
            'oos_win_rate': test_metrics.get('win_rate'),
            'test_equity_curve': test_equity_curve,
            'test_data': test_data
        }
        return result

    def run(self, verbose: bool = True) -> pd.DataFrame:
        self.windows = self._generate_windows()
        if not self.windows:
            raise ValueError("無法生成任何窗口，請檢查窗口大小設置")

        if verbose:
            results = [self._process_window(w) for w in tqdm(self.windows, desc="Processing Walk-Forward Windows", disable=os.environ.get('TQDM_DISABLE') == '1')]
        else:
            results = [self._process_window(w) for w in self.windows]

        self.results_df = pd.DataFrame(results)
        self._combine_equity_curves(results)
        self._calculate_combined_metrics()

        if verbose:
            self.print_summary()

        return self.results_df

    def _combine_equity_curves(self, results: List[Dict]):
        if not results:
            return

        all_equity_curves = [r['test_equity_curve'] for r in results if 'test_equity_curve' in r and not r['test_equity_curve'].empty]
        if not all_equity_curves:
            return

        # 連接所有OOS窗口的權益曲線
        start_capital = all_equity_curves[0].iloc[0]
        combined_equity = pd.Series(dtype=float)
        for curve in all_equity_curves:
            if combined_equity.empty:
                combined_equity = curve
            else:
                next_curve_scaled = curve * (combined_equity.iloc[-1] / curve.iloc[0])
                combined_equity = pd.concat([combined_equity, next_curve_scaled.iloc[1:]])
        
        self.combined_equity_curve = combined_equity

        # 計算長期持有基準
        full_test_period_data = pd.concat([r['test_data'] for r in results if 'test_data' in r])
        full_test_period_data = full_test_period_data[~full_test_period_data.index.duplicated(keep='first')]
        buy_hold_returns = full_test_period_data['close'].pct_change().fillna(0)
        self.buy_hold_equity_curve = start_capital * (1 + buy_hold_returns).cumprod()
        self.buy_hold_equity_curve = self.buy_hold_equity_curve.reindex(self.combined_equity_curve.index, method='ffill')

    def _calculate_combined_metrics(self):
        if self.results_df is None or self.results_df.empty:
            return
        self.combined_metrics = {
            'avg_oos_sharpe': self.results_df['oos_sharpe_ratio'].mean(),
            'consistency': (self.results_df['oos_total_return'] > 0).mean() if not self.results_df['oos_total_return'].empty else 0,
        }

    def print_summary(self):
        if self.combined_metrics is None:
            return
        print("\n" + "=" * 70)
        print("Walk-Forward Analysis Summary")
        print("=" * 70)
        print(f"Total Windows: {len(self.results_df)}")
        print(f"Consistency (Positive Return Windows): {self.combined_metrics['consistency'] * 100:.2f}%")
        print(f"Average Out-of-Sample Sharpe Ratio: {self.combined_metrics['avg_oos_sharpe']:.3f}")
        print("=" * 70)

    def save_results(self, data_source: str = "unknown", output_dir: Optional[Path] = None) -> Path:
        if self.results_df is None:
            raise RuntimeError("Please run optimize() first.")
        if output_dir is None:
            output_dir = Path.cwd() / "results"
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = output_dir / f"WalkForward_{data_source}_{self.objective}_{date_str}"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # ... (save logic) ...
        print(f"\nWalk-Forward results saved to: {result_dir}")
        return result_dir