"""
Parameter Optimizer
- Supports grid search and random search for both vectorized and event-driven backends.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Callable, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import json
import random
from tqdm import tqdm
import os

from .backtest import quick_backtest, EventDrivenBacktestEngine
from .base_strategy import BaseStrategy


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.datetime64, pd.Timestamp)):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

# --- TOP-LEVEL OBJECTIVE FUNCTIONS (Required for multiprocessing) ---
def _objective_sharpe_ratio(metrics: Dict) -> float:
    return metrics.get('sharpe_ratio', -np.inf)

def _objective_profit_factor(metrics: Dict) -> float:
    return metrics.get('profit_factor', -np.inf)

def _objective_calmar_ratio(metrics: Dict) -> float:
    return metrics.get('calmar_ratio', -np.inf)

def _objective_total_return(metrics: Dict) -> float:
    return metrics.get('total_return', -np.inf)

def _objective_max_drawdown(metrics: Dict) -> float:
    # We want to MINIMIZE drawdown, so we MAXIMIZE the negative drawdown
    return -metrics.get('max_drawdown', np.inf)

def _get_objective_function(objective: str) -> Callable:
    objective_map = {
        'sharpe_ratio': _objective_sharpe_ratio,
        'profit_factor': _objective_profit_factor,
        'calmar_ratio': _objective_calmar_ratio,
        'total_return': _objective_total_return,
        'max_drawdown': _objective_max_drawdown,
    }
    if objective not in objective_map:
        raise ValueError(f"Unknown objective: {objective}. Available: {list(objective_map.keys())}")
    return objective_map[objective]

# --- Worker for Vectorized Optimizer ---
def _worker_evaluate_params_vectorized(strategy_func, data, params, transaction_cost, slippage, periods_per_year, objective_func):
    try:
        signals = strategy_func(data, **params)
        if signals is None:
            return {**params, 'objective_value': -np.inf, 'error': 'Invalid parameter combination'}
        
        _, metrics = quick_backtest(
            data=data,
            signals=signals,
            transaction_cost=transaction_cost,
            slippage=slippage,
            periods_per_year=periods_per_year
        )
        objective_value = objective_func(metrics)
        result = {**params, 'objective_value': objective_value, **metrics}
        return result
    except Exception as e:
        return {**params, 'objective_value': -np.inf, 'error': str(e)}

# --- Original Vectorized Optimizer ---
class ParameterOptimizer:
    """Parameter optimizer for vectorized backtesting strategies."""
    def __init__(
        self,
        strategy_func: Callable,
        data: pd.DataFrame,
        param_grid: Dict[str, List],
        objective: str = 'sharpe_ratio',
        method: str = 'grid',
        n_iter: int = 100,
        transaction_cost: float = 0.0006,
        slippage: float = 0.0001,
        n_jobs: int = -1,
        periods_per_year: int = 252
    ):
        self.strategy_func = strategy_func
        self.data = data.copy()
        self.param_grid = param_grid
        self.objective_name = objective
        self.method = method
        self.n_iter = n_iter
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.n_jobs = n_jobs
        self.periods_per_year = periods_per_year
        self.objective_func = _get_objective_function(objective)
        self.param_combinations = self._generate_param_combinations()
        self.results_df = None
        self.best_params = None
        self.best_score = None

    def _generate_param_combinations(self) -> List[Dict]:
        if self.method == 'grid':
            keys, values = self.param_grid.keys(), self.param_grid.values()
            return [dict(zip(keys, v)) for v in product(*values)]
        elif self.method == 'random':
            keys, values = self.param_grid.keys(), self.param_grid.values()
            total_possible = np.prod([len(v) for v in values])
            n_iter = min(self.n_iter, total_possible)
            combinations = set()
            while len(combinations) < n_iter:
                combo = tuple(random.choice(v) for v in values)
                combinations.add(combo)
            return [dict(zip(keys, c)) for c in combinations]
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def optimize(self, verbose: bool = True) -> pd.DataFrame:
        total_combinations = len(self.param_combinations)
        if total_combinations == 0:
            self.results_df = pd.DataFrame()
            return self.results_df

        if verbose: print(f"Optimizing {total_combinations} combinations using '{self.method}' search...")
        
        results = []
        with ProcessPoolExecutor(max_workers=self.n_jobs if self.n_jobs > 0 else None) as executor:
            futures = {executor.submit(_worker_evaluate_params_vectorized, self.strategy_func, self.data, params, self.transaction_cost, self.slippage, self.periods_per_year, self.objective_func): params for params in self.param_combinations}
            iterator = tqdm(as_completed(futures), total=total_combinations, desc="Optimizing", disable=os.environ.get('TQDM_DISABLE') == '1') if verbose else as_completed(futures)
            for future in iterator:
                results.append(future.result())

        self.results_df = pd.DataFrame(results).sort_values('objective_value', ascending=False).reset_index(drop=True)
        if not self.results_df.empty and 'objective_value' in self.results_df.columns:
            valid_results = self.results_df.dropna(subset=['objective_value']).loc[np.isfinite(self.results_df['objective_value'])]
            if not valid_results.empty:
                best_row = valid_results.iloc[0]
                self.best_params = {key: convert_numpy_types(best_row.get(key)) for key in self.param_grid.keys()}
                self.best_score = convert_numpy_types(best_row.get('objective_value'))
        return self.results_df

    def save_results(self, data_source: str = "unknown", output_dir: Optional[Path] = None) -> Path:
        if self.results_df is None:
            raise RuntimeError("Please run optimize() first.")
        if output_dir is None:
            output_dir = Path.cwd() / "results"
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = output_dir / f"Optimization_{data_source}_{self.objective_name}_{date_str}" / "optimization"
        result_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            'data_source': data_source,
            'optimization_date': date_str,
            'method': self.method,
            'n_iter': self.n_iter if self.method == 'random' else None,
            'total_combinations_tested': len(self.param_combinations),
            'param_grid': self.param_grid,
            'objective': self.objective_name,
            'best_params': self.best_params,
            'best_score': self.best_score,
        }
        summary = convert_numpy_types(summary)
        with open(result_dir / 'optimization_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.results_df.to_csv(result_dir / 'optimization_results.csv', index=False)
        best_params_converted = convert_numpy_types(self.best_params)
        with open(result_dir / 'best_params.json', 'w', encoding='utf-8') as f:
            json.dump(best_params_converted, f, indent=2, ensure_ascii=False)
        print(f"\nOptimization results saved to: {result_dir}")
        return result_dir

    def get_best_params(self) -> Dict:
        return self.best_params or {}

# --- NEW COMPONENTS FOR EVENT-DRIVEN BACKTESTING ---

def _worker_evaluate_params_event_driven(StrategyClass, data, params, transaction_cost, slippage, periods_per_year, objective_func):
    try:
        strategy = StrategyClass(**params)
        lookback = strategy.get_lookback()
        engine = EventDrivenBacktestEngine(
            data=data,
            strategy=strategy,
            transaction_cost=transaction_cost,
            slippage=slippage,
            lookback=lookback
        )
        results = engine.run()
        metrics = results['metrics']
        objective_value = objective_func(metrics)
        serializable_metrics = convert_numpy_types(metrics)
        result = {**params, 'objective_value': objective_value, **serializable_metrics}
        return result
    except Exception as e:
        return {**params, 'objective_value': -np.inf, 'error': str(e)}

class EventDrivenParameterOptimizer(ParameterOptimizer):
    """Parameter optimizer for event-driven backtesting strategies."""
    def __init__(
        self,
        StrategyClass: type[BaseStrategy],
        data: pd.DataFrame,
        param_grid: Dict[str, List],
        **kwargs
    ):
        super().__init__(strategy_func=lambda: None, data=data, param_grid=param_grid, **kwargs)
        self.StrategyClass = StrategyClass

    def optimize(self, verbose: bool = True) -> pd.DataFrame:
        total_combinations = len(self.param_combinations)
        if total_combinations == 0:
            self.results_df = pd.DataFrame()
            return self.results_df

        if verbose: print(f"Optimizing {total_combinations} combinations for EVENT-DRIVEN strategy...")

        results = []
        with ProcessPoolExecutor(max_workers=self.n_jobs if self.n_jobs > 0 else None) as executor:
            futures = {executor.submit(_worker_evaluate_params_event_driven, self.StrategyClass, self.data, params, self.transaction_cost, self.slippage, self.periods_per_year, self.objective_func): params for params in self.param_combinations}
            iterator = tqdm(as_completed(futures), total=total_combinations, desc="Optimizing (Event-Driven)", disable=os.environ.get('TQDM_DISABLE') == '1') if verbose else as_completed(futures)
            for future in iterator:
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"A worker process failed: {e}")

        self.results_df = pd.DataFrame(results).sort_values('objective_value', ascending=False).reset_index(drop=True)
        if not self.results_df.empty and 'objective_value' in self.results_df.columns:
            valid_results = self.results_df.dropna(subset=['objective_value']).loc[np.isfinite(self.results_df['objective_value'])]
            if not valid_results.empty:
                best_row = valid_results.iloc[0]
                self.best_params = {key: convert_numpy_types(best_row.get(key)) for key in self.param_grid.keys()}
                self.best_score = convert_numpy_types(best_row.get('objective_value'))
        return self.results_df
