"""
Monte Carlo Permutation Test (MCPT)
蒙地卡羅排列測試 - 驗證策略是否存在過度擬合
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Callable, Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from tqdm import tqdm
import os

from .permutation import advanced_permutation
from .backtest import quick_backtest
from .optimizer import ParameterOptimizer, convert_numpy_types


class MonteCarloTester:
    """
    蒙地卡羅排列測試器
    """

    def __init__(
        self,
        strategy_func: Callable,
        data: pd.DataFrame,
        strategy_params: Dict,
        start_index: int = 100,
        n_permutations: int = 1000,
        mode: str = 'normal',
        metric: str = 'sharpe_ratio',
        optimization_method: str = 'grid', # 新增
        n_iter: int = 100, # 新增
        transaction_cost: float = 0.0006,
        slippage: float = 0.0001,
        n_jobs: int = -1,
        periods_per_year: int = 252
    ):
        self.strategy_func = strategy_func
        self.data = data.copy()
        self.strategy_params = strategy_params
        self.start_index = start_index
        self.n_permutations = n_permutations
        self.mode = mode
        self.metric = metric
        self.optimization_method = optimization_method
        self.n_iter = n_iter
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.n_jobs = n_jobs
        self.periods_per_year = periods_per_year

        if mode not in ['normal', 'walkforward']:
            raise ValueError("mode必須是'normal'或'walkforward'")

        self.original_score = None
        self.permutation_scores = None
        self.p_value = None
        self.test_result = None

    def _evaluate_original(self) -> float:
        """評估原始數據的績效"""
        if self.mode == 'normal':
            signals = self.strategy_func(self.data, **self.strategy_params)
            _, metrics = quick_backtest(self.data, signals, self.transaction_cost, self.slippage, self.periods_per_year)
            return metrics[self.metric]
        else:  # walkforward模式
            optimizer = ParameterOptimizer(
                strategy_func=self.strategy_func,
                data=self.data,
                param_grid=self.strategy_params,
                objective=self.metric,
                method=self.optimization_method,
                n_iter=self.n_iter,
                transaction_cost=self.transaction_cost,
                slippage=self.slippage,
                n_jobs=self.n_jobs,
                periods_per_year=self.periods_per_year
            )
            optimizer.optimize(verbose=False)
            best_params = optimizer.get_best_params()
            signals = self.strategy_func(self.data, **best_params)
            _, metrics = quick_backtest(self.data, signals, self.transaction_cost, self.slippage, self.periods_per_year)
            return metrics[self.metric]

    def _evaluate_permutation(self, seed: int) -> float:
        """評估單個排列的績效"""
        try:
            perm_data = advanced_permutation(self.data, start_index=self.start_index, seed=seed)
            if self.mode == 'normal':
                signals = self.strategy_func(perm_data, **self.strategy_params)
                _, metrics = quick_backtest(perm_data, signals, self.transaction_cost, self.slippage, self.periods_per_year)
                return metrics[self.metric]
            else:  # walkforward模式
                optimizer = ParameterOptimizer(
                    strategy_func=self.strategy_func,
                    data=perm_data,
                    param_grid=self.strategy_params,
                    objective=self.metric,
                    method=self.optimization_method,
                    n_iter=self.n_iter,
                    n_jobs=1, # 在並行環境中，每個排列使用單進程優化
                    periods_per_year=self.periods_per_year
                )
                optimizer.optimize(verbose=False)
                best_params = optimizer.get_best_params()
                signals = self.strategy_func(perm_data, **best_params)
                _, metrics = quick_backtest(perm_data, signals, self.transaction_cost, self.slippage, self.periods_per_year)
                return metrics[self.metric]
        except Exception as e:
            return -np.inf

    def run(self, verbose: bool = True) -> Dict:
        """執行MCPT測試"""
        if verbose:
            print("\n" + "=" * 70)
            print("Running Monte Carlo Permutation Test (MCPT)")
            print(f"Mode: {self.mode}, Method: {self.optimization_method if self.mode == 'walkforward' else 'N/A'}")
            print("=" * 70)

        self.original_score = self._evaluate_original()
        if verbose:
            print(f"Original {self.metric}: {self.original_score:.4f}")

        permutation_scores = []
        with ProcessPoolExecutor(max_workers=self.n_jobs if self.n_jobs > 0 else None) as executor:
            futures = {executor.submit(self._evaluate_permutation, seed) for seed in range(self.n_permutations)}
            iterator = tqdm(as_completed(futures), total=self.n_permutations, desc="Permutation Progress", disable=os.environ.get('TQDM_DISABLE') == '1') if verbose else as_completed(futures)
            for future in iterator:
                permutation_scores.append(future.result())
        
        self.permutation_scores = np.array(permutation_scores)
        self.p_value = np.mean(self.permutation_scores >= self.original_score)
        percentile = np.mean(self.permutation_scores < self.original_score) * 100
        is_significant = self.p_value < 0.05

        self.test_result = {
            'original_score': self.original_score,
            'permutation_scores': self.permutation_scores.tolist(),
            'p_value': self.p_value,
            'is_significant': is_significant,
            'percentile': percentile
        }
        if verbose:
            self.print_results()
        return self.test_result

    def print_results(self):
        if self.test_result is None:
            raise RuntimeError("Please run() first.")
        print("\n" + "=" * 70)
        print("MCPT Results")
        print("=" * 70)
        print(f"p-value: {self.p_value:.4f}")
        print(f"Significance (alpha=0.05): {'Yes' if self.test_result['is_significant'] else 'No'}")
        print("=" * 70)

    def save_results(self, data_source: str, output_dir: Path) -> Optional[Path]:
        """
        保存MCPT測試結果到指定目錄

        Args:
            data_source (str): 數據源名稱，用於創建子目錄
            output_dir (Path): 結果保存的根目錄

        Returns:
            Optional[Path]: 成功時返回保存路徑，失敗時返回None
        """
        if self.test_result is None:
            print("錯誤: 測試結果尚未生成，請先運行 .run() 方法。")
            return None

        try:
            # 創建保存路徑
            save_path = output_dir / data_source
            save_path.mkdir(parents=True, exist_ok=True)

            # 保存結果到JSON文件
            results_path = save_path / "mcpt_results.json"
            
            # 使用自定義轉換器處理numpy類型
            serializable_results = convert_numpy_types(self.test_result)

            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=4, ensure_ascii=False)
            
            print(f"MCPT結果已成功保存到: {results_path}")
            return save_path

        except (IOError, OSError, json.JSONDecodeError) as e:
            print(f"錯誤: 保存MCPT結果失敗: {e}")
            return None