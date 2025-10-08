"""
Monte Carlo Permutation Test (MCPT)
蒙地卡羅排列測試 - 驗證策略是否存在過度擬合

理論基礎（White 2000）：
通過對歷史數據進行隨機排列，生成多個「假」的歷史序列，
然後在這些排列序列上運行相同的策略，比較原始績效與排列績效的分佈。

測試原理：
H0（虛無假設）：策略沒有預測能力，績效來自隨機性
- 如果原始績效顯著優於排列績效 (p < 0.05)，則拒絕H0，策略有效
- 如果原始績效在排列分佈中（p > 0.05），則策略可能過度擬合

兩種模式：
1. Normal MCPT：使用固定參數在排列數據上測試
2. Walk-Forward MCPT：每個排列都重新優化參數（更嚴格）

參考文獻：
White, H. (2000). "A Reality Check for Data Snooping." Econometrica, 68(5), 1097-1126.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Callable, Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from tqdm import tqdm

from .permutation import advanced_permutation
from .backtest import quick_backtest
from .optimizer import ParameterOptimizer


class MonteCarloTester:
    """
    蒙地卡羅排列測試器

    用於驗證策略的統計顯著性，避免過度擬合

    Attributes:
        strategy_func: 策略函數
        data: OHLCV數據
        strategy_params: 策略參數（Normal模式）或參數網格（Walk-Forward模式）
        start_index: 開始排列的索引（之前的數據用於指標計算）
        n_permutations: 排列次數
        mode: 測試模式（'normal' 或 'walkforward'）
        metric: 評估指標
        n_jobs: 並行進程數
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
        transaction_cost: float = 0.0006,
        slippage: float = 0.0001,
        n_jobs: int = -1,
        periods_per_year: int = 252
    ):
        """
        初始化MCPT測試器

        Args:
            strategy_func: 策略函數，格式為 func(data, **params) -> pd.Series
            data: OHLCV數據
            strategy_params:
                - Normal模式：固定參數字典，例如 {'period': 20}
                - Walk-Forward模式：參數網格，例如 {'period': [10, 20, 30]}
            start_index: 開始排列的索引（保留前面數據用於指標計算）
            n_permutations: 排列次數（建議1000+）
            mode: 測試模式
                - 'normal': 使用固定參數
                - 'walkforward': 每次排列重新優化參數（更嚴格）
            metric: 評估指標（'sharpe_ratio', 'profit_factor', 等）
            transaction_cost: 交易成本
            slippage: 滑價
            n_jobs: 並行進程數
            periods_per_year: 每年的時間週期數 (默認252)

        Example:
            >>> # Normal MCPT
            >>> tester = MonteCarloTester(
            ...     strategy_func=my_strategy,
            ...     data=df,
            ...     strategy_params={'period': 20},
            ...     n_permutations=1000,
            ...     mode='normal'
            ... )
            >>> results = tester.run()

            >>> # Walk-Forward MCPT（更嚴格）
            >>> tester = MonteCarloTester(
            ...     strategy_func=my_strategy,
            ...     data=df,
            ...     strategy_params={'period': [10, 20, 30, 40]},
            ...     n_permutations=500,
            ...     mode='walkforward'
            ... )
            >>> results = tester.run()
        """
        self.strategy_func = strategy_func
        self.data = data.copy()
        self.strategy_params = strategy_params
        self.start_index = start_index
        self.n_permutations = n_permutations
        self.mode = mode
        self.metric = metric
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.n_jobs = n_jobs
        self.periods_per_year = periods_per_year

        # 驗證模式
        if mode not in ['normal', 'walkforward']:
            raise ValueError("mode必須是'normal'或'walkforward'")

        # 結果存儲
        self.original_score = None
        self.permutation_scores = None
        self.p_value = None
        self.test_result = None

    def _evaluate_original(self) -> float:
        """評估原始數據的績效"""
        if self.mode == 'normal':
            # Normal模式：使用固定參數
            signals = self.strategy_func(self.data, **self.strategy_params)
            _, metrics = quick_backtest(
                data=self.data,
                signals=signals,
                transaction_cost=self.transaction_cost,
                slippage=self.slippage,
                periods_per_year=self.periods_per_year
            )
            return metrics[self.metric]

        else:  # walkforward模式
            # Walk-Forward模式：先優化參數
            optimizer = ParameterOptimizer(
                strategy_func=self.strategy_func,
                data=self.data,
                param_grid=self.strategy_params,
                objective=self.metric,
                transaction_cost=self.transaction_cost,
                slippage=self.slippage,
                n_jobs=self.n_jobs,
                periods_per_year=self.periods_per_year
            )
            optimizer.optimize(verbose=False)

            # 使用最佳參數評估
            best_params = optimizer.get_best_params()
            signals = self.strategy_func(self.data, **best_params)
            _, metrics = quick_backtest(
                data=self.data,
                signals=signals,
                transaction_cost=self.transaction_cost,
                slippage=self.slippage,
                periods_per_year=self.periods_per_year
            )
            return metrics[self.metric]

    def _evaluate_permutation(self, seed: int) -> float:
        """
        評估單個排列的績效

        Args:
            seed: 隨機種子

        Returns:
            float: 排列數據的績效指標值
        """
        try:
            # 生成排列數據
            perm_data = advanced_permutation(
                self.data,
                start_index=self.start_index,
                seed=seed
            )

            if self.mode == 'normal':
                # Normal模式：使用相同參數
                signals = self.strategy_func(perm_data, **self.strategy_params)
                _, metrics = quick_backtest(
                    data=perm_data,
                    signals=signals,
                    transaction_cost=self.transaction_cost,
                    slippage=self.slippage,
                    periods_per_year=self.periods_per_year
                )
                return metrics[self.metric]

            else:  # walkforward模式
                # Walk-Forward模式：重新優化參數
                optimizer = ParameterOptimizer(
                    strategy_func=self.strategy_func,
                    data=perm_data,
                    param_grid=self.strategy_params,
                    objective=self.metric,
                    transaction_cost=self.transaction_cost,
                    slippage=self.slippage,
                    n_jobs=1,  # 在並行環境中，每個排列使用單進程優化
                    periods_per_year=self.periods_per_year
                )
                optimizer.optimize(verbose=False)

                best_params = optimizer.get_best_params()
                signals = self.strategy_func(perm_data, **best_params)
                _, metrics = quick_backtest(
                    data=perm_data,
                    signals=signals,
                    transaction_cost=self.transaction_cost,
                    slippage=self.slippage,
                    periods_per_year=self.periods_per_year
                )
                return metrics[self.metric]

        except Exception as e:
            # 如果排列失敗，返回極小值
            print(f"Warning: Permutation {seed} failed: {e}")
            return -np.inf

    def run(self, verbose: bool = True) -> Dict:
        """
        執行MCPT測試

        Returns:
            Dict: 測試結果字典，包含：
                - original_score: 原始績效
                - permutation_scores: 所有排列的績效列表
                - p_value: p值（原始績效在排列分佈中的位置）
                - is_significant: 是否顯著（p < 0.05）
                - percentile: 原始績效的百分位數
                - summary: 統計摘要

        Example:
            >>> results = tester.run()
            >>> print(f"p-value: {results['p_value']:.4f}")
            >>> print(f"是否顯著: {results['is_significant']}")
        """
        if verbose:
            print("\n" + "=" * 70)
            print("蒙地卡羅排列測試 (MCPT)")
            print("=" * 70)
            print(f"測試模式: {self.mode.upper()}")
            print(f"排列次數: {self.n_permutations}")
            print(f"評估指標: {self.metric}")
            print(f"開始索引: {self.start_index}")
            print(f"並行進程: {self.n_jobs if self.n_jobs > 0 else 'all'}")
            print("=" * 70)

        # 步驟1: 評估原始數據
        if verbose:
            print("\n[1/2] 評估原始數據績效...")

        self.original_score = self._evaluate_original()

        if verbose:
            print(f"原始{self.metric}: {self.original_score:.4f}")

        # 步驟2: 評估排列數據
        if verbose:
            print(f"\n[2/2] 執行{self.n_permutations}次排列測試...")

        permutation_scores = []

        if self.n_jobs == 1:
            # 單進程
            iterator = range(self.n_permutations)
            if verbose:
                iterator = tqdm(iterator, desc="排列進度")

            for seed in iterator:
                score = self._evaluate_permutation(seed)
                permutation_scores.append(score)

        else:
            # 多進程並行
            max_workers = None if self.n_jobs == -1 else self.n_jobs

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._evaluate_permutation, seed): seed
                    for seed in range(self.n_permutations)
                }

                if verbose:
                    iterator = tqdm(
                        as_completed(futures),
                        total=self.n_permutations,
                        desc="排列進度"
                    )
                else:
                    iterator = as_completed(futures)

                for future in iterator:
                    score = future.result()
                    permutation_scores.append(score)

        self.permutation_scores = np.array(permutation_scores)

        # 步驟3: 計算統計量
        # p值 = 排列績效 >= 原始績效的比例
        self.p_value = np.mean(self.permutation_scores >= self.original_score)

        # 計算百分位數
        percentile = np.mean(self.permutation_scores < self.original_score) * 100

        # 判斷顯著性
        is_significant = self.p_value < 0.05

        # 準備結果
        self.test_result = {
            'original_score': self.original_score,
            'permutation_scores': self.permutation_scores.tolist(),
            'p_value': self.p_value,
            'is_significant': is_significant,
            'percentile': percentile,
            'summary': {
                'mode': self.mode,
                'metric': self.metric,
                'n_permutations': self.n_permutations,
                'start_index': self.start_index,
                'perm_mean': float(np.mean(self.permutation_scores)),
                'perm_std': float(np.std(self.permutation_scores)),
                'perm_min': float(np.min(self.permutation_scores)),
                'perm_max': float(np.max(self.permutation_scores)),
                'perm_median': float(np.median(self.permutation_scores))
            }
        }

        if verbose:
            self.print_results()

        return self.test_result

    def print_results(self):
        """打印測試結果"""
        if self.test_result is None:
            raise RuntimeError("請先執行 run() 方法")

        print("\n" + "=" * 70)
        print("MCPT測試結果")
        print("=" * 70)
        print(f"原始{self.metric}:     {self.original_score:>10.4f}")
        print(f"排列平均值:           {self.test_result['summary']['perm_mean']:>10.4f}")
        print(f"排列標準差:           {self.test_result['summary']['perm_std']:>10.4f}")
        print(f"排列中位數:           {self.test_result['summary']['perm_median']:>10.4f}")
        print(f"排列範圍:             {self.test_result['summary']['perm_min']:>10.4f} ~ "
              f"{self.test_result['summary']['perm_max']:>10.4f}")
        print("-" * 70)
        print(f"p值:                  {self.p_value:>10.4f}")
        print(f"百分位數:             {self.test_result['percentile']:>10.2f}%")
        print(f"顯著性 (α=0.05):      {'是' if self.test_result['is_significant'] else '否':>10}")
        print("=" * 70)

        if self.test_result['is_significant']:
            print("\n結論: 策略顯著優於隨機（p < 0.05），可能具有真實的預測能力。")
        else:
            print("\n結論: 策略未顯著優於隨機（p >= 0.05），可能存在過度擬合。")
        print("=" * 70)

    def save_results(
        self,
        data_source: str = "unknown",
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        保存MCPT測試結果

        自動創建目錄結構：results/{data_source}_{date}/mcpt/
        保存以下文件：
        - mcpt_summary.json: 測試摘要
        - permutation_scores.csv: 所有排列的績效分數

        Args:
            data_source: 數據源名稱
            output_dir: 輸出目錄（可選）

        Returns:
            Path: 保存結果的目錄路徑
        """
        if self.test_result is None:
            raise RuntimeError("請先執行 run() 方法")

        # 確定輸出目錄
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "results"

        # 創建帶日期的子目錄
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = output_dir / f"{data_source}_{date_str}" / "mcpt"
        result_dir.mkdir(parents=True, exist_ok=True)

        # 保存測試摘要
        # 轉換 numpy 類型為 Python 原生類型（修復 JSON 序列化錯誤）
        summary_data = {
            'test_info': {
                'data_source': data_source,
                'test_date': date_str,
                'mode': self.mode,
                'metric': self.metric,
                'n_permutations': int(self.n_permutations),
                'start_index': int(self.start_index),
                'transaction_cost': float(self.transaction_cost),
                'slippage': float(self.slippage)
            },
            'strategy_params': self.strategy_params,
            'results': {
                'original_score': float(self.original_score),
                'p_value': float(self.p_value),
                'is_significant': bool(self.test_result['is_significant']),  # 修復: numpy.bool_ -> bool
                'percentile': float(self.test_result['percentile'])
            },
            'summary': self.test_result['summary']
        }

        with open(result_dir / 'mcpt_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        # 保存排列分數
        perm_df = pd.DataFrame({
            'permutation': range(self.n_permutations),
            'score': self.permutation_scores
        })
        perm_df.to_csv(result_dir / 'permutation_scores.csv', index=False)

        print(f"\nMCPT結果已保存到: {result_dir}")
        print(f"- mcpt_summary.json: 測試摘要")
        print(f"- permutation_scores.csv: 排列分數")

        return result_dir


if __name__ == "__main__":
    # 測試MCPT
    print("蒙地卡羅排列測試器測試\n")

    # 創建測試數據
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=500, freq='1H')

    returns = np.random.normal(0.0001, 0.02, 500)
    prices = 30000 * (1 + returns).cumprod()

    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 500)),
        'high': prices * (1 + np.abs(np.random.normal(0.002, 0.001, 500))),
        'low': prices * (1 - np.abs(np.random.normal(0.002, 0.001, 500))),
        'close': prices,
        'volume': np.random.uniform(100, 1000, 500)
    }, index=dates)

    # 定義測試策略
    def ma_strategy(data: pd.DataFrame, period: int = 20):
        """簡單移動平均策略"""
        ma = data['close'].rolling(period).mean()
        signals = pd.Series(0, index=data.index)
        signals[data['close'] > ma] = 1
        signals[data['close'] < ma] = -1
        return signals

    # 測試1: Normal MCPT
    print("=" * 70)
    print("測試1: Normal MCPT（固定參數）")
    print("=" * 70)

    tester_normal = MonteCarloTester(
        strategy_func=ma_strategy,
        data=data,
        strategy_params={'period': 20},
        start_index=50,
        n_permutations=100,  # 少量排列用於測試
        mode='normal',
        metric='sharpe_ratio',
        n_jobs=2
    )

    results_normal = tester_normal.run()

    # 保存結果
    save_path = tester_normal.save_results(data_source="test_mcpt_normal")

    # 測試2: Walk-Forward MCPT
    print("\n\n" + "=" * 70)
    print("測試2: Walk-Forward MCPT（每次重新優化）")
    print("=" * 70)

    tester_wf = MonteCarloTester(
        strategy_func=ma_strategy,
        data=data,
        strategy_params={'period': [10, 20, 30]},  # 參數網格
        start_index=50,
        n_permutations=50,  # 更少的排列（因為每次要優化）
        mode='walkforward',
        metric='sharpe_ratio',
        n_jobs=2
    )

    results_wf = tester_wf.run()

    # 保存結果
    save_path_wf = tester_wf.save_results(data_source="test_mcpt_walkforward")

    print("\n所有測試完成!")
