# -*- coding: utf-8 -*-
"""
Walk-Forward Analysis 模塊

包含兩個版本的 Walk-Forward 分析器：
1. WalkForwardAnalyzer - 基礎版本（保留向後兼容）
2. WalkForward - 增強版本（基於 cvilliq 實現，推薦使用）
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Callable, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random
import logging
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 設置 Python 輸出編碼
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    import io
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from .optimizer import ParameterOptimizer, convert_numpy_types
from .backtest import quick_backtest

# 失敗結果的默認值
FAILED_SHARPE = -999


class WalkForwardAnalyzer:
    """
    Walk-Forward分析器（基礎版本，保留向後兼容）
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
        optimization_method: str = 'grid',
        n_iter: int = 100,
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
            method=self.optimization_method,
            n_iter=self.n_iter,
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

        print(f"\nWalk-Forward results saved to: {result_dir}")
        return result_dir


# ==================== 增強版 WalkForward 類（推薦使用）====================

def calculate_num_workers(n_jobs: int = None) -> int:
    """計算並行工作進程數"""
    cpu_count = os.cpu_count() or 1
    if n_jobs is None or n_jobs == -1:
        return cpu_count
    elif n_jobs < 0:
        return max(1, cpu_count + n_jobs)
    else:
        return max(1, n_jobs)


def generate_all_combinations(param_grid: dict) -> List[dict]:
    """生成所有可能的參數組合"""
    import itertools
    param_names = list(param_grid.keys())
    param_values = [list(v) for v in param_grid.values()]
    all_combos = list(itertools.product(*param_values))
    return [dict(zip(param_names, combo)) for combo in all_combos]


def random_sample_combinations(param_grid: dict, n_samples: int) -> List[dict]:
    """隨機採樣n個參數組合"""
    combinations = []
    for _ in range(n_samples):
        combo = {name: random.choice(list(values)) for name, values in param_grid.items()}
        combinations.append(combo)
    return combinations


class WalkForward:
    """
    增強版 Walk-Forward 分析器（推薦使用）

    整合 cvilliq 的所有優秀邏輯，提供統一的走向測試框架。

    主要特性：
    - 多階段參數篩選 + 穩健性分析
    - 預熱期處理，避免指標邊界效應
    - 失敗窗口備份參數 + 權益曲線填補
    - 基於連續權益曲線的正確整體指標計算
    - 自動數據頻率檢測，自適應窗口配置
    """

    def __init__(
        self,
        strategy,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        train_days: Optional[int] = None,
        test_days: Optional[int] = None,
        step_days: Optional[int] = None,
        train_window_points: Optional[int] = None,
        test_window_points: Optional[int] = None,
        step_window_points: Optional[int] = None,
        param_grid: Dict[str, List] = None,
        backtest_config: Dict = None,
        sensitivity_config: Dict = None,
        n_trials: int = 10000,
        n_jobs: int = -1,
    ):
        """
        初始化 WalkForward 分析器

        Args:
            strategy: 策略類實例
            data: 完整數據 DataFrame
            start_date: 分析開始日期 (YYYY-MM-DD)
            end_date: 分析結束日期 (YYYY-MM-DD)
            train_days: 訓練窗口天數（針對日級數據）
            test_days: 測試窗口天數
            step_days: 滾動步長天數
            train_window_points: 訓練窗口數據點數（針對非日級數據）
            test_window_points: 測試窗口數據點數
            step_window_points: 滾動步長數據點數
            param_grid: 參數網格
            backtest_config: 回測配置（初始資金、手續費、滑點等）
            sensitivity_config: 敏感性分析配置（各種閾值）
            n_trials: 隨機搜索次數
            n_jobs: 並行工作數
        """
        self.strategy = strategy
        self.data = data.copy()
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.param_grid = param_grid or {}
        self.backtest_config = backtest_config or {}
        self.sensitivity_config = sensitivity_config or {}
        self.n_trials = n_trials
        self.n_jobs = n_jobs

        # 回測配置默認值
        self.initial_capital = self.backtest_config.get('initial_capital', 100000)
        self.transaction_cost = self.backtest_config.get('commission_bps', 5) / 10000
        self.slippage = self.backtest_config.get('slippage_bps', 2) / 10000
        self.stop_loss_fee = self.backtest_config.get('stop_loss_fee', 0.00055)

        # 敏感性配置默認值
        self.sharpe_threshold = self.sensitivity_config.get('sharpe_threshold', 1.25)
        self.calmar_threshold = self.sensitivity_config.get('calmar_threshold', 2.5)
        self.annual_return_threshold = self.sensitivity_config.get('annual_return_threshold', 0.5)
        self.max_drawdown_threshold = self.sensitivity_config.get('max_drawdown_threshold', 0.3)

        # 自動檢測數據頻率並設置窗口
        self._detect_frequency_and_setup_windows(
            train_days, test_days, step_days,
            train_window_points, test_window_points, step_window_points
        )

        # 結果存儲
        self.windows = []
        self.all_oos_metrics = []
        self.all_params_evolution = []
        self.all_equity_curves = {}
        self.combined_equity = None

    def _detect_frequency_and_setup_windows(
        self,
        train_days, test_days, step_days,
        train_window_points, test_window_points, step_window_points
    ):
        """自動檢測數據頻率並設置窗口"""
        if len(self.data) < 2:
            raise ValueError("數據不足以檢測時間頻率")

        time_diff = (self.data.index[1] - self.data.index[0]).total_seconds() / 3600

        if time_diff >= 24:
            self.window_unit = "天"
            if train_days is None:
                raise ValueError("日級數據需要指定 train_days, test_days, step_days")
            self.window_delta = timedelta(days=train_days)
            self.test_delta = timedelta(days=test_days)
            self.step_delta = timedelta(days=step_days)
        else:
            self.window_unit = "小時"
            if train_window_points is None:
                raise ValueError("非日級數據需要指定 train_window_points, test_window_points, step_window_points")
            self.window_delta = timedelta(hours=train_window_points)
            self.test_delta = timedelta(hours=test_window_points)
            self.step_delta = timedelta(hours=step_window_points)

        self.periods_per_year = 365 if self.window_unit == "天" else 365 * 24

    def _generate_windows(self) -> List[Dict]:
        """生成 Walk-Forward 窗口"""
        windows = []
        current_train_start = self.start_date
        full_end_dt = self.end_date

        window_id = 1
        while current_train_start < full_end_dt:
            train_start = current_train_start
            train_end = train_start + self.window_delta
            test_start = train_end
            test_end = test_start + self.test_delta

            if test_end > full_end_dt:
                test_end = full_end_dt

            if test_end <= test_start:
                break

            windows.append({
                "window_id": window_id,
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d")
            })

            current_train_start += self.step_delta
            window_id += 1

        return windows

    def _evaluate_single_params(self, params: dict, train_data: pd.DataFrame) -> dict:
        """評估單個參數組合"""
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["PYTHONWARNINGS"] = "ignore"

        try:
            from .backtest import BacktestEngine

            # 提取 leverage 和 capital_allocation，計算最終槓桿
            leverage = params.get('leverage', 1)
            capital_allocation = params.get('capital_allocation', 1.0)
            final_leverage = leverage * capital_allocation

            signals = self.strategy.generate_signals(train_data, **params)

            engine = BacktestEngine(
                data=train_data,
                signals=signals,
                transaction_cost=self.transaction_cost,
                slippage=self.slippage,
                initial_capital=self.initial_capital,
                periods_per_year=self.periods_per_year,
                leverage=final_leverage,
                stop_loss_fee=self.stop_loss_fee
            )

            results = engine.run()

            metrics = results.get('metrics', {})
            sharpe_ratio = metrics.get('sharpe_ratio', FAILED_SHARPE)
            calmar_ratio = metrics.get('calmar_ratio', FAILED_SHARPE)
            annual_return = metrics.get('annualized_return', FAILED_SHARPE)
            max_drawdown = metrics.get('max_drawdown', FAILED_SHARPE)

            return {
                **params,
                "sharpe_ratio": float(sharpe_ratio),
                "calmar_ratio": float(calmar_ratio),
                "annual_return": float(annual_return),
                "max_drawdown": float(max_drawdown)
            }

        except Exception as e:
            error_info = {
                **params,
                "sharpe_ratio": FAILED_SHARPE,
                "calmar_ratio": FAILED_SHARPE,
                "annual_return": FAILED_SHARPE,
                "max_drawdown": FAILED_SHARPE
            }
            if str(e):
                error_info["error_type"] = type(e).__name__
                error_info["error_msg"] = str(e)[:100]
            return error_info

    def _optimize_window(self, window: Dict) -> Tuple[dict, pd.DataFrame, dict]:
        """優化單個訓練窗口"""
        from .parameter_selection import filter_top_params, filter_robust_params

        wid = window["window_id"]
        train_start = window["train_start"]
        train_end = window["train_end"]

        logger.info(f"\n【窗口 {wid}】{train_start} → {train_end}")

        train_start_dt = pd.to_datetime(train_start)
        train_end_dt = pd.to_datetime(train_end)
        train_data = self.data[
            (self.data.index >= train_start_dt) &
            (self.data.index <= train_end_dt)
        ]

        if train_data.empty:
            logger.error(f"訓練數據為空: {train_start} - {train_end}")
            return {}, pd.DataFrame(), {}

        logger.info("→ 生成所有可能的參數組合...")
        all_combinations = generate_all_combinations(self.param_grid)
        total_combinations = len(all_combinations)
        logger.info(f"參數空間大小: {total_combinations:,} 個組合")

        param_combinations = all_combinations
        if "long_entry_quantile" in self.param_grid and "short_entry_quantile" in self.param_grid:
            original_count = len(param_combinations)
            param_combinations = [
                p for p in param_combinations
                if p.get('long_entry_quantile', 1.0) >= p.get('short_entry_quantile', 0.0)
            ]
            filtered_count = original_count - len(param_combinations)
            if filtered_count > 0:
                logger.info(f"過濾掉 {filtered_count:,} 個無效組合")

        if not param_combinations:
            logger.error("沒有有效的參數組合可供評估")
            return {}, pd.DataFrame(), {}

        logger.info(f"有效組合數: {len(param_combinations):,}")

        if len(param_combinations) > self.n_trials:
            logger.info(f"→ 有效組合數大於 N_TRIALS ({self.n_trials})，將從中隨機抽樣")
            param_combinations = random.sample(param_combinations, self.n_trials)
        else:
            logger.info("→ 使用所有有效組合進行評估")

        logger.info(f"最終評估組合數: {len(param_combinations):,}\n")

        num_workers = calculate_num_workers(self.n_jobs)
        worker = partial(self._evaluate_single_params, train_data=train_data)

        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker, params) for params in param_combinations]
            for future in tqdm(futures, desc=f"窗口{wid}優化", unit="組合", ncols=100,
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
                try:
                    result = future.result(timeout=300)
                    results.append(result)
                except Exception:
                    pass

        results_df = pd.DataFrame(results)

        if results_df.empty or "sharpe_ratio" not in results_df.columns:
            logger.error("優化失敗: 結果為空")
            return {}, pd.DataFrame(), {}

        valid_results = results_df[results_df['sharpe_ratio'] > FAILED_SHARPE].copy()
        if valid_results.empty:
            logger.error("優化失敗: 所有結果都失敗")
            return {}, pd.DataFrame(), {}

        results_df = valid_results

        top_params_df = filter_top_params(
            results_df=results_df,
            sharpe_threshold=self.sharpe_threshold,
            calmar_threshold=self.calmar_threshold,
            annual_return_threshold=self.annual_return_threshold,
            max_drawdown_threshold=-self.max_drawdown_threshold
        )
        logger.info(f"  ├─ 同時符合所有條件: {len(top_params_df)} 個組合")

        best_params = {}
        sensitivity_report = {}
        param_names = list(self.param_grid.keys())

        if not top_params_df.empty:
            logger.info(f"  ├─ [情況A] 找到 {len(top_params_df)} 個符合條件的參數，進行穩健性分析...")
            try:
                robust_params_df, best_robust_params = filter_robust_params(
                    candidates_df=top_params_df,
                    all_results_df=results_df,
                    param_grid=self.param_grid,
                    metric="sharpe_ratio",
                    sharpe_threshold=self.sharpe_threshold,
                    top_n=10
                )

                if not robust_params_df.empty and best_robust_params:
                    best_robust_row = robust_params_df.iloc[0]
                    best_params = {pname: best_robust_row[pname] for pname in param_names if pname in best_robust_row}

                    match_mask = pd.Series(True, index=top_params_df.index)
                    for pname, pval in best_params.items():
                        if pname in top_params_df.columns:
                            match_mask &= (top_params_df[pname] == pval)
                    matched_rows = top_params_df[match_mask]

                    if not matched_rows.empty:
                        matched_row = matched_rows.iloc[0]
                        logger.info(f"  ├─ 從 {len(top_params_df)} 個候選中選出排名前 {len(robust_params_df)} 的穩健參數")
                        logger.info(f"  ├─ [選中] 排名第 1 的參數:")
                        logger.info(f"  |  - Sharpe={matched_row['sharpe_ratio']:.4f}, Calmar={matched_row.get('calmar_ratio', 0):.4f}")
                        logger.info(f"  |  - 穩健性分數={best_robust_row.get('robustness_score', 0):.4f}")

                        sensitivity_report = {
                            "selection_method": "robust_from_filtered",
                            "robustness_score": float(best_robust_row.get("robustness_score", 0)),
                            "best_sharpe": float(matched_row["sharpe_ratio"]),
                            "n_filtered_candidates": len(top_params_df),
                            "n_robust_candidates": len(robust_params_df),
                            "fallback_used": False
                        }
                    else:
                        best_params = {}

                if not best_params:
                    best_result = top_params_df.iloc[0]
                    best_params = {pname: best_result[pname] for pname in param_names}
                    logger.info(f"  ├─ 穩健性分析無有效結果，回退到該窗口最高夏普比率的參數")
                    logger.info(f"  |  - Sharpe={best_result['sharpe_ratio']:.4f}")

                    sensitivity_report = {
                        "selection_method": "fallback_from_filtered",
                        "best_sharpe": float(best_result["sharpe_ratio"]),
                        "n_filtered_candidates": len(top_params_df),
                        "fallback_used": True,
                        "fallback_reason": "robustness_analysis_failed"
                    }

            except Exception as e:
                best_result = top_params_df.iloc[0]
                best_params = {pname: best_result[pname] for pname in param_names}
                logger.info(f"  ├─ 穩健性分析異常，回退到該窗口最高夏普比率的參數")
                logger.info(f"  |  - Sharpe={best_result['sharpe_ratio']:.4f}")

                sensitivity_report = {
                    "selection_method": "fallback_from_filtered",
                    "best_sharpe": float(best_result["sharpe_ratio"]),
                    "n_filtered_candidates": len(top_params_df),
                    "fallback_used": True,
                    "fallback_reason": f"exception: {str(e)[:50]}"
                }

        else:
            best_idx = results_df["sharpe_ratio"].idxmax()
            best_result = results_df.loc[best_idx]
            best_params = {pname: best_result[pname] for pname in param_names}
            logger.info(f"  ├─ [情況B] 無符合條件參數，回退到全局最高夏普比率")
            logger.info(f"  |  - Sharpe={best_result['sharpe_ratio']:.4f}")

            sensitivity_report = {
                "selection_method": "global_best_sharpe",
                "best_sharpe": float(best_result["sharpe_ratio"]),
                "n_total_candidates": len(results_df),
                "fallback_used": False
            }

        return best_params, results_df, sensitivity_report

    def _run_oos_test(self, window: Dict, best_params: dict) -> Tuple[pd.Series, dict]:
        """運行樣本外測試"""
        from .backtest import BacktestEngine

        wid = window["window_id"]
        test_start = window["test_start"]
        test_end = window["test_end"]

        try:
            required_lookback = self.strategy.get_required_lookback(**best_params) \
                if hasattr(self.strategy, 'get_required_lookback') else 0

            test_start_dt = pd.to_datetime(test_start)
            test_end_dt = pd.to_datetime(test_end)

            try:
                test_start_idx = self.data.index.get_loc(test_start_dt)
            except KeyError:
                test_start_idx = self.data.index.searchsorted(test_start_dt)
                if test_start_idx >= len(self.data):
                    logger.error(f"OOS測試失敗: test_start_dt {test_start_dt} 晚於最後數據點")
                    return pd.Series(), {}

            warmup_start_idx = max(0, test_start_idx - required_lookback)
            warmup_start_dt = self.data.index[warmup_start_idx]

            extended_test_data = self.data.loc[warmup_start_dt:test_end_dt].copy()

            if len(extended_test_data) < required_lookback:
                logger.error(f"OOS測試失敗: 擴展後的數據長度小於所需的回看期")
                return pd.Series(), {}

            signals = self.strategy.generate_signals(extended_test_data, **best_params)

            signals = signals.reindex(extended_test_data.index).loc[test_start_dt:test_end_dt]
            test_data = extended_test_data.loc[test_start_dt:test_end_dt]

            if test_data.empty:
                return pd.Series(), {}

            # 提取 leverage 和 capital_allocation，計算最終槓桿
            leverage = best_params.get('leverage', 1)
            capital_allocation = best_params.get('capital_allocation', 1.0)
            final_leverage = leverage * capital_allocation

            engine = BacktestEngine(
                data=test_data,
                signals=signals,
                transaction_cost=self.transaction_cost,
                slippage=self.slippage,
                initial_capital=self.initial_capital,
                periods_per_year=self.periods_per_year,
                leverage=final_leverage,
                stop_loss_fee=self.stop_loss_fee
            )

            results = engine.run()
            equity_curve = results.get('equity_curve', pd.Series())

            metrics = results.get('metrics', {})
            oos_metrics = {
                "sharpe_ratio": float(metrics.get('sharpe_ratio', 0)),
                "calmar_ratio": float(metrics.get('calmar_ratio', 0)),
                "annual_return": float(metrics.get('annualized_return', 0)),
                "max_drawdown": float(metrics.get('max_drawdown', 0)),
                "data_length": len(test_data)
            }

            return equity_curve, oos_metrics

        except Exception as e:
            logger.error(f"OOS測試失敗: {e}")
            return pd.Series(), {}

    def run(self) -> Dict:
        """運行 Walk-Forward 分析"""
        logger.info("=" * 80)
        logger.info("Walk-Forward 優化與敏感性分析")
        logger.info("=" * 80)
        logger.info("")

        logger.info("【Step 1】生成 Walk-Forward 窗口")
        self.windows = self._generate_windows()
        logger.info(f"✓ 生成 {len(self.windows)} 個 Walk-Forward 窗口\n")

        logger.info("【Step 2】運行 Walk-Forward 分析")
        last_successful_params = None
        last_successful_window_id = None

        for window in self.windows:
            wid = window["window_id"]
            train_start = window["train_start"]
            train_end = window["train_end"]
            test_start = window["test_start"]
            test_end = window["test_end"]

            best_params, train_results_df, sensitivity_report = self._optimize_window(window)

            if not best_params:
                if last_successful_params:
                    window_gap = wid - last_successful_window_id
                    logger.warning(f"窗口 {wid} 優化失敗，使用窗口 {last_successful_window_id} 的參數作為備份")
                    best_params = last_successful_params.copy()
                else:
                    logger.warning(f"窗口 {wid} 優化失敗，沒有備份參數可用，跳過")
                    continue
            else:
                last_successful_params = best_params.copy()
                last_successful_window_id = wid

            oos_equity, oos_metrics = self._run_oos_test(window, best_params)

            self.all_oos_metrics.append({
                "window_id": wid,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                **best_params,
                **sensitivity_report,
                **oos_metrics
            })

            self.all_params_evolution.append({
                "window_id": wid,
                **best_params
            })

            if not oos_equity.empty:
                self.all_equity_curves[f"window_{wid}"] = oos_equity

        logger.info("\n【Step 3】生成報告和圖表")
        logger.info("-" * 80)

        overall_sharpe = 0
        overall_calmar = 0
        overall_annual_return = 0
        overall_max_drawdown = 0

        if self.all_equity_curves:
            try:
                self.combined_equity = self._combine_equity_curves()
                if self.combined_equity is not None and not self.combined_equity.empty:
                    from .backtest import calculate_all_metrics

                    combined_returns = self.combined_equity.pct_change().fillna(0)
                    overall_metrics = calculate_all_metrics(
                        returns=combined_returns,
                        signals=pd.Series(np.ones(len(combined_returns)), index=combined_returns.index),
                        initial_capital=self.combined_equity.iloc[0],
                        periods_per_year=self.periods_per_year
                    )

                    overall_sharpe = overall_metrics.get('sharpe_ratio', 0)
                    overall_calmar = overall_metrics.get('calmar_ratio', 0)
                    overall_annual_return = overall_metrics.get('annualized_return', 0)
                    overall_max_drawdown = overall_metrics.get('max_drawdown', 0)

                    logger.info(f"\n🎯 整體 OOS 性能（所有窗口連接）:")
                    logger.info(f"   夏普比率: {overall_sharpe:.4f}")
                    logger.info(f"   卡瑪比率: {overall_calmar:.4f}")
                    logger.info(f"   年報酬: {overall_annual_return*100:.2f}%")
                    logger.info(f"   最大回撤: {overall_max_drawdown*100:.2f}%")
                    logger.info("")
            except Exception as e:
                logger.warning(f"整體性能計算失敗: {e}")

        return {
            'performance_summary': {
                'sharpe_ratio': overall_sharpe,
                'calmar_ratio': overall_calmar,
                'annual_return': overall_annual_return,
                'max_drawdown': overall_max_drawdown
            },
            'walkforward_results': pd.DataFrame(self.all_oos_metrics) if self.all_oos_metrics else pd.DataFrame(),
            'params_evolution': pd.DataFrame(self.all_params_evolution) if self.all_params_evolution else pd.DataFrame(),
            'combined_equity': self.combined_equity
        }

    def _combine_equity_curves(self) -> Optional[pd.Series]:
        """合併所有窗口的權益曲線"""
        if not self.all_equity_curves:
            return None

        all_equity_list = []
        window_ids = sorted(self.all_equity_curves.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)

        for window_id in window_ids:
            equity = self.all_equity_curves[window_id]

            if equity is None or equity.empty or len(equity) < 2:
                logger.warning(f"窗口 {window_id} 的權益曲線數據不足，跳過")
                continue

            try:
                equity = pd.Series(equity.values, index=equity.index, dtype='float64').dropna()
                if equity.empty or equity.isna().all():
                    logger.warning(f"窗口 {window_id} 的權益曲線全為NaN，跳過")
                    continue
            except Exception as e:
                logger.error(f"窗口 {window_id} 權益曲線驗證失敗: {e}")
                continue

            equity_copy = equity.copy()

            if len(all_equity_list) > 0:
                prev_equity = all_equity_list[-1]
                prev_end_value = prev_equity.iloc[-1]
                curr_start_value = equity_copy.iloc[0]

                if pd.notna(prev_end_value) and pd.notna(curr_start_value) and curr_start_value != 0:
                    adjustment_factor = prev_end_value / curr_start_value
                    equity_copy *= adjustment_factor
                    logger.info(f"  窗口 {window_id} 鏈式銜接 - 調整係數: {adjustment_factor:.4f}")
                else:
                    logger.warning(f"  窗口 {window_id} 數值異常，使用前一窗口終值填充")
                    equity_copy.iloc[:] = prev_end_value

            all_equity_list.append(equity_copy)

        if not all_equity_list:
            logger.warning("沒有有效的權益曲線數據")
            return None

        combined_equity = pd.concat(all_equity_list, ignore_index=False)
        combined_equity = combined_equity[~combined_equity.index.duplicated(keep='first')]
        combined_equity = combined_equity.sort_index()

        try:
            freq = pd.infer_freq(self.data.index)
            if freq:
                full_range = pd.date_range(start=combined_equity.index[0], end=combined_equity.index[-1], freq=freq)
                combined_equity = combined_equity.reindex(full_range)

                n_missing = combined_equity.isna().sum()
                if n_missing > 0:
                    if (n_missing / len(full_range) * 100) > 10:
                        combined_equity = combined_equity.interpolate(method='linear', limit_direction='both')
                        combined_equity = combined_equity.bfill().ffill()
                    else:
                        combined_equity = combined_equity.ffill()
        except Exception as e:
            logger.warning(f"填補空窗期時出錯: {e}")

        logger.info(f"合併權益曲線: 總長度={len(combined_equity)}, 起始={combined_equity.iloc[0]:.2f}, 結束={combined_equity.iloc[-1]:.2f}")

        return combined_equity
