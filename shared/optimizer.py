"""
Parameter Optimizer
參數優化器 - Medium風格的並行參數優化

功能：
1. 網格搜索（Grid Search）參數優化
2. 並行計算支持（ProcessPoolExecutor）
3. 可自定義優化目標（夏普比率、獲利因子等）
4. 自動保存完整結果到results/{data_source}_{date}/
5. 支持交易成本配置

使用場景：
- In-Sample參數優化
- Walk-Forward分析的每個訓練窗口
- 策略參數敏感性分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Callable, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import json
from tqdm import tqdm

from .backtest import quick_backtest
from .metrics import calculate_sharpe_ratio, calculate_profit_factor, calculate_calmar_ratio


# Progress bar configuration
TQDM_OPTIMIZER_CONFIG = {
    'smoothing': 0.1,      # 平滑ETA計算
    'mininterval': 0.5,    # 0.5秒更新適合快速組合評估
    'unit': 'combo'        # 使用ASCII避免編碼問題
}


def convert_numpy_types(obj):
    """
    遞歸轉換 NumPy 類型為 Python 原生類型，以支持 JSON 序列化

    Args:
        obj: 需要轉換的對象（可以是 NumPy 標量、數組、字典、列表或元組）

    Returns:
        轉換後的對象，所有 NumPy 類型替換為 Python 原生類型

    Examples:
        >>> convert_numpy_types(np.int64(42))
        42
        >>> convert_numpy_types({'a': np.float64(1.5), 'b': [np.int32(2)]})
        {'a': 1.5, 'b': [2]}

    Notes:
        - np.integer -> int
        - np.floating -> float
        - np.bool_ -> bool
        - np.ndarray -> list
        - 遞歸處理 dict, list, tuple 中的嵌套對象
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # 處理 NaN 和 Inf 值
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
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


# ==================== 模塊級別的目標函數（支持多進程序列化） ====================

def _objective_sharpe_ratio(metrics: Dict) -> float:
    """優化目標：夏普比率"""
    return metrics['sharpe_ratio']


def _objective_profit_factor(metrics: Dict) -> float:
    """優化目標：獲利因子"""
    return metrics['profit_factor']


def _objective_calmar_ratio(metrics: Dict) -> float:
    """優化目標：卡瑪比率"""
    return metrics['calmar_ratio']


def _objective_total_return(metrics: Dict) -> float:
    """優化目標：總收益率"""
    return metrics['total_return']


def _objective_max_drawdown(metrics: Dict) -> float:
    """優化目標：最大回撤（負值，因為要最小化回撤）"""
    return -metrics['max_drawdown']

# ==================== 多進程工作函數 (頂層函數) ====================

def _worker_evaluate_params(strategy_func, data, params, transaction_cost, slippage, periods_per_year, objective_func):
    """
    評估單個參數組合的工作函數 (為多進程設計)

    Args:
        strategy_func: 策略函數
        data: OHLCV數據
        params: 參數字典
        transaction_cost: 交易成本
        slippage: 滑價
        periods_per_year: 年化週期
        objective_func: 目標函數

    Returns:
        Dict: 包含參數和績效指標的結果字典
    """
    try:
        # 運行策略獲取信號
        signals = strategy_func(data, **params)

        # 執行快速回測
        _, metrics = quick_backtest(
            data=data,
            signals=signals,
            transaction_cost=transaction_cost,
            slippage=slippage,
            periods_per_year=periods_per_year
        )

        # 計算目標值
        objective_value = objective_func(metrics)

        # 構建結果字典
        result = {
            **params,
            'objective_value': objective_value,
            **metrics
        }
        return result

    except Exception as e:
        # 如果參數組合導致錯誤，返回帶有錯誤信息的結果
        result = {
            **params,
            'objective_value': -np.inf,
            'error': str(e)
        }
        return result


class ParameterOptimizer:
    """
    參數優化器類

    使用網格搜索和並行計算找到最佳參數組合

    Attributes:
        strategy_func: 策略函數，接受(data, **params)返回signals
        data: OHLCV數據
        param_grid: 參數網格字典
        objective: 優化目標函數名稱或自定義函數
        transaction_cost: 交易成本
        slippage: 滑價
        n_jobs: 並行進程數
    """

    def __init__(
        self,
        strategy_func: Callable,
        data: pd.DataFrame,
        param_grid: Dict[str, List],
        objective: str = 'sharpe_ratio',
        transaction_cost: float = 0.0006,
        slippage: float = 0.0001,
        n_jobs: int = -1,
        periods_per_year: int = 252
    ):
        self.strategy_func = strategy_func
        self.data = data.copy()
        self.param_grid = param_grid
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.n_jobs = n_jobs
        self.periods_per_year = periods_per_year

        # 設置優化目標函數
        if isinstance(objective, str):
            self.objective_name = objective
            self.objective_func = self._get_objective_function(objective)
        else:
            self.objective_name = 'custom'
            self.objective_func = objective

        # 生成所有參數組合
        self.param_combinations = self._generate_param_combinations()
        self.results_df = None
        self.best_params = None
        self.best_score = None

    def _get_objective_function(self, objective: str) -> Callable:
        """根據名稱獲取優化目標函數"""
        objective_map = {
            'sharpe_ratio': _objective_sharpe_ratio,
            'profit_factor': _objective_profit_factor,
            'calmar_ratio': _objective_calmar_ratio,
            'total_return': _objective_total_return,
            'max_drawdown': _objective_max_drawdown,
        }

        if objective not in objective_map:
            raise ValueError(
                f"未知的優化目標: {objective}. "
                f"可選: {list(objective_map.keys())}"
            )

        return objective_map[objective]

    def _generate_param_combinations(self) -> List[Dict]:
        """生成所有參數組合"""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        combinations = []
        for value_combo in product(*values):
            param_dict = dict(zip(keys, value_combo))
            combinations.append(param_dict)

        return combinations

    def optimize(self, verbose: bool = True) -> pd.DataFrame:
        """
        執行參數優化

        Args:
            verbose: 是否顯示進度條

        Returns:
            pd.DataFrame: 完整的優化結果DataFrame，按objective_value降序排序
        """
        total_combinations = len(self.param_combinations)
        
        if verbose:
            core_count = self.n_jobs if self.n_jobs > 0 else 'all available'
            print(f"Optimizing {total_combinations} combinations using {core_count} cores...")

        results = []

        # 根據 n_jobs 決定執行方式
        if self.n_jobs == 1:
            # 單進程（用於調試）
            iterator = tqdm(
                self.param_combinations,
                total=total_combinations,
                desc="優化進度",
                **TQDM_OPTIMIZER_CONFIG
            ) if verbose else self.param_combinations

            for params in iterator:
                result = _worker_evaluate_params(
                    self.strategy_func, self.data, params, self.transaction_cost,
                    self.slippage, self.periods_per_year, self.objective_func
                )
                results.append(result)
        else:
            # 多進程並行
            max_workers = None if self.n_jobs == -1 else self.n_jobs

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任務
                futures = {
                    executor.submit(
                        _worker_evaluate_params,
                        self.strategy_func, self.data, params, self.transaction_cost,
                        self.slippage, self.periods_per_year, self.objective_func
                    ): params for params in self.param_combinations
                }

                # 收集結果
                iterator = tqdm(
                    as_completed(futures),
                    total=total_combinations,
                    desc="優化進度",
                    **TQDM_OPTIMIZER_CONFIG
                ) if verbose else as_completed(futures)

                for future in iterator:
                    result = future.result()
                    results.append(result)

        # 轉換為DataFrame並排序
        self.results_df = pd.DataFrame(results)
        
        if self.results_df.empty:
            print("警告: 優化未產生任何有效結果。 সন")
            self.best_params = {}
            self.best_score = None
            return self.results_df

        self.results_df = self.results_df.sort_values(
            'objective_value',
            ascending=False
        ).reset_index(drop=True)

        # 提取最佳參數
        best_row = self.results_df.iloc[0]
        self.best_params = {
            key: convert_numpy_types(best_row[key])
            for key in self.param_grid.keys()
        }
        self.best_score = convert_numpy_types(best_row['objective_value'])

        return self.results_df

    def save_results(
        self,
        data_source: str = "unknown",
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        保存優化結果
        """
        if self.results_df is None:
            raise RuntimeError("請先執行 optimize() 方法")

        # 確定輸出目錄
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "results"

        # 創建帶日期的子目錄
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = output_dir / f"{data_source}_{date_str}" / "optimization"
        result_dir.mkdir(parents=True, exist_ok=True)

        # 保存優化摘要
        summary = {
            'data_source': data_source,
            'optimization_date': date_str,
            'total_combinations': len(self.param_combinations),
            'param_grid': self.param_grid,
            'objective': self.objective_name,
            'transaction_cost': self.transaction_cost,
            'slippage': self.slippage,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'data_info': {
                'start_date': str(self.data.index[0]),
                'end_date': str(self.data.index[-1]),
                'data_points': len(self.data)
            }
        }

        # 轉換 NumPy 類型為 Python 原生類型
        summary = convert_numpy_types(summary)

        with open(result_dir / 'optimization_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 保存完整結果表
        self.results_df.to_csv(
            result_dir / 'optimization_results.csv',
            index=False
        )

        # 保存最佳參數（單獨文件，方便讀取）
        best_params_converted = convert_numpy_types(self.best_params)
        with open(result_dir / 'best_params.json', 'w', encoding='utf-8') as f:
            json.dump(best_params_converted, f, indent=2, ensure_ascii=False)

        print(f"\n優化結果已保存到: {result_dir}")
        print(f"- optimization_summary.json: 優化摘要")
        print(f"- optimization_results.csv: 完整結果表")
        print(f"- best_params.json: 最佳參數")

        return result_dir

    def get_best_params(self) -> Dict:
        """獲取最佳參數"""
        if self.best_params is None:
            # 如果沒有最佳參數（可能因為沒有有效結果），返回空字典
            return {}
        return self.best_params

    def get_param_sensitivity(self, param_name: str) -> pd.DataFrame:
        """
        分析單個參數的敏感性
        """
        if self.results_df is None or self.results_df.empty:
            raise RuntimeError("請先執行 optimize() 且必須有有效結果")

        if param_name not in self.param_grid:
            raise ValueError(f"參數 '{param_name}' 不在參數網格中")

        # 按參數分組並計算平均目標值
        sensitivity = self.results_df.groupby(param_name)['objective_value'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()

        return sensitivity.sort_values(param_name)


# ==================== 輔助函數 ====================

def quick_optimize(
    strategy_func: Callable,
    data: pd.DataFrame,
    param_grid: Dict[str, List],
    objective: str = 'sharpe_ratio',
    n_jobs: int = -1,
    periods_per_year: int = 252
) -> Dict:
    """
    快速優化函數（簡化接口）
    """
    optimizer = ParameterOptimizer(
        strategy_func=strategy_func,
        data=data,
        param_grid=param_grid,
        objective=objective,
        n_jobs=n_jobs,
        periods_per_year=periods_per_year
    )
    optimizer.optimize(verbose=False)
    return optimizer.get_best_params()


if __name__ == "__main__":
    # 測試參數優化器
    print("參數優化器測試\n")

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

    # 定義簡單的移動平均策略
    def ma_cross_strategy(data: pd.DataFrame, fast_period: int = 10, slow_period: int = 20):
        """移動平均交叉策略"""
        ma_fast = data['close'].rolling(fast_period).mean()
        ma_slow = data['close'].rolling(slow_period).mean()

        signals = pd.Series(0, index=data.index)
        signals[ma_fast > ma_slow] = 1
        signals[ma_fast < ma_slow] = -1

        return signals

    # 測試1: 基本優化
    print("=" * 70)
    print("測試1: 基本參數優化")
    print("=" * 70)

    optimizer = ParameterOptimizer(
        strategy_func=ma_cross_strategy,
        data=data,
        param_grid={
            'fast_period': [5, 10, 15, 20],
            'slow_period': [20, 30, 40, 50]
        },
        objective='sharpe_ratio',
        n_jobs=2
    )

    results = optimizer.optimize()

    # 測試2: 參數敏感性分析
    print("\n" + "=" * 70)
    print("測試2: 參數敏感性分析")
    print("=" * 70)

    fast_sensitivity = optimizer.get_param_sensitivity('fast_period')
    print("\nfast_period敏感性:")
    print(fast_sensitivity)

    slow_sensitivity = optimizer.get_param_sensitivity('slow_period')
    print("\nslow_period敏感性:")
    print(slow_sensitivity)

    # 測試3: 保存結果
    print("\n" + "=" * 70)
    print("測試3: 保存結果")
    print("=" * 70)

    save_path = optimizer.save_results(data_source="test_optimization")

    # 測試4: 快速優化函數
    print("\n" + "=" * 70)
    print("測試4: 快速優化函數")
    print("=" * 70)

    best_params = quick_optimize(
        strategy_func=ma_cross_strategy,
        data=data,
        param_grid={
            'fast_period': [5, 10, 15],
            'slow_period': [20, 30, 40]
        },
        objective='profit_factor',
        n_jobs=2
    )
    print(f"快速優化結果: {best_params}")

    print("\n所有測試完成!")