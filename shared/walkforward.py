"""
Walk-Forward Analysis
Walk-Forward分析器 - 模擬實盤的滾動優化和樣本外測試

理論基礎：
Walk-Forward分析通過滾動窗口的方式，在訓練期優化參數，
然後在測試期（樣本外）評估績效，模擬實盤交易的參數更新過程。

工作流程：
1. 將數據分為多個重疊或不重疊的窗口
2. 在每個訓練窗口優化參數
3. 在相應的測試窗口評估樣本外績效
4. 滾動窗口，重複步驟2-3
5. 彙總所有測試期的績效

優點：
- 避免前視偏差（Look-Ahead Bias）
- 評估參數穩定性
- 模擬實盤定期重新優化的情況
- 更真實的績效估計

注意：
- 訓練期過短：參數不穩定
- 訓練期過長：參數過時
- 建議訓練期 = 2-5倍測試期
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Callable, List, Optional
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from .optimizer import ParameterOptimizer, convert_numpy_types
from .backtest import quick_backtest


# Progress bar configuration
TQDM_WALKFORWARD_CONFIG = {
    'smoothing': 0.1,      # 平滑ETA計算
    'mininterval': 1,      # 1秒更新適合耗時的窗口處理
    'unit': 'window'       # 使用ASCII避免編碼問題
}


class WalkForwardAnalyzer:
    """
    Walk-Forward分析器

    使用滾動窗口進行參數優化和樣本外測試

    Attributes:
        strategy_func: 策略函數
        data: OHLCV數據
        param_grid: 參數網格
        train_window: 訓練窗口大小（K棒數量）
        test_window: 測試窗口大小（K棒數量）
        step: 滾動步長（K棒數量）
        objective: 優化目標
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
        transaction_cost: float = 0.0006,
        slippage: float = 0.0001,
        n_jobs: int = -1,
        periods_per_year: int = 252
    ):
        """
        初始化Walk-Forward分析器

        Args:
            strategy_func: 策略函數，格式為 func(data, **params) -> pd.Series
            data: OHLCV數據
            param_grid: 參數網格字典
            train_window: 訓練窗口大小（K棒數量）
            test_window: 測試窗口大小（K棒數量）
            step: 滾動步長（默認等於test_window，不重疊）
            objective: 優化目標
            transaction_cost: 交易成本
            slippage: 滑價
            n_jobs: 優化時的並行進程數
            periods_per_year: 每年的時間週期數 (默認252)

        Example:
            >>> analyzer = WalkForwardAnalyzer(
            ...     strategy_func=my_strategy,
            ...     data=df,
            ...     param_grid={'period': [10, 20, 30, 40, 50]},
            ...     train_window=1000,  # 訓練期1000根K線
            ...     test_window=250,    # 測試期250根K線
            ...     step=250,           # 每次滾動250根K線
            ...     objective='sharpe_ratio'
            ... )
            >>> results = analyzer.run()
        """
        self.strategy_func = strategy_func
        self.data = data.copy()
        self.param_grid = param_grid
        self.train_window = train_window
        self.test_window = test_window
        self.step = step if step is not None else test_window
        self.objective = objective
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.n_jobs = n_jobs
        self.periods_per_year = periods_per_year

        # 驗證窗口大小
        if train_window + test_window > len(data):
            raise ValueError(
                f"訓練窗口({train_window}) + 測試窗口({test_window}) "
                f"大於數據長度({len(data)})"
            )

        # 結果存儲
        self.windows = []
        self.results_df = None
        self.combined_metrics = None
        self.combined_equity_curve = None  # 合併的OOS權益曲線
        self.buy_hold_equity_curve = None  # 長期持有權益曲線

    def _generate_windows(self) -> List[Dict]:
        """
        生成所有訓練/測試窗口

        Returns:
            List[Dict]: 窗口信息列表
        """
        windows = []
        current_pos = 0

        while current_pos + self.train_window + self.test_window <= len(self.data):
            train_start = current_pos
            train_end = current_pos + self.train_window
            test_start = train_end
            test_end = test_start + self.test_window

            window = {
                'window_id': len(windows),
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_start_date': self.data.index[train_start],
                'train_end_date': self.data.index[train_end - 1],
                'test_start_date': self.data.index[test_start],
                'test_end_date': self.data.index[test_end - 1]
            }

            windows.append(window)
            current_pos += self.step

        return windows

    def _process_window(self, window: Dict) -> Dict:
        """
        處理單個窗口：訓練期優化 + 測試期評估

        Args:
            window: 窗口信息字典

        Returns:
            Dict: 窗口結果字典
        """
        # 提取訓練和測試數據
        train_data = self.data.iloc[window['train_start']:window['train_end']]
        test_data = self.data.iloc[window['test_start']:window['test_end']]

        # 步驟1: 在訓練期優化參數
        # 注意：在Walk-Forward過程中禁用優化器的進度條，避免嵌套顯示
        optimizer = ParameterOptimizer(
            strategy_func=self.strategy_func,
            data=train_data,
            param_grid=self.param_grid,
            objective=self.objective,
            transaction_cost=self.transaction_cost,
            slippage=self.slippage,
            n_jobs=self.n_jobs,
            periods_per_year=self.periods_per_year
        )

        optimizer.optimize(verbose=False)  # 避免嵌套進度條
        best_params = optimizer.get_best_params()

        # 計算訓練期績效
        train_signals = self.strategy_func(train_data, **best_params)
        _, train_metrics = quick_backtest(
            data=train_data,
            signals=train_signals,
            transaction_cost=self.transaction_cost,
            slippage=self.slippage,
            periods_per_year=self.periods_per_year
        )

        # 步驟2: 在測試期（樣本外）評估
        test_signals = self.strategy_func(test_data, **best_params)
        test_returns, test_metrics = quick_backtest(
            data=test_data,
            signals=test_signals,
            transaction_cost=self.transaction_cost,
            slippage=self.slippage,
            periods_per_year=self.periods_per_year
        )

        # 將收益率轉換為權益曲線
        initial_capital = 10000.0
        test_equity_curve = initial_capital * (1 + test_returns).cumprod()

        # 構建結果
        result = {
            'window_id': window['window_id'],
            'train_start': str(window['train_start_date']),
            'train_end': str(window['train_end_date']),
            'test_start': str(window['test_start_date']),
            'test_end': str(window['test_end_date']),
            'optimal_params': best_params,
            # 訓練期指標（加is_前綴）
            'is_sharpe_ratio': train_metrics['sharpe_ratio'],
            'is_total_return': train_metrics['total_return'],
            'is_max_drawdown': train_metrics['max_drawdown'],
            'is_profit_factor': train_metrics['profit_factor'],
            'is_win_rate': train_metrics['win_rate'],
            'is_total_trades': train_metrics['total_trades'],
            # 測試期指標（加oos_前綴）
            'oos_sharpe_ratio': test_metrics['sharpe_ratio'],
            'oos_total_return': test_metrics['total_return'],
            'oos_max_drawdown': test_metrics['max_drawdown'],
            'oos_profit_factor': test_metrics['profit_factor'],
            'oos_win_rate': test_metrics['win_rate'],
            'oos_total_trades': test_metrics['total_trades'],
            # 保存權益曲線用於後續合併
            'test_equity_curve': test_equity_curve,
            'test_data': test_data
        }

        return result

    def run(self, verbose: bool = True) -> pd.DataFrame:
        """
        執行Walk-Forward分析

        Returns:
            pd.DataFrame: 完整的Walk-Forward結果DataFrame

        Example:
            >>> results = analyzer.run()
            >>> print(results)
            >>> print(f"平均樣本外夏普比率: {results['oos_sharpe_ratio'].mean():.3f}")
        """
        # 生成窗口
        self.windows = self._generate_windows()
        n_windows = len(self.windows)

        if verbose:
            print("\n" + "=" * 70)
            print("Walk-Forward分析")
            print("=" * 70)
            print(f"數據總長度:       {len(self.data)} K棒")
            print(f"訓練窗口:         {self.train_window} K棒")
            print(f"測試窗口:         {self.test_window} K棒")
            print(f"滾動步長:         {self.step} K棒")
            print(f"總窗口數:         {n_windows}")
            print(f"優化目標:         {self.objective}")
            print(f"參數空間:         {self.param_grid}")
            print("=" * 70)

        if n_windows == 0:
            raise ValueError("無法生成任何窗口，請檢查窗口大小設置")

        # 處理每個窗口
        results = []

        if verbose:
            iterator = tqdm(
                self.windows,
                desc="處理窗口",
                **TQDM_WALKFORWARD_CONFIG
            )
        else:
            iterator = self.windows

        for window in iterator:
            result = self._process_window(window)
            results.append(result)

        # 轉換為DataFrame
        self.results_df = pd.DataFrame(results)

        # 合併所有OOS窗口的權益曲線
        self._combine_equity_curves(results)

        # 計算綜合績效指標
        self._calculate_combined_metrics()

        if verbose:
            self.print_summary()

        return self.results_df

    def _combine_equity_curves(self, results: List[Dict]):
        """
        合併所有OOS窗口的權益曲線，並計算長期持有基準

        實現邏輯：
        1. 將每個窗口的權益曲線標準化到當前資金水平
        2. 按時間順序連接所有窗口的權益點（保存所有點，不只是最後一點）
        3. 同步計算長期持有權益曲線作為基準

        Args:
            results: 所有窗口的結果列表，每個包含：
                - test_equity_curve: 該窗口的權益曲線
                - test_data: 該窗口的價格數據
        """
        import warnings

        if len(results) == 0:
            return

        # 初始化合併的權益曲線
        combined_equity = []
        combined_index = []
        buy_hold_equity = []

        # 初始資金
        current_capital = 10000.0
        initial_price = None

        for i, result in enumerate(results):
            equity_curve = result['test_equity_curve']
            test_data = result['test_data']

            # 驗證權益曲線有效性
            if len(equity_curve) == 0:
                continue

            # 檢查初始權益是否有效
            initial_equity = equity_curve.iloc[0]
            if initial_equity == 0 or np.isnan(initial_equity) or np.isinf(initial_equity):
                continue

            # 計算該窗口的收益率（用於連接下一窗口）
            final_equity = equity_curve.iloc[-1]
            if np.isnan(final_equity) or np.isinf(final_equity):
                continue

            window_return = final_equity / initial_equity - 1

            # 檢查收益率是否有效
            if np.isnan(window_return) or np.isinf(window_return):
                window_return = 0.0

            # 標準化該窗口的權益曲線（從當前資金開始）
            # 保存所有點，而不只是最後一點
            normalized_equity = current_capital * (equity_curve / initial_equity)

            # 添加所有權益點
            combined_equity.extend(normalized_equity.values)
            combined_index.extend(normalized_equity.index)

            # 計算長期持有
            if initial_price is None:
                if 'close' not in test_data.columns or len(test_data) == 0:
                    continue

                initial_close = test_data['close'].iloc[0]
                if initial_close == 0 or np.isnan(initial_close) or np.isinf(initial_close):
                    continue

                initial_price = initial_close

            # 為該窗口計算長期持有權益（所有點）
            current_prices = test_data['close']
            buy_hold_returns = current_prices / initial_price
            buy_hold_equity.extend((10000.0 * buy_hold_returns).values)

            # 更新下一窗口的初始資金
            current_capital = current_capital * (1 + window_return)

        # 創建合併的權益曲線Series
        if len(combined_equity) > 0:
            self.combined_equity_curve = pd.Series(combined_equity, index=combined_index)
            self.buy_hold_equity_curve = pd.Series(buy_hold_equity, index=combined_index)
        else:
            import warnings
            warnings.warn("All windows have invalid equity curves - combined_equity_curve will be None")
            self.combined_equity_curve = None
            self.buy_hold_equity_curve = None

    def _calculate_combined_metrics(self):
        """計算所有測試期的綜合績效"""
        # 合併所有測試期的收益率
        # 注意：這裡簡化處理，實際應該重構equity curve
        oos_metrics = {
            'avg_oos_sharpe': self.results_df['oos_sharpe_ratio'].mean(),
            'std_oos_sharpe': self.results_df['oos_sharpe_ratio'].std(),
            'avg_oos_return': self.results_df['oos_total_return'].mean(),
            'avg_oos_max_dd': self.results_df['oos_max_drawdown'].mean(),
            'avg_oos_profit_factor': self.results_df['oos_profit_factor'].mean(),
            'avg_oos_win_rate': self.results_df['oos_win_rate'].mean(),
            'total_oos_trades': self.results_df['oos_total_trades'].sum(),
            'positive_windows': (self.results_df['oos_total_return'] > 0).sum(),
            'negative_windows': (self.results_df['oos_total_return'] <= 0).sum(),
            'consistency': (self.results_df['oos_total_return'] > 0).mean(),
        }

        # IS vs OOS比較
        oos_metrics['is_oos_sharpe_ratio'] = (
            self.results_df['oos_sharpe_ratio'].mean() /
            self.results_df['is_sharpe_ratio'].mean()
            if self.results_df['is_sharpe_ratio'].mean() != 0 else 0
        )

        self.combined_metrics = oos_metrics

    def print_summary(self):
        """打印Walk-Forward分析摘要"""
        if self.combined_metrics is None:
            raise RuntimeError("請先執行 run() 方法")

        print("\n" + "=" * 70)
        print("Walk-Forward分析摘要")
        print("=" * 70)
        print(f"總窗口數:             {len(self.results_df)}")
        print(f"正收益窗口:           {self.combined_metrics['positive_windows']}")
        print(f"負收益窗口:           {self.combined_metrics['negative_windows']}")
        print(f"一致性:               {self.combined_metrics['consistency']*100:.2f}%")
        print("-" * 70)
        print("樣本外(OOS)平均績效:")
        print(f"  平均夏普比率:       {self.combined_metrics['avg_oos_sharpe']:.3f}")
        print(f"  夏普比率標準差:     {self.combined_metrics['std_oos_sharpe']:.3f}")
        print(f"  平均總收益率:       {self.combined_metrics['avg_oos_return']*100:.2f}%")
        print(f"  平均最大回撤:       {self.combined_metrics['avg_oos_max_dd']*100:.2f}%")
        print(f"  平均獲利因子:       {self.combined_metrics['avg_oos_profit_factor']:.3f}")
        print(f"  平均勝率:           {self.combined_metrics['avg_oos_win_rate']*100:.2f}%")
        print(f"  總交易次數:         {self.combined_metrics['total_oos_trades']:.0f}")
        print("-" * 70)
        print("樣本內 vs 樣本外:")
        print(f"  OOS/IS夏普比率:     {self.combined_metrics['is_oos_sharpe_ratio']:.3f}")
        print("  (接近1表示沒有過度擬合)")
        print("=" * 70)

    def save_results(
        self,
        data_source: str = "unknown",
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        保存Walk-Forward分析結果

        自動創建目錄結構：results/{data_source}_{date}/walkforward/
        保存以下文件：
        - walkforward_summary.json: 分析摘要
        - walkforward_results.csv: 完整結果表
        - window_params.json: 每個窗口的最佳參數

        Args:
            data_source: 數據源名稱
            output_dir: 輸出目錄（可選）

        Returns:
            Path: 保存結果的目錄路徑
        """
        if self.results_df is None:
            raise RuntimeError("請先執行 run() 方法")

        # 確定輸出目錄
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "results"

        # 創建帶日期的子目錄
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = output_dir / f"{data_source}_{date_str}" / "walkforward"
        result_dir.mkdir(parents=True, exist_ok=True)

        # 保存摘要
        summary_data = {
            'analysis_info': {
                'data_source': data_source,
                'analysis_date': date_str,
                'train_window': self.train_window,
                'test_window': self.test_window,
                'step': self.step,
                'n_windows': len(self.windows),
                'objective': self.objective,
                'param_grid': self.param_grid,
                'transaction_cost': self.transaction_cost,
                'slippage': self.slippage
            },
            'combined_metrics': self.combined_metrics,
            'data_info': {
                'start_date': str(self.data.index[0]),
                'end_date': str(self.data.index[-1]),
                'total_bars': len(self.data)
            }
        }

        # 轉換 NumPy 類型以進行 JSON 序列化
        summary_data = convert_numpy_types(summary_data)

        with open(result_dir / 'walkforward_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        # 保存完整結果表
        # 將optimal_params從字典轉為字符串（CSV兼容）
        results_to_save = self.results_df.copy()
        results_to_save['optimal_params'] = results_to_save['optimal_params'].apply(str)
        results_to_save.to_csv(result_dir / 'walkforward_results.csv', index=False)

        # 單獨保存參數（JSON格式，方便讀取）
        window_params = {}
        for _, row in self.results_df.iterrows():
            window_params[f"window_{row['window_id']}"] = row['optimal_params']

        with open(result_dir / 'window_params.json', 'w', encoding='utf-8') as f:
            json.dump(window_params, f, indent=2, ensure_ascii=False)

        print(f"\nWalk-Forward結果已保存到: {result_dir}")
        print(f"- walkforward_summary.json: 分析摘要")
        print(f"- walkforward_results.csv: 完整結果表")
        print(f"- window_params.json: 窗口參數")

        return result_dir

    def plot_walk_forward_results(
        self,
        figsize: tuple = (15, 10),
        save_path: Optional[Path] = None,
        show: bool = True
    ):
        """
        繪製Walk-Forward分析結果

        包含4個子圖 (2x2):
        1. 樣本外獲利因子趨勢
        2. 樣本外夏普比率趨勢
        3. 樣本外總收益率趨勢
        4. 樣本外最大回撤趨勢

        Args:
            figsize: 圖表尺寸 (寬, 高)，默認 (15, 10)
            save_path: 保存路徑，如果提供則保存圖表
            show: 是否顯示圖表，默認 True

        Returns:
            matplotlib.figure.Figure: 圖表對象

        Raises:
            ValueError: 如果未執行 run() 方法或結果數據無效

        Example:
            >>> analyzer.run()
            >>> fig = analyzer.plot_walk_forward_results()
            >>> # 或保存圖表
            >>> analyzer.plot_walk_forward_results(save_path=Path('results/plot.png'))
        """
        # 驗證數據存在
        if self.results_df is None or len(self.results_df) == 0:
            raise ValueError("必須先執行 run() 方法")

        # 驗證必要列存在
        required_cols = [
            'test_start',
            'oos_profit_factor',
            'oos_sharpe_ratio',
            'oos_total_return',
            'oos_max_drawdown'
        ]
        missing_cols = [col for col in required_cols if col not in self.results_df.columns]
        if missing_cols:
            raise ValueError(f"結果數據缺少必要列: {missing_cols}")

        # 檢查是否有有效數據
        for col in ['oos_profit_factor', 'oos_sharpe_ratio',
                    'oos_total_return', 'oos_max_drawdown']:
            if self.results_df[col].isna().all():
                raise ValueError(f"列 '{col}' 全為 NaN 值，無法繪圖")

        # 轉換日期
        try:
            test_start_dates = pd.to_datetime(self.results_df['test_start'])
        except Exception as e:
            raise ValueError(f"無法將 'test_start' 轉換為日期格式: {e}")

        # 創建圖表
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 子圖1: 獲利因子
        axes[0, 0].plot(test_start_dates, self.results_df['oos_profit_factor'], 'o-')
        axes[0, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Out-of-Sample Profit Factor')
        axes[0, 0].set_ylabel('Profit Factor')
        axes[0, 0].grid(True, alpha=0.3)

        # 子圖2: 夏普比率
        axes[0, 1].plot(test_start_dates, self.results_df['oos_sharpe_ratio'], 'o-', color='green')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Out-of-Sample Sharpe Ratio')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].grid(True, alpha=0.3)

        # 子圖3: 總收益率
        axes[1, 0].plot(test_start_dates, self.results_df['oos_total_return'] * 100, 'o-', color='purple')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('Out-of-Sample Total Return (%)')
        axes[1, 0].set_ylabel('Return (%)')
        axes[1, 0].grid(True, alpha=0.3)

        # 子圖4: 最大回撤
        axes[1, 1].plot(test_start_dates, self.results_df['oos_max_drawdown'] * 100, 'o-', color='orange')
        axes[1, 1].set_title('Out-of-Sample Max Drawdown (%)')
        axes[1, 1].set_ylabel('Drawdown (%)')
        axes[1, 1].grid(True, alpha=0.3)

        # 調整佈局
        plt.tight_layout()

        # 保存圖表
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖表已保存至: {save_path}")

        # 顯示圖表
        if show:
            plt.show()

        return fig


if __name__ == "__main__":
    # 測試Walk-Forward分析器
    print("Walk-Forward分析器測試\n")

    # 創建測試數據
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=2000, freq='1H')

    returns = np.random.normal(0.0001, 0.02, 2000)
    prices = 30000 * (1 + returns).cumprod()

    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 2000)),
        'high': prices * (1 + np.abs(np.random.normal(0.002, 0.001, 2000))),
        'low': prices * (1 - np.abs(np.random.normal(0.002, 0.001, 2000))),
        'close': prices,
        'volume': np.random.uniform(100, 1000, 2000)
    }, index=dates)

    # 定義策略
    def ma_strategy(data: pd.DataFrame, period: int = 20):
        """移動平均策略"""
        ma = data['close'].rolling(period).mean()
        signals = pd.Series(0, index=data.index)
        signals[data['close'] > ma] = 1
        signals[data['close'] < ma] = -1
        return signals

    # 測試Walk-Forward分析
    print("=" * 70)
    print("執行Walk-Forward分析")
    print("=" * 70)

    analyzer = WalkForwardAnalyzer(
        strategy_func=ma_strategy,
        data=data,
        param_grid={
            'period': [10, 20, 30, 40, 50]
        },
        train_window=500,   # 訓練期500根K線
        test_window=125,    # 測試期125根K線
        step=125,           # 每次滾動125根K線（不重疊）
        objective='sharpe_ratio',
        n_jobs=2
    )

    results = analyzer.run()

    # 查看結果
    print("\n前3個窗口的結果:")
    print(results.head(3))

    print("\n樣本外夏普比率分佈:")
    print(results['oos_sharpe_ratio'].describe())

    # 保存結果
    save_path = analyzer.save_results(data_source="test_walkforward")

    print("\n所有測試完成!")
