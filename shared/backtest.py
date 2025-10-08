"""
Backtest Engine
回測引擎 - 包含完整的交易成本模擬

功能：
1. 計算策略收益（含交易成本）
2. 生成權益曲線
3. 計算13個績效指標
4. 自動保存結果到results/{data_source}_{date}/

交易成本模型：
- transaction_cost: 固定交易成本比例（例如 0.0006 = 0.06%）
- slippage: 滑價比例（例如 0.0001 = 0.01%）
- 總成本 = (transaction_cost + slippage) * |倉位變化量|
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import json

from .metrics import calculate_all_metrics


class BacktestEngine:
    """
    回測引擎類

    用於模擬策略交易並計算績效指標，包含完整的交易成本處理

    Attributes:
        data: OHLCV數據DataFrame
        signals: 交易信號Series（1=多頭, -1=空頭, 0=空倉）
        transaction_cost: 固定交易成本比例
        slippage: 滑價比例
        initial_capital: 初始資金
        periods_per_year: 每年的時間週期數
    """

    def __init__(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        transaction_cost: float = 0.0006,
        slippage: float = 0.0001,
        initial_capital: float = 100000,
        periods_per_year: int = 8760
    ):
        """
        初始化回測引擎

        Args:
            data: OHLCV數據，必須包含'close'列
            signals: 交易信號序列（1=多頭, -1=空頭, 0=空倉）
            transaction_cost: 固定交易成本比例（默認0.06%）
            slippage: 滑價比例（默認0.01%）
            initial_capital: 初始資金（默認100,000）
            periods_per_year: 每年的時間週期數（默認8760，1小時K線）

        Example:
            >>> engine = BacktestEngine(
            ...     data=df,
            ...     signals=signals,
            ...     transaction_cost=0.0006,
            ...     slippage=0.0001
            ... )
            >>> results = engine.run()
        """
        self.data = data.copy()
        self.signals = signals.copy()
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.initial_capital = initial_capital
        self.periods_per_year = periods_per_year

        # 確保索引對齊
        self.signals = self.signals.reindex(self.data.index, fill_value=0)

        # 結果存儲
        self.equity_curve = None
        self.returns = None
        self.metrics = None

    def run(self) -> Dict:
        """
        執行回測

        Returns:
            Dict: 包含以下鍵的結果字典
                - equity_curve: 權益曲線Series
                - returns: 收益率Series（已扣除交易成本）
                - metrics: 13個績效指標字典
                - summary: 回測摘要信息
        """
        # 計算價格收益率
        price_returns = self.data['close'].pct_change()

        # 計算倉位變化（用於計算交易成本）
        position_changes = self.signals.diff().fillna(self.signals)

        # 計算交易成本
        # 成本 = (交易成本 + 滑價) * |倉位變化量|
        total_cost_rate = self.transaction_cost + self.slippage
        trade_costs = total_cost_rate * position_changes.abs()

        # 計算策略收益（信號 * 價格變動 - 交易成本）
        strategy_returns = self.signals.shift(1) * price_returns - trade_costs
        strategy_returns = strategy_returns.fillna(0)

        # 計算權益曲線
        self.equity_curve = self.initial_capital * (1 + strategy_returns).cumprod()
        self.returns = strategy_returns

        # 計算績效指標
        self.metrics = calculate_all_metrics(
            returns=strategy_returns,
            signals=self.signals,
            initial_capital=self.initial_capital,
            periods_per_year=self.periods_per_year
        )

        # 準備摘要信息
        summary = self._prepare_summary()

        return {
            'equity_curve': self.equity_curve,
            'returns': self.returns,
            'metrics': self.metrics,
            'summary': summary
        }

    def _prepare_summary(self) -> Dict:
        """準備回測摘要信息"""
        return {
            'data_points': len(self.data),
            'start_date': str(self.data.index[0]),
            'end_date': str(self.data.index[-1]),
            'initial_capital': self.initial_capital,
            'final_capital': self.equity_curve.iloc[-1] if self.equity_curve is not None else 0,
            'transaction_cost': self.transaction_cost,
            'slippage': self.slippage,
            'total_cost_rate': self.transaction_cost + self.slippage,
            'periods_per_year': self.periods_per_year
        }

    def save_results(
        self,
        data_source: str = "unknown",
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        保存回測結果到文件

        自動創建目錄結構：results/{data_source}_{date}/
        保存以下文件：
        - backtest_summary.json: 回測摘要和績效指標
        - equity_curve.csv: 權益曲線數據
        - returns.csv: 收益率數據

        Args:
            data_source: 數據源名稱（例如 "bybit_btc_1h"）
            output_dir: 輸出目錄（可選，默認為項目根目錄的results文件夾）

        Returns:
            Path: 保存結果的目錄路徑

        Example:
            >>> engine.run()
            >>> save_path = engine.save_results(data_source="bybit_btc_1h")
            >>> print(f"結果已保存到: {save_path}")
        """
        if self.equity_curve is None:
            raise RuntimeError("請先執行 run() 方法進行回測")

        # 確定輸出目錄
        if output_dir is None:
            # 默認使用項目根目錄的results文件夾
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "results"

        # 創建帶日期的子目錄
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = output_dir / f"{data_source}_{date_str}"
        result_dir.mkdir(parents=True, exist_ok=True)

        # 保存回測摘要和績效指標
        summary_data = {
            'summary': self._prepare_summary(),
            'metrics': self.metrics
        }

        with open(result_dir / 'backtest_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        # 保存權益曲線
        equity_df = pd.DataFrame({
            'timestamp': self.equity_curve.index,
            'equity': self.equity_curve.values
        })
        equity_df.to_csv(result_dir / 'equity_curve.csv', index=False)

        # 保存收益率
        returns_df = pd.DataFrame({
            'timestamp': self.returns.index,
            'returns': self.returns.values
        })
        returns_df.to_csv(result_dir / 'returns.csv', index=False)

        # 保存信號和價格數據
        combined_df = pd.DataFrame({
            'timestamp': self.data.index,
            'close': self.data['close'].values,
            'signal': self.signals.values,
            'returns': self.returns.values,
            'equity': self.equity_curve.values
        })
        combined_df.to_csv(result_dir / 'backtest_data.csv', index=False)

        print(f"\n回測結果已保存到: {result_dir}")
        print(f"- backtest_summary.json: 摘要和績效指標")
        print(f"- equity_curve.csv: 權益曲線")
        print(f"- returns.csv: 收益率數據")
        print(f"- backtest_data.csv: 完整回測數據")

        return result_dir

    def print_summary(self):
        """打印回測摘要和績效指標"""
        if self.metrics is None:
            raise RuntimeError("請先執行 run() 方法進行回測")

        print("\n" + "=" * 70)
        print("回測摘要")
        print("=" * 70)

        summary = self._prepare_summary()
        print(f"數據點數:      {summary['data_points']:>10}")
        print(f"開始日期:      {summary['start_date']:>25}")
        print(f"結束日期:      {summary['end_date']:>25}")
        print(f"初始資金:      ${summary['initial_capital']:>10,.2f}")
        print(f"最終資金:      ${summary['final_capital']:>10,.2f}")
        print(f"交易成本:      {summary['transaction_cost']*100:>10.4f}%")
        print(f"滑價:          {summary['slippage']*100:>10.4f}%")
        print(f"總成本率:      {summary['total_cost_rate']*100:>10.4f}%")

        print("\n" + "=" * 70)
        print("績效指標")
        print("=" * 70)
        print(f"總收益率:      {self.metrics['total_return']*100:>10.2f}%")
        print(f"年化收益率:    {self.metrics['annualized_return']*100:>10.2f}%")
        print(f"年化波動率:    {self.metrics['volatility']*100:>10.2f}%")
        print(f"夏普比率:      {self.metrics['sharpe_ratio']:>10.3f}")
        print(f"最大回撤:      {self.metrics['max_drawdown']*100:>10.2f}%")
        print(f"卡瑪比率:      {self.metrics['calmar_ratio']:>10.3f}")
        print("-" * 70)
        print(f"勝率:          {self.metrics['win_rate']*100:>10.2f}%")
        print(f"平均獲利:      {self.metrics['avg_win']*100:>10.4f}%")
        print(f"平均虧損:      {self.metrics['avg_loss']*100:>10.4f}%")
        print(f"獲利因子:      {self.metrics['profit_factor']:>10.3f}")
        print("-" * 70)
        print(f"總交易次數:    {self.metrics['total_trades']:>10}")
        print(f"獲利交易:      {self.metrics['winning_trades']:>10}")
        print(f"虧損交易:      {self.metrics['losing_trades']:>10}")
        print("=" * 70)


def quick_backtest(
    data: pd.DataFrame,
    signals: pd.Series,
    transaction_cost: float = 0.0006,
    slippage: float = 0.0001,
    periods_per_year: int = 252
) -> Tuple[pd.Series, Dict]:
    """
    快速回測函數（用於優化和MCPT等需要大量回測的場景）

    Args:
        data: OHLCV數據
        signals: 交易信號
        transaction_cost: 交易成本
        slippage: 滑價
        periods_per_year: 每年的時間週期數 (默認252，日K線)

    Returns:
        Tuple[pd.Series, Dict]: (收益率序列, 績效指標字典)
    """
    engine = BacktestEngine(
        data=data,
        signals=signals,
        transaction_cost=transaction_cost,
        slippage=slippage,
        periods_per_year=periods_per_year
    )
    results = engine.run()
    return results['returns'], results['metrics']


if __name__ == "__main__":
    # 測試回測引擎
    print("回測引擎測試\n")

    # 創建模擬數據
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='1H')

    # 模擬價格（隨機遊走）
    returns = np.random.normal(0.0001, 0.02, 1000)
    prices = 30000 * (1 + returns).cumprod()

    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 1000)),
        'high': prices * (1 + np.abs(np.random.normal(0.002, 0.001, 1000))),
        'low': prices * (1 - np.abs(np.random.normal(0.002, 0.001, 1000))),
        'close': prices,
        'volume': np.random.uniform(100, 1000, 1000)
    }, index=dates)

    # 創建簡單的移動平均策略信號
    ma_short = data['close'].rolling(20).mean()
    ma_long = data['close'].rolling(50).mean()
    signals = pd.Series(0, index=data.index)
    signals[ma_short > ma_long] = 1
    signals[ma_short < ma_long] = -1

    # 執行回測
    print("執行回測...")
    engine = BacktestEngine(
        data=data,
        signals=signals,
        transaction_cost=0.0006,
        slippage=0.0001,
        initial_capital=100000
    )

    results = engine.run()

    # 打印結果
    engine.print_summary()

    # 保存結果
    save_path = engine.save_results(data_source="test_strategy")

    # 測試快速回測函數
    print("\n\n快速回測函數測試...")
    returns, metrics = quick_backtest(data, signals)
    print(f"夏普比率: {metrics['sharpe_ratio']:.3f}")
    print(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")
    print(f"總收益率: {metrics['total_return']*100:.2f}%")
