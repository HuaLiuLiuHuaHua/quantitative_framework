"""
Backtest Engine
- Simulates trading with full transaction cost modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import json
import dataclasses

from .metrics import calculate_all_metrics
from .base_strategy import BaseStrategy, Position


class EventDrivenBacktestEngine:
    """
    Event-driven backtesting engine.

    This engine simulates trading bar-by-bar, allowing for complex, state-dependent
    exit logic such as trailing stops, breakeven stops, etc.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        initial_capital: float = 100000,
        transaction_cost: float = 0.0006,
        slippage: float = 0.0001,
        lookback: int = 0,
    ):
        self.data = data.copy()
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.lookback = lookback
        self.periods_per_year = 8760 # Assuming 1H data, adjust if necessary

        # Ensure data has necessary columns from strategy
        self.data = self.strategy.add_indicators(self.data)

    def run(self) -> Dict:
        """Executes the event-driven backtest."""
        capital = self.initial_capital
        position: Optional[Position] = None
        equity_history = []
        trades = []
        trailing_stop_price = 0.0

        # Get strategy parameters from the strategy object
        params = self.strategy.params
        atr_multiplier_sl = params.get('atr_multiplier_sl', 0.6) # Updated default
        atr_multiplier_tp = params.get('atr_multiplier_tp', 2.5) # Updated default
        max_hold_periods = params.get('max_hold_periods', 96) # Updated default
        breakeven_atr_multiplier = params.get('breakeven_atr_multiplier', 0.4) # Updated default
        profit_lock_atr_multiplier = params.get('profit_lock_atr_multiplier', 0.4) # Updated default
        trailing_stop_multiplier = params.get('trailing_stop_multiplier', 0.065) # Updated default
        risk_per_trade_percent = params.get('risk_per_trade_percent', 0.0065) # New param, default from test_backtest.py
        max_position_size = params.get('max_position_size', 0.15) # New param, default from test_backtest.py
        volatility_size_factor = params.get('volatility_size_factor', 0.015) # New param, default from test_backtest.py
        dynamic_position_sizing_method = params.get('dynamic_position_sizing_method', 'hybrid') # New param, default from test_backtest.py
        min_risk_reward = params.get('min_risk_reward', 0.25) # New param, default from test_backtest.py

        for i in range(self.lookback, len(self.data)):
            current_candle = self.data.iloc[i]
            atr_val = self.data['atr'].iloc[i]

            # --- DYNAMIC EXIT LOGIC (Breakeven and Trailing Stop) ---
            if position:
                # 1. Breakeven Stop-Loss Logic
                profit_margin_to_breakeven = atr_val * breakeven_atr_multiplier
                if position.direction == 1 and current_candle['close'] > position.entry_price + profit_margin_to_breakeven:
                    position.stop_loss_price = max(position.stop_loss_price, position.entry_price)
                elif position.direction == -1 and current_candle['close'] < position.entry_price - profit_margin_to_breakeven:
                    position.stop_loss_price = min(position.stop_loss_price, position.entry_price)

                # 2. Profit-Lock Trailing Stop Logic
                profit_lock_trigger = atr_val * profit_lock_atr_multiplier
                if position.direction == 1 and current_candle['close'] > position.entry_price + profit_lock_trigger:
                    new_trailing_stop = current_candle['close'] * (1 - trailing_stop_multiplier)
                    trailing_stop_price = max(trailing_stop_price, new_trailing_stop)
                elif position.direction == -1 and current_candle['close'] < position.entry_price - profit_lock_trigger:
                    new_trailing_stop = current_candle['close'] * (1 + trailing_stop_multiplier)
                    trailing_stop_price = min(trailing_stop_price, new_trailing_stop) if trailing_stop_price != 0 else new_trailing_stop

            # --- EXIT CONDITIONS ---
            if position:
                exit_price = 0
                exit_reason = ""
                
                # Check for SL/TP/Time Exit
                if position.direction == 1: # Long position
                    if current_candle['low'] <= position.stop_loss_price:
                        exit_price, exit_reason = position.stop_loss_price, "Stop-Loss"
                    elif trailing_stop_price > 0 and current_candle['low'] <= trailing_stop_price:
                        exit_price, exit_reason = trailing_stop_price, "Trailing-Stop"
                    elif current_candle['high'] >= position.take_profit_price:
                        exit_price, exit_reason = position.take_profit_price, "Take-Profit"
                elif position.direction == -1: # Short position
                    if current_candle['high'] >= position.stop_loss_price:
                        exit_price, exit_reason = position.stop_loss_price, "Stop-Loss"
                    elif trailing_stop_price > 0 and current_candle['high'] >= trailing_stop_price:
                        exit_price, exit_reason = trailing_stop_price, "Trailing-Stop"
                    elif current_candle['low'] <= position.take_profit_price:
                        exit_price, exit_reason = position.take_profit_price, "Take-Profit"

                # Time-based exit
                periods_in_position = i - self.data.index.get_loc(position.entry_timestamp)
                if not exit_price and periods_in_position > max_hold_periods:
                    exit_price, exit_reason = current_candle['close'], "Time-Based Exit"

                # Custom strategy exit signal
                if not exit_price and self.strategy.check_exit_signal(self.data, i, position):
                    exit_price, exit_reason = current_candle['close'], "Strategy Signal"

                if exit_price:
                    exit_price_with_slippage = exit_price * (1 - self.slippage) if position.direction == 1 else exit_price * (1 + self.slippage)
                    if position.direction == 1:
                        pnl = (exit_price_with_slippage - position.entry_price) * position.quantity
                    else:
                        pnl = (position.entry_price - exit_price_with_slippage) * position.quantity
                    trade_value = exit_price_with_slippage * position.quantity
                    cost = trade_value * self.transaction_cost
                    capital += pnl - cost
                    trades.append({
                        "entry_time": position.entry_timestamp, "exit_time": current_candle.name,
                        "direction": position.direction, "pnl": pnl - cost, "exit_reason": exit_reason
                    })
                    position = None
                    trailing_stop_price = 0.0 # Reset trailing stop

            # --- ENTRY LOGIC ---
            if not position:
                entry_signal = self.strategy.check_entry_signal(self.data, i)
                if entry_signal != 0:
                    entry_price = current_candle['close']
                    sl_distance_in_price = atr_val * atr_multiplier_sl
                    tp_distance_in_price = atr_val * atr_multiplier_tp

                    # Assume trade is valid initially
                    trade_is_valid = True

                    # 1. Min Risk-Reward Filter
                    if sl_distance_in_price > 0 and tp_distance_in_price / sl_distance_in_price < min_risk_reward:
                        trade_is_valid = False

                    if trade_is_valid:
                        # 2. Dynamic Position Sizing
                        quantity = 0.0
                        if dynamic_position_sizing_method == "fixed_ratio":
                            quantity = (capital * max_position_size) / entry_price
                        elif dynamic_position_sizing_method == "risk_based":
                            if sl_distance_in_price > 0:
                                risk_per_trade_amount = capital * risk_per_trade_percent
                                quantity = risk_per_trade_amount / sl_distance_in_price
                        elif dynamic_position_sizing_method == "volatility_based":
                            if atr_val > 0:
                                position_size_factor = (1.0 / atr_val) * (capital * volatility_size_factor)
                                quantity = min(position_size_factor, (capital * max_position_size) / entry_price)
                        elif dynamic_position_sizing_method == "hybrid":
                            if sl_distance_in_price > 0 and atr_val > 0:
                                risk_per_trade_amount = capital * risk_per_trade_percent
                                risk_qty = risk_per_trade_amount / sl_distance_in_price
                                vol_qty = (1.0 / atr_val) * (capital * volatility_size_factor)
                                quantity = min(risk_qty, vol_qty)
                        
                        # Cap quantity by max_position_size
                        max_qty_by_capital = (capital * max_position_size) / entry_price if entry_price > 0 else 0
                        quantity = min(quantity, max_qty_by_capital) if quantity > 0 else 0

                        if quantity <= 0:
                            trade_is_valid = False

                    if trade_is_valid:
                        entry_price_with_slippage = entry_price * (1 + self.slippage) if entry_signal == 1 else entry_price * (1 - self.slippage)
                        trade_value = entry_price_with_slippage * quantity
                        cost = trade_value * self.transaction_cost
                        capital -= cost
                        
                        # Dynamic Take-Profit Logic (existing)
                        rolling_atr_mean = self.data['atr'].iloc[max(0, i-50):i].mean()
                        current_tp_multiplier = atr_multiplier_tp
                        if atr_val > rolling_atr_mean:
                            current_tp_multiplier *= 1.5

                        if entry_signal == 1:
                            sl_price = entry_price_with_slippage - sl_distance_in_price
                            tp_price = entry_price_with_slippage + atr_val * current_tp_multiplier
                        else:
                            sl_price = entry_price_with_slippage + sl_distance_in_price
                            tp_price = entry_price_with_slippage - atr_val * current_tp_multiplier

                        position = Position(
                            entry_price=entry_price_with_slippage, quantity=quantity, direction=entry_signal,
                            entry_timestamp=current_candle.name, stop_loss_price=sl_price, take_profit_price=tp_price
                        )

            # --- UPDATE EQUITY ---
            current_equity = capital
            if position:
                unrealized_pnl = (current_candle['close'] - position.entry_price) * position.quantity if position.direction == 1 else (position.entry_price - current_candle['close']) * position.quantity
                current_equity += unrealized_pnl
            equity_history.append(current_equity)

        # --- FINAL METRICS CALCULATION ---
        equity_curve = pd.Series(equity_history, index=self.data.index[self.lookback:])
        returns = equity_curve.pct_change().fillna(0)
        signals_for_metrics = pd.Series(0, index=self.data.index)
        for trade in trades:
            signals_for_metrics.loc[trade['entry_time']:trade['exit_time']] = trade['direction']
        
        metrics = calculate_all_metrics(
            returns=returns, signals=signals_for_metrics,
            initial_capital=self.initial_capital, periods_per_year=self.periods_per_year
        )
        
        return {
            'equity_curve': equity_curve, 'returns': returns,
            'metrics': metrics, 'trades': trades
        }


class BacktestEngine:
    """
    Vectorized backtesting engine.

    Simulates strategy performance based on a static signal series.
    Includes full transaction cost handling.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        transaction_cost: float = 0.0006,
        slippage: float = 0.0001,
        initial_capital: float = 100000,
        periods_per_year: int = 8760,
        leverage: int = 1,
        stop_loss_fee: float = 0.00055
    ):
        # 驗證 leverage 必須是數字 (Issue #7, modified to allow float)
        if not isinstance(leverage, (int, np.integer, float, np.floating)):
            raise TypeError(f"leverage 必須是數字，得到 {type(leverage).__name__}")
        if leverage < 0:
            raise ValueError(f"leverage 必須 >= 0，得到 {leverage}")

        self.data = data.copy()
        self.signals = signals.copy()
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.initial_capital = initial_capital
        self.periods_per_year = periods_per_year
        self.leverage = float(leverage)  # 允許浮點數槓桿
        self.stop_loss_fee = stop_loss_fee  # 止損手續費 (0.055%)
        self.signals = self.signals.reindex(self.data.index, fill_value=0)
        self.equity_curve = None
        self.returns = None
        self.metrics = None

        # 動態止損位置計算
        self.stop_loss_pct = -1.0 / self.leverage if self.leverage > 0 else -np.inf

    def run(self) -> Dict:
        """
        Executes the vectorized backtest with leverage and stop-loss support.

        杠杆和止损逻辑：
        - 应用杠杆倍数到策略收益
        - 检测止损触发（收益 <= stop_loss_pct）
        - 止损时收取额外手续费 (stop_loss_fee)
        
        关键修复：
        - 信号应该在当期直接应用到价格收益（不延迟）
        - 交易成本仅在信号变化时收取（position_changes != 0）
        - 累积权益使用逐期计算方式，避免复合误差
        """
        price_returns = self.data['close'].pct_change().fillna(0)
        position_changes = self.signals.diff().fillna(self.signals)
        total_cost_rate = self.transaction_cost + self.slippage

        # ==================== 修复1：正确应用信号到价格收益 ====================
        # 信号应该延迟一期(shift(1))再应用，因为信号是根据T日的收盘价计算的，
        # 只能在T+1日才能根据该信号进行交易并产生收益。
        # 修复前的错误: strategy_returns = self.signals * price_returns
        # 原因：未延迟信号会导致使用未来函数，即假设在当天信号产生时就能获得当天的收益，这是不现实的。
        
        # 正确逻辑：
        # 1. 使用前一天的信号 (self.signals.shift(1)) 来决定当天的仓位和收益。
        # 2. 仅在position_changes != 0时收取交易成本。
        strategy_returns = self.signals.shift(1) * price_returns  # 使用前一天的信号乘以当天的收益
        
        # 交易成本：仅在开仓/平仓时收取
        # 这里使用简化模型：交易成本 = 开仓成本 + 平仓成本
        # 在position_change时，成本按绝对值倍数收取
        # 例如：+1→-1 的反向，需要收取 2x transaction_cost (平仓+开仓)
        trade_costs = np.zeros_like(strategy_returns)
        trade_costs[position_changes != 0] = total_cost_rate * np.abs(position_changes[position_changes != 0])
        
        # ==================== 修复2：正确计算周期收益 ====================
        # 周期收益 = 头寸收益 - 交易成本（仅在变化时）
        period_returns = strategy_returns - trade_costs
        period_returns = period_returns.fillna(0)

        # ==================== 应用杠杆 ====================
        # 杠杆直接应用到周期收益
        leveraged_period_returns = period_returns * self.leverage

        # ==================== 修复3：检测止损和累积权益 ====================
        # 逐期计算权益，便于正确检测止损触发
        cumulative_pnl = leveraged_period_returns.cumsum()  # 累积PnL
        
        # 识别止损触发点 (累积收益 <= -100%/leverage)
        # 注意：这里检查的是相对收益率
        stop_loss_level = self.stop_loss_pct  # -1.0
        
        # ==================== 修复4：应用止损手续费 ====================
        # 在触发止损的期间收取额外手续费
        stop_loss_costs = np.zeros_like(leveraged_period_returns)
        
        # 找出第一次触发止损的索引
        stop_loss_indices = np.where(cumulative_pnl <= stop_loss_level)[0]
        if len(stop_loss_indices) > 0:
            first_stop_loss_idx = stop_loss_indices[0]
            stop_loss_costs[first_stop_loss_idx] = self.stop_loss_fee
        
        # ==================== 修复5：最终收益计算 ====================
        # 最终周期收益 = 杠杆收益 - 止损手续费
        final_period_returns = leveraged_period_returns - stop_loss_costs
        final_period_returns = final_period_returns.fillna(0)

        # ==================== 修复6：计算权益曲线 ====================
        # 使用累积乘积计算权益曲线：Equity[t] = InitialCapital * ∏(1 + r[i])
        # 这样可以正确反映复合收益
        cumulative_returns_factor = (1 + final_period_returns).cumprod()
        self.equity_curve = self.initial_capital * cumulative_returns_factor
        
        self.returns = final_period_returns
        self.metrics = calculate_all_metrics(
            returns=final_period_returns,
            signals=self.signals,
            initial_capital=self.initial_capital,
            periods_per_year=self.periods_per_year
        )
        summary = self._prepare_summary()
        return {
            'equity_curve': self.equity_curve,
            'returns': self.returns,
            'metrics': self.metrics,
            'summary': summary
        }

    def _prepare_summary(self) -> Dict:
        """Prepares the backtest summary dictionary."""
        return {
            'data_points': len(self.data),
            'start_date': str(self.data.index[0]),
            'end_date': str(self.data.index[-1]),
            'initial_capital': self.initial_capital,
            'final_capital': self.equity_curve.iloc[-1] if self.equity_curve is not None else 0,
            'transaction_cost': self.transaction_cost,
            'slippage': self.slippage,
            'total_cost_rate': self.transaction_cost + self.slippage,
            'leverage': self.leverage,
            'stop_loss_pct': self.stop_loss_pct,
            'stop_loss_fee': self.stop_loss_fee,
            'periods_per_year': self.periods_per_year
        }

    def save_results(
        self,
        data_source: str = "unknown",
        output_dir: Optional[Path] = None
    ) -> Path:
        """Saves backtest results to files."""
        if self.equity_curve is None:
            raise RuntimeError("Please run() the backtest first.")

        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "results"

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = output_dir / f"{data_source}_{date_str}"
        result_dir.mkdir(parents=True, exist_ok=True)

        summary_data = {
            'summary': self._prepare_summary(),
            'metrics': self.metrics
        }

        with open(result_dir / 'backtest_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        equity_df = pd.DataFrame({
            'timestamp': self.equity_curve.index,
            'equity': self.equity_curve.values
        })
        equity_df.to_csv(result_dir / 'equity_curve.csv', index=False)

        returns_df = pd.DataFrame({
            'timestamp': self.returns.index,
            'returns': self.returns.values
        })
        returns_df.to_csv(result_dir / 'returns.csv', index=False)

        combined_df = pd.DataFrame({
            'timestamp': self.data.index,
            'close': self.data['close'].values,
            'signal': self.signals.values,
            'returns': self.returns.values,
            'equity': self.equity_curve.values
        })
        combined_df.to_csv(result_dir / 'backtest_data.csv', index=False)

        print(f"\nBacktest results saved to: {result_dir}")
        return result_dir

    def print_summary(self):
        """Prints the backtest summary and performance metrics."""
        if self.metrics is None:
            raise RuntimeError("Please run() the backtest first.")

        print("\n" + "=" * 70)
        print("Backtest Summary")
        print("=" * 70)

        summary = self._prepare_summary()
        print(f"Data Points:      {summary['data_points']:>10}")
        print(f"Start Date:      {summary['start_date']:>25}")
        print(f"End Date:      {summary['end_date']:>25}")
        print(f"Initial Capital:      ${summary['initial_capital']:>10,.2f}")
        print(f"Final Capital:      ${summary['final_capital']:>10,.2f}")
        print(f"Transaction Cost:      {summary['transaction_cost']*100:>10.4f}%")
        print(f"Slippage:          {summary['slippage']*100:>10.4f}%")
        print(f"Total Cost Rate:      {summary['total_cost_rate']*100:>10.4f}%")

        print("\n" + "=" * 70)
        print("Performance Metrics")
        print("=" * 70)
        print(f"Total Return:      {self.metrics['total_return']*100:>10.2f}%")
        print(f"Annualized Return:    {self.metrics['annualized_return']*100:>10.2f}%")
        print(f"Annualized Volatility:    {self.metrics['volatility']*100:>10.2f}%")
        print(f"Sharpe Ratio:      {self.metrics['sharpe_ratio']:>10.3f}")
        print(f"Max Drawdown:      {self.metrics['max_drawdown']*100:>10.2f}%")
        print(f"Calmar Ratio:      {self.metrics['calmar_ratio']:>10.3f}")
        print("-" * 70)
        print(f"Win Rate:          {self.metrics['win_rate']*100:>10.2f}%")
        print(f"Avg Win:      {self.metrics['avg_win']*100:>10.4f}%")
        print(f"Avg Loss:      {self.metrics['avg_loss']*100:>10.4f}%")
        print(f"Profit Factor:      {self.metrics['profit_factor']:>10.3f}")
        print("-" * 70)
        print(f"Total Trades:    {self.metrics['total_trades']:>10}")
        print(f"Winning Trades:      {self.metrics['winning_trades']:>10}")
        print(f"Losing Trades:      {self.metrics['losing_trades']:>10}")
        print("=" * 70)


def quick_backtest(
    data: pd.DataFrame,
    signals: pd.Series,
    transaction_cost: float = 0.0006,
    slippage: float = 0.0001,
    periods_per_year: int = 252,
    leverage: int = 1,
    stop_loss_fee: float = 0.00055
) -> Tuple[pd.Series, Dict]:
    """
    Quick backtest function for optimization scenarios.

    Args:
        data: OHLCV DataFrame
        signals: Trading signals Series
        transaction_cost: Transaction cost rate (default 0.06%)
        slippage: Slippage rate (default 0.01%)
        periods_per_year: Annual periods (252 for daily, 8760 for hourly)
        leverage: Leverage multiplier (default 1)
        stop_loss_fee: Stop-loss fee rate (default 0.055%)

    Returns:
        (returns Series, metrics Dict)
    """
    engine = BacktestEngine(
        data=data,
        signals=signals,
        transaction_cost=transaction_cost,
        slippage=slippage,
        periods_per_year=periods_per_year,
        leverage=leverage,
        stop_loss_fee=stop_loss_fee
    )
    results = engine.run()
    return results['returns'], results['metrics']


if __name__ == "__main__":
    # Test for the backtest engine
    print("Testing Backtest Engine\n")

    # Create mock data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='1H')

    # Simulate price (random walk)
    returns = np.random.normal(0.0001, 0.02, 1000)
    prices = 30000 * (1 + returns).cumprod()

    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 1000)),
        'high': prices * (1 + np.abs(np.random.normal(0.002, 0.001, 1000))),
        'low': prices * (1 - np.abs(np.random.normal(0.002, 0.001, 1000))),
        'close': prices,
        'volume': np.random.uniform(100, 1000, 1000)
    }, index=dates)

    # Create simple moving average strategy signals
    ma_short = data['close'].rolling(20).mean()
    ma_long = data['close'].rolling(50).mean()
    signals = pd.Series(0, index=data.index)
    signals[ma_short > ma_long] = 1
    signals[ma_short < ma_long] = -1

    # Execute backtest
    print("Executing backtest...")
    engine = BacktestEngine(
        data=data,
        signals=signals,
        transaction_cost=0.0006,
        slippage=0.0001,
        initial_capital=100000
    )

    results = engine.run()

    # Print results
    engine.print_summary()

    # Save results
    save_path = engine.save_results(data_source="test_strategy")

    # Test quick_backtest function
    print("\n\nTesting quick_backtest function...")
    returns, metrics = quick_backtest(data, signals)
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
