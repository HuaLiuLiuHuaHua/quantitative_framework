import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from typing import Optional

from shared.data_loader import load_local_data
from shared.visualization import plot_walkforward_performance
from shared.position_strategies import (
    BasePositionStrategy,
    TopBottomPositionStrategy,
    RotatingPositionStrategy
)

logger = logging.getLogger(__name__)

class CrossSectionalBacktester:
    """
    A class to perform cross-sectional backtesting of a factor.
    Supports both fixed and dynamic universes.

    Uses Strategy Pattern for position sizing - each factor can provide its own
    position_strategy to customize how positions are calculated.
    """

    def __init__(
        self,
        factor_calculator,
        position_strategy: Optional[BasePositionStrategy] = None,
        tickers: list[str] = None, # Optional for dynamic universe
        start_date: str = None,
        end_date: str = None,
        timeframe: str = '1d',
        quantiles: int = 5,
        top_n: int = None,
        bottom_n: int = None,
        position_weight: float = None,
        rebalance_freq: str = 'M',
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        stop_loss: float = None,
        min_volume: float = None,
        pre_load_lookback: int = None,
        universe_csv_path: str = None, # Path to the dynamic universe CSV
        universe_top_n: int = 40,     # Number of top assets to select from the universe CSV
        debug_mode: bool = True,
        leverage: float = 1.0,                    # 新增: 槓桿倍數
        stop_loss_pct: Optional[float] = None,    # 新增: 止損閾值 (例如 -0.5 表示 -50%)
        stop_loss_commission_bps: float = 55.0,   # 新增: 止損手續費 (basis points)
    ):
        self.factor_calculator = factor_calculator
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.quantiles = quantiles
        self.top_n = top_n
        self.bottom_n = bottom_n
        self.position_weight = position_weight
        self.rebalance_freq = rebalance_freq
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.benchmark_data = None
        self.all_data = {}

        self.universe_csv_path = universe_csv_path
        self.universe_top_n = universe_top_n
        self.universe_df = None
        self.debug_mode = debug_mode

        # 止損相關參數
        self.leverage = leverage
        self.stop_loss_pct = stop_loss_pct
        self.stop_loss_commission_bps = stop_loss_commission_bps

        # Position strategy: use provided strategy or default to TopBottom
        if position_strategy is None:
            self.position_strategy = TopBottomPositionStrategy()
        else:
            self.position_strategy = position_strategy

        if self.universe_csv_path:
            self._load_universe_data()
            if self.universe_df is not None and not self.universe_df.empty:
                all_possible_tickers = self.universe_df.iloc[:, 1:].astype(str).values.ravel()
                all_possible_tickers = [t for t in all_possible_tickers if t not in ['nan', 'None']]
                all_possible_tickers = np.unique(all_possible_tickers)
                self.tickers = [f"{t}USDT" for t in all_possible_tickers]

        if not self.tickers:
            raise ValueError("No tickers specified.")

        if pre_load_lookback is not None:
            self.all_data = self._load_all_data(lookback_days=pre_load_lookback)

    def _load_universe_data(self):
        try:
            df = pd.read_csv(self.universe_csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
            df = df.set_index('timestamp').sort_index()
            self.universe_df = df
        except Exception as e:
            print(f"Error loading dynamic universe data: {e}")
            self.universe_df = None

    def _get_universe_for_date(self, date: pd.Timestamp) -> list[str]:
        if self.universe_df is None or self.universe_df.empty:
            return self.tickers

        try:
            date = pd.to_datetime(date)
            latest_universe_data = self.universe_df[self.universe_df.index <= date]
            if latest_universe_data.empty:
                return []
            
            latest_universe_row = latest_universe_data.iloc[-1]
            
            universe = []
            for col in self.universe_df.columns:
                if col.lower() == 'timestamp':
                    continue
                value = latest_universe_row[col]
                if pd.notna(value) and str(value).strip() != "":
                    universe.append(str(value))
            
            universe = universe[:self.universe_top_n]
            universe_with_suffix = [f"{ticker}USDT" for ticker in universe]
            return universe_with_suffix

        except Exception as e:
            print(f"ERROR in _get_universe_for_date for date {date}: {e}")
            return []

    def _load_all_data(self, lookback_days: int = 0) -> dict[str, pd.DataFrame]:
        all_data = {}
        
        for ticker in self.tickers:
            df = load_local_data(
                symbol=ticker, 
                data_source=self.timeframe, 
                start_date=self.start_date, 
                end_date=self.end_date,
                lookback_days=lookback_days
            )
            if df is not None and not df.empty:
                all_data[ticker] = df

        self.benchmark_data = load_local_data(
            symbol='BTCUSDT',
            data_source=self.timeframe,
            start_date=self.start_date,
            end_date=self.end_date,
            lookback_days=lookback_days
        )

        return all_data

    def _get_rebalance_dates(self, factor_df: pd.DataFrame):
        """
        Generate rebalance dates based on the backtest period (start_date to end_date).

        This ensures that each walk-forward window has its own rebalance schedule,
        preventing all windows from sharing the same global rebalance grid.

        The rebalance dates are aligned to the start_date to ensure consistency
        across different windows.

        Returns:
            pd.DatetimeIndex: Rebalance dates within the backtest period
        """
        import logging
        logger = logging.getLogger(__name__)

        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)

        # Filter factor_df to the backtest period
        factor_in_period = factor_df[
            (factor_df.index >= start_date) & (factor_df.index <= end_date)
        ]

        if factor_in_period.empty:
            logger.warning(f"No factor data in period {start_date} to {end_date}")
            return pd.DatetimeIndex([])

        # Create a complete date range from start to end for alignment
        # This ensures rebalance dates are aligned to the start_date
        full_period_index = pd.date_range(
            start=start_date,
            end=end_date,
            freq='D'
        )

        # Create temporary series for resampling
        temp_series = pd.Series(index=full_period_index, dtype=float)
        rebalance_dates = pd.to_datetime(
            temp_series.resample(self.rebalance_freq).first().index
        )

        # Filter to only include dates where factor data exists
        available_dates = factor_in_period.index
        rebalance_dates = rebalance_dates[rebalance_dates.isin(available_dates)]

        return rebalance_dates

    def run(self, factor_df=None, **factor_params) -> dict:
        """
        Run backtest with support for both standard and rotating tranche strategies.

        If RotatingPositionStrategy is used, delegates to _run_with_rotating_tranches.
        Otherwise, uses standard rebalancing logic.
        """
        lookback_period = factor_params.get('lookback_period', 0)
        if not self.all_data:
            self.all_data = self._load_all_data(lookback_days=0)

        if not self.all_data:
            print("No data loaded. Aborting backtest.")
            return {}

        if factor_df is None:
            # Filter out strategy-specific parameters that shouldn't be passed to factor calculation
            factor_calc_params = {
                k: v for k, v in factor_params.items()
                if k not in ['holding_days', 'rebalance_freq_days']
            }
            factor_df = self.factor_calculator.calculate(self.all_data, **factor_calc_params)

        # Check if using RotatingPositionStrategy
        if isinstance(self.position_strategy, RotatingPositionStrategy):
            return self._run_with_rotating_tranches(factor_df, **factor_params)

        # Standard (non-rotating) backtest logic
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)
        rebalance_dates = self._get_rebalance_dates(factor_df)

        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Backtest period: {start_date.date()} to {end_date.date()}")
        logger.debug(f"Rebalance dates: {[d.date() for d in rebalance_dates]}")
        logger.debug(f"Number of rebalances: {len(rebalance_dates)}")

        all_dates_in_range = factor_df.index[(factor_df.index >= start_date) & (factor_df.index <= end_date)]
        if all_dates_in_range.empty:
            return {
                'equity_curve': pd.Series([1.0], index=[start_date]),
                'performance_summary': {'sharpe_ratio': 0, 'total_return': 0}
            }

        portfolio_returns = pd.Series(index=all_dates_in_range, dtype=float).fillna(0.0)
        sub_universe_returns = pd.Series(index=all_dates_in_range, dtype=float).fillna(0.0)
        
        first_trade_date = None
        previous_portfolio = set()
        turnover_values = []

        for i in range(len(rebalance_dates)):
            rebalance_date = rebalance_dates[i]
            decision_date = rebalance_date - pd.Timedelta(days=1)
            
            factor_data_for_decision = factor_df[factor_df.index <= decision_date]
            if factor_data_for_decision.empty:
                continue
            
            factor_snapshot = factor_data_for_decision.iloc[-1].dropna()

            active_universe = self._get_universe_for_date(rebalance_date)
            if not active_universe:
                continue
            
            factor_snapshot = factor_snapshot[factor_snapshot.index.isin(active_universe)]

            if factor_snapshot.empty:
                continue

            # Delegate position calculation to strategy
            # Gather strategy-specific parameters from factor_params
            strategy_params = {**factor_params}
            # Add legacy parameters for backward compatibility with TopBottomPositionStrategy
            if self.top_n is not None:
                strategy_params['top_n'] = self.top_n
            if self.bottom_n is not None:
                strategy_params['bottom_n'] = self.bottom_n

            position_weights = self.position_strategy.calculate_positions(
                factor_snapshot=factor_snapshot,
                all_data=self.all_data,
                decision_date=decision_date,
                **strategy_params
            )

            if not position_weights:
                continue

            # Calculate turnover
            current_portfolio = set(position_weights.keys())
            if previous_portfolio and (len(previous_portfolio) + len(current_portfolio)) > 0:
                turnover = len(previous_portfolio.symmetric_difference(current_portfolio)) / (len(previous_portfolio) + len(current_portfolio))
                turnover_values.append(turnover)
            previous_portfolio = current_portfolio

            # Holding period: execute at next day open to avoid lookahead bias
            # Timeline:
            #   decision_date (rebalance_date - 1): Analyze data, make decision using data up to this date
            #   rebalance_date: Signal generation day (we don't trade this day's movement)
            #   holding_start (rebalance_date + 1): Execute positions at next day's close, start capturing returns
            holding_start = rebalance_date + pd.Timedelta(days=1)
            holding_end = rebalance_dates[i+1] if i + 1 < len(rebalance_dates) else end_date
            holding_end = min(holding_end, end_date)

            # Calculate weighted returns for each position
            weighted_returns = []
            for asset, (weight, direction) in position_weights.items():
                if asset in self.all_data:
                    # Return calculation logic:
                    # - pct_change() at holding_start calculates: (close[holding_start] - close[holding_start-1]) / close[holding_start-1]
                    # - holding_start-1 = rebalance_date, so we enter at close[rebalance_date]
                    # - First return captures movement from close[rebalance_date] to close[rebalance_date+1]
                    # - This correctly excludes the rebalance_date movement (which triggered the signal)
                    # - Aligns with framework principle: signals at bar close → execute at next bar's close
                    asset_returns = self.all_data[asset]['close'].pct_change().loc[holding_start:holding_end]

                    # Apply direction: long = normal returns, short = negative returns
                    if direction == 'short':
                        asset_returns = -asset_returns

                    # Apply weight
                    weighted_asset_returns = asset_returns * weight
                    weighted_returns.append(weighted_asset_returns)

            if not weighted_returns:
                continue

            # Portfolio returns = Σ(weight × individual returns)
            combined_returns = pd.concat(weighted_returns, axis=1).sum(axis=1).fillna(0.0)
            portfolio_returns.update(combined_returns)

            sub_universe_period_returns = []
            for asset in active_universe:
                if asset in self.all_data:
                    asset_returns = self.all_data[asset]['close'].pct_change().loc[holding_start:holding_end]
                    sub_universe_period_returns.append(asset_returns)
            
            if sub_universe_period_returns:
                sub_universe_daily_returns = pd.concat(sub_universe_period_returns, axis=1).mean(axis=1).fillna(0.0)
                sub_universe_returns.update(sub_universe_daily_returns)

            if first_trade_date is None and len(combined_returns) > 0 and (combined_returns != 0).any():
                first_trade_date = combined_returns.index[0]

        cost = (self.commission_bps + self.slippage_bps) / 10000.0
        rebalance_points = portfolio_returns.index.intersection(pd.to_datetime(rebalance_dates))
        portfolio_returns[rebalance_points] -= cost

        # 應用止損邏輯 (永續合約強制平倉機制)
        if self.stop_loss_pct is not None and self.stop_loss_pct < 0:
            portfolio_returns = self._apply_stop_loss(
                portfolio_returns,
                pd.to_datetime(rebalance_dates)
            )

        equity_curve = (1 + portfolio_returns).cumprod()

        trading_returns = portfolio_returns[portfolio_returns.index >= first_trade_date] if first_trade_date else portfolio_returns
        
        sharpe_ratio = 0
        if trading_returns.std() > 0:
            sharpe_ratio = np.sqrt(365) * trading_returns.mean() / trading_returns.std()

        avg_turnover = np.mean(turnover_values) if turnover_values else 0

        sub_universe_trading_returns = sub_universe_returns[sub_universe_returns.index >= first_trade_date] if first_trade_date else sub_universe_returns
        subniverse_sharpe = 0
        if sub_universe_trading_returns.std() > 0:
            subniverse_sharpe = np.sqrt(365) * sub_universe_trading_returns.mean() / sub_universe_trading_returns.std()

        # Calculate total turnover value and num rebalances for detailed reporting
        total_turnover_value = sum(turnover_values)
        num_rebalances = len(rebalance_dates)

        # Calculate max drawdown
        if not equity_curve.empty:
            running_max = equity_curve.cummax()
            drawdown = (equity_curve - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0

        performance_summary = {
            'sharpe_ratio': sharpe_ratio,
            'total_return': equity_curve.iloc[-1] - 1 if not equity_curve.empty else 0,
            'max_drawdown': max_drawdown,  # BUGFIX: Added missing max_drawdown calculation
            'daily_return': trading_returns,
            'first_trade_date': first_trade_date.strftime('%Y-%m-%d') if first_trade_date is not None else None,
            'turnover': avg_turnover,
            'sub_universe_sharpe': subniverse_sharpe,
            'total_turnover_value': total_turnover_value,  # NEW: For aggregated reporting
            'num_rebalances': num_rebalances  # NEW: For aggregated reporting
        }

        return {
            'equity_curve': equity_curve,
            'performance_summary': performance_summary,
        }

    def _run_with_rotating_tranches(self, factor_df: pd.DataFrame, **factor_params) -> dict:
        """
        Run backtest with rotating tranche strategy.

        This method handles overlapping position lifetimes where:
        - Capital is divided into (holding_days / rebalance_freq_days) tranches
        - Every rebalance_freq_days, a new tranche opens positions
        - Each tranche holds for holding_days, then closes
        - Multiple tranches are active simultaneously

        Args:
            factor_df: Factor values DataFrame (dates × symbols)
            **factor_params: Parameters including holding_days, rebalance_freq_days

        Returns:
            Dictionary with equity_curve and performance_summary
        """
        import logging
        logger = logging.getLogger(__name__)

        # Extract rotating strategy parameters
        holding_days = factor_params.get('holding_days', 30)
        rebalance_freq_days = factor_params.get('rebalance_freq_days', 1)

        if holding_days <= 0 or rebalance_freq_days <= 0:
            raise ValueError(
                f"holding_days ({holding_days}) and rebalance_freq_days ({rebalance_freq_days}) "
                "must be positive"
            )

        n_tranches = holding_days / rebalance_freq_days
        logger.info(f"Rotating tranche strategy: holding_days={holding_days}, "
                    f"rebalance_freq_days={rebalance_freq_days}, n_tranches={n_tranches:.1f}")

        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)

        # Get all trading dates in the backtest period
        all_dates_in_range = factor_df.index[
            (factor_df.index >= start_date) & (factor_df.index <= end_date)
        ]

        if all_dates_in_range.empty:
            return {
                'equity_curve': pd.Series([1.0], index=[start_date]),
                'performance_summary': {'sharpe_ratio': 0, 'total_return': 0}
            }

        # Generate rebalance dates based on rebalance_freq_days
        rebalance_dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq=f'{rebalance_freq_days}D'
        )
        # Filter to only include dates in factor_df
        rebalance_dates = rebalance_dates[rebalance_dates.isin(factor_df.index)]

        logger.debug(f"Backtest period: {start_date.date()} to {end_date.date()}")
        logger.debug(f"Rebalance dates: {len(rebalance_dates)} dates")
        logger.debug(f"First rebalance: {rebalance_dates[0].date() if len(rebalance_dates) > 0 else 'N/A'}")
        logger.debug(f"Last rebalance: {rebalance_dates[-1].date() if len(rebalance_dates) > 0 else 'N/A'}")

        # Initialize return series
        portfolio_returns = pd.Series(index=all_dates_in_range, dtype=float).fillna(0.0)
        sub_universe_returns = pd.Series(index=all_dates_in_range, dtype=float).fillna(0.0)

        # Track active tranches: {tranche_id: {'entry_date': date, 'exit_date': date, 'positions': {...}}}
        active_tranches = {}
        next_tranche_id = 0
        first_trade_date = None
        previous_portfolio = set()
        turnover_values = []

        # Iterate through all trading dates
        for current_date in all_dates_in_range:
            # Check if today is a rebalance date (open new tranche)
            if current_date in rebalance_dates:
                decision_date = current_date - pd.Timedelta(days=1)

                # Get factor snapshot for decision
                factor_data_for_decision = factor_df[factor_df.index <= decision_date]
                if not factor_data_for_decision.empty:
                    factor_snapshot = factor_data_for_decision.iloc[-1].dropna()

                    # Filter to active universe
                    active_universe = self._get_universe_for_date(current_date)
                    if active_universe:
                        factor_snapshot = factor_snapshot[factor_snapshot.index.isin(active_universe)]

                        if not factor_snapshot.empty:
                            # Calculate positions for new tranche
                            strategy_params = {**factor_params}
                            if self.top_n is not None:
                                strategy_params['top_n'] = self.top_n
                            if self.bottom_n is not None:
                                strategy_params['bottom_n'] = self.bottom_n

                            position_weights = self.position_strategy.calculate_positions(
                                factor_snapshot=factor_snapshot,
                                all_data=self.all_data,
                                decision_date=decision_date,
                                **strategy_params
                            )

                            if position_weights:
                                # Create new tranche
                                tranche_entry_date = current_date
                                tranche_exit_date = current_date + pd.Timedelta(days=holding_days)

                                active_tranches[next_tranche_id] = {
                                    'entry_date': tranche_entry_date,
                                    'exit_date': tranche_exit_date,
                                    'positions': position_weights
                                }

                                logger.debug(
                                    f"Opened tranche {next_tranche_id} on {current_date.date()}: "
                                    f"{len(position_weights)} positions, exit on {tranche_exit_date.date()}"
                                )

                                next_tranche_id += 1

            # Calculate returns from all active tranches on current_date
            daily_return = 0.0

            for tranche_id, tranche_info in list(active_tranches.items()):
                # Check if tranche should be closed (after completing holding period)
                if current_date > tranche_info['exit_date']:
                    logger.debug(f"Closed tranche {tranche_id} after {current_date.date()}")
                    del active_tranches[tranche_id]
                    continue

                # Calculate returns for this tranche
                for asset, (weight, direction) in tranche_info['positions'].items():
                    if asset in self.all_data:
                        asset_data = self.all_data[asset]
                        if current_date in asset_data.index:
                            # Get today's return (correctly without lookahead bias)
                            asset_returns_series = asset_data['close'].pct_change()
                            if current_date in asset_returns_series.index:
                                asset_return = asset_returns_series.loc[current_date]

                                if not np.isnan(asset_return):
                                    # Apply direction
                                    if direction == 'short':
                                        asset_return = -asset_return

                                    # Add weighted return to daily total
                                    daily_return += weight * asset_return

            portfolio_returns.loc[current_date] = daily_return

            # Track first trade date
            if first_trade_date is None and daily_return != 0:
                first_trade_date = current_date

            # Calculate sub-universe returns (equal-weighted benchmark)
            active_universe = self._get_universe_for_date(current_date)
            if active_universe:
                universe_returns = []
                for asset in active_universe:
                    if asset in self.all_data:
                        asset_data = self.all_data[asset]
                        if current_date in asset_data.index:
                            # Get today's return (correctly without lookahead bias)
                            asset_returns_series = asset_data['close'].pct_change()
                            if current_date in asset_returns_series.index:
                                asset_return = asset_returns_series.loc[current_date]
                                if not np.isnan(asset_return):
                                    universe_returns.append(asset_return)

                if universe_returns:
                    sub_universe_returns.loc[current_date] = np.mean(universe_returns)

        # Apply transaction costs at rebalance points
        # For rotating strategy, we only open ONE new tranche at each rebalance
        # So cost should be proportional to tranche size, not full portfolio
        base_cost = (self.commission_bps + self.slippage_bps) / 10000.0
        tranche_cost = base_cost * (1.0 / n_tranches) if n_tranches > 0 else base_cost
        rebalance_points = portfolio_returns.index.intersection(pd.to_datetime(rebalance_dates))
        portfolio_returns.loc[rebalance_points] -= tranche_cost

        # Calculate equity curve
        equity_curve = (1 + portfolio_returns).cumprod()

        # Calculate performance metrics
        trading_returns = portfolio_returns[portfolio_returns.index >= first_trade_date] if first_trade_date else portfolio_returns

        sharpe_ratio = 0
        if trading_returns.std() > 0:
            sharpe_ratio = np.sqrt(365) * trading_returns.mean() / trading_returns.std()

        sub_universe_trading_returns = sub_universe_returns[sub_universe_returns.index >= first_trade_date] if first_trade_date else sub_universe_returns
        subniverse_sharpe = 0
        if sub_universe_trading_returns.std() > 0:
            subniverse_sharpe = np.sqrt(365) * sub_universe_trading_returns.mean() / sub_universe_trading_returns.std()

        # Calculate max drawdown
        if not equity_curve.empty:
            running_max = equity_curve.cummax()
            drawdown = (equity_curve - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0

        # Calculate turnover (simplified for rotating strategy)
        # Note: This is approximate as we're opening new tranches but not closing all positions
        avg_turnover = 1.0 / n_tranches if n_tranches > 0 else 0

        performance_summary = {
            'sharpe_ratio': sharpe_ratio,
            'total_return': equity_curve.iloc[-1] - 1 if not equity_curve.empty else 0,
            'max_drawdown': max_drawdown,
            'daily_return': trading_returns,
            'first_trade_date': first_trade_date.strftime('%Y-%m-%d') if first_trade_date is not None else None,
            'turnover': avg_turnover,
            'sub_universe_sharpe': subniverse_sharpe,
            'total_turnover_value': avg_turnover * len(rebalance_dates),
            'num_rebalances': len(rebalance_dates),
            'num_tranches': n_tranches,
            'holding_days': holding_days,
            'rebalance_freq_days': rebalance_freq_days
        }

        logger.info(f"Rotating tranche backtest complete: Sharpe={sharpe_ratio:.2f}, "
                    f"Return={performance_summary['total_return']*100:.2f}%")

        return {
            'equity_curve': equity_curve,
            'performance_summary': performance_summary,
        }

    def _apply_stop_loss(
        self,
        returns: pd.Series,
        rebalance_dates: pd.DatetimeIndex
    ) -> pd.Series:
        """
        應用止損邏輯 (模擬永續合約強制平倉機制)

        永續合約邏輯:
        1. 當虧損達到止損閾值時,交易所強制平倉
        2. 平倉後賬戶只剩下剩餘資金,沒有倉位
        3. 持有現金直到下一個再平衡日
        4. 再平衡日重新開倉

        例如: 2x 槓桿, 止損 -50%
        - Day 1: 開倉 $20,000 (2x 槓桿)
        - Day 2: 虧損 60% → 觸發強制平倉 → 剩餘 $5,000, 無倉位
        - Day 3-4: 持有現金 (收益 0%)
        - Day 5: 再平衡日重新開倉

        Args:
            returns: 原始日收益率序列
            rebalance_dates: 再平衡日期索引

        Returns:
            應用止損後的收益率序列
        """
        if self.stop_loss_pct is None or self.stop_loss_pct >= 0:
            return returns

        adjusted_returns = returns.copy()
        liquidated = False  # 是否已被強制平倉 (無倉位狀態)
        stop_loss_count = 0

        for i in range(len(returns)):
            current_date = returns.index[i]

            # 檢查是否是再平衡日 (可以重新開倉)
            if current_date in rebalance_dates:
                if liquidated:
                    logger.debug(
                        f"再平衡日: {current_date.date()}, "
                        f"結束強制平倉狀態, 重新開倉"
                    )
                liquidated = False  # 重置狀態,重新開倉

            # 如果處於強制平倉狀態 (無倉位,持有現金)
            if liquidated:
                adjusted_returns.iloc[i] = 0.0  # 現金收益為 0
                continue

            # 檢查是否觸發止損 (強制平倉)
            if returns.iloc[i] < self.stop_loss_pct:
                # 截斷收益為止損值
                adjusted_returns.iloc[i] = self.stop_loss_pct

                # 扣除強制平倉手續費 (通常高於正常手續費)
                extra_commission = (
                    self.stop_loss_commission_bps - self.commission_bps
                ) / 10000.0
                adjusted_returns.iloc[i] -= extra_commission

                # 標記為強制平倉狀態
                liquidated = True
                stop_loss_count += 1

                logger.info(
                    f"強制平倉觸發: {current_date.date()}, "
                    f"原始虧損={returns.iloc[i]*100:.2f}%, "
                    f"止損截斷={self.stop_loss_pct*100:.2f}%, "
                    f"手續費={extra_commission*100:.3f}%, "
                    f"進入持有現金狀態"
                )

        if stop_loss_count > 0:
            logger.info(
                f"總強制平倉次數: {stop_loss_count}, "
                f"止損保護了資金免於進一步虧損"
            )

        return adjusted_returns
