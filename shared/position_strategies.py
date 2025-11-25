"""
Position Strategy Classes for Cross-Sectional Backtesting

This module implements the Strategy Pattern to decouple position sizing logic
from the backtesting engine. Each factor can define its own position strategy.

Created: 2025-11-05
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class BasePositionStrategy(ABC):
    """
    Abstract base class for position sizing strategies.

    Each strategy determines:
    1. Which assets to include in the portfolio
    2. The weight/size of each position
    3. The direction (long/short) of each position
    """

    def __init__(self, strategy_name: str = "BaseStrategy"):
        self.strategy_name = strategy_name

    @abstractmethod
    def calculate_positions(
        self,
        factor_snapshot: pd.Series,
        all_data: Dict[str, pd.DataFrame],
        decision_date: pd.Timestamp,
        **kwargs
    ) -> Dict[str, Tuple[float, str]]:
        """
        Calculate position weights and directions for the current rebalance.

        Args:
            factor_snapshot: Factor values for all assets at decision time (Series with asset names as index)
            all_data: Dictionary of price data {asset: DataFrame}
            decision_date: Date when position decision is made (use data up to and including this date)
            **kwargs: Additional strategy-specific parameters

        Returns:
            Dictionary mapping asset -> (weight, direction)
            - weight: Position size as fraction of portfolio (0.0 to 1.0)
            - direction: 'long' or 'short'

        Example:
            {
                'BTCUSDT': (0.2, 'long'),
                'ETHUSDT': (0.3, 'short'),
                ...
            }
        """
        pass

    @abstractmethod
    def get_required_params(self) -> List[str]:
        """
        Return list of required parameter names for this strategy.

        Returns:
            List of parameter names (e.g., ['top_n', 'bottom_n'])
        """
        pass


class TopBottomPositionStrategy(BasePositionStrategy):
    """
    Standard top/bottom selection strategy.

    Selects top N assets (highest factor values) for long positions
    and bottom N assets (lowest factor values) for short positions.

    Each position is equally weighted within its group.
    """

    def __init__(self):
        super().__init__(strategy_name="TopBottomSelection")

    def calculate_positions(
        self,
        factor_snapshot: pd.Series,
        all_data: Dict[str, pd.DataFrame],
        decision_date: pd.Timestamp,
        top_n: int = 5,
        bottom_n: int = 5,
        **kwargs
    ) -> Dict[str, Tuple[float, str]]:
        """
        Calculate positions using top/bottom selection.

        Args:
            factor_snapshot: Factor values for all assets
            all_data: Price data dictionary (not used in this strategy)
            decision_date: Decision date (not used in this strategy)
            top_n: Number of assets to long (highest factor values)
            bottom_n: Number of assets to short (lowest factor values)

        Returns:
            Position dictionary with equal weights for selected assets
        """
        if factor_snapshot.empty:
            return {}

        # Rank assets by factor value (descending)
        ranked_assets = factor_snapshot.sort_values(ascending=False)

        # Select top and bottom assets
        long_assets = ranked_assets.head(top_n).index.tolist()
        short_assets = ranked_assets.tail(bottom_n).index.tolist()

        total_positions = len(long_assets) + len(short_assets)
        if total_positions == 0:
            return {}

        # Equal weight for all positions
        equal_weight = 1.0 / total_positions

        positions = {}
        for asset in long_assets:
            positions[asset] = (equal_weight, 'long')

        for asset in short_assets:
            positions[asset] = (equal_weight, 'short')

        return positions

    def get_required_params(self) -> List[str]:
        return ['top_n', 'bottom_n']


class ExcessReturnPositionStrategy(BasePositionStrategy):
    """
    Relative return mean reversion weighting strategy for market-neutral portfolio.

    This strategy implements mean reversion by:
    1. Calculating each asset's return over lookback_period
    2. Computing market average return (mean of all assets)
    3. Computing relative return: r_i - <r_j> (deviation from market)
    4. Using negative relative return to determine weights (mean reversion signal)
    5. Allocating weights proportionally to |relative_return|
    
    Key Formula (Mean Reversion):
    w_i = - (r_i - <r_j>) / Σ|r_k - <r_j>|
    
    Logic:
    - Negative relative return (underperformers) → positive weight (LONG)
    - Positive relative return (overperformers) → negative weight (SHORT)
    - Weight magnitude scales with strength of relative performance deviation
    - Total absolute weight sum always equals 1.0 (market neutral)
    """

    def __init__(self):
        super().__init__(strategy_name="RelativeReturnMeanReversion")

    def calculate_positions(
        self,
        factor_snapshot: pd.Series,
        all_data: Dict[str, pd.DataFrame],
        decision_date: pd.Timestamp,
        lookback_period: int = 20,
        **kwargs
    ) -> Dict[str, Tuple[float, str]]:
        """
        Calculate market-neutral positions using relative return weighting.

        Formula:
        1. Relative Return: rel_ret_i = r_i - <r_j>
        2. Weight: w_i = -(rel_ret_i) / Σ|rel_ret_k|
        3. If w_i > 0: LONG position
           If w_i < 0: SHORT position

        Args:
            factor_snapshot: Factor values (used to determine asset universe)
            all_data: Price data dictionary (required for return calculation)
            decision_date: Date to calculate returns up to (avoid lookahead bias)
            lookback_period: Number of days to calculate returns over

        Returns:
            Position dictionary mapping asset -> (abs_weight, direction)
            where abs_weight is the absolute value of position weight
            and direction is 'long' or 'short' based on signal sign

        Raises:
            ValueError: If lookback_period is not a positive integer
        """
        # Validate parameters
        if lookback_period <= 0:
            raise ValueError(f"lookback_period must be positive, got {lookback_period}")

        if factor_snapshot.empty or not all_data:
            return {}

        # ========== Step 1: Calculate period returns for each asset ==========
        asset_period_returns = {}
        for asset in factor_snapshot.index:
            if asset not in all_data:
                continue

            asset_data = all_data[asset]
            # Use data up to and including decision_date (avoid lookahead bias)
            decision_data = asset_data[asset_data.index <= decision_date]

            # Need lookback_period + 1 bars to calculate lookback_period return
            # Example: lookback_period=5 requires 6 bars (bars 0-5)
            if len(decision_data) >= lookback_period + 1:
                price_current = decision_data['close'].iloc[-1]
                price_past = decision_data['close'].iloc[-(lookback_period + 1)]

                if price_past != 0:  # Avoid division by zero
                    period_return = (price_current - price_past) / price_past
                    asset_period_returns[asset] = period_return

        if not asset_period_returns:
            logger.debug(f"No valid asset returns for decision date {decision_date.date()}")
            return {}

        # ========== Step 2: Calculate market average return ==========
        market_avg_return = np.mean(list(asset_period_returns.values()))
        
        logger.debug(
            f"Decision date: {decision_date.date()}, "
            f"Market avg return: {market_avg_return:.4f}, "
            f"Assets with returns: {len(asset_period_returns)}"
        )

        # ========== Step 3: Calculate relative returns ==========
        # relative_return_i = r_i - <r_j>
        relative_returns = {
            asset: ret - market_avg_return
            for asset, ret in asset_period_returns.items()
        }

        # ========== Step 4: Calculate normalization factor ==========
        # Σ|r_k - <r_j>| - sum of absolute relative returns
        total_abs_relative_return = sum(
            abs(rel_ret) for rel_ret in relative_returns.values()
        )

        if total_abs_relative_return == 0:
            logger.debug(f"Skipping {decision_date.date()}: zero total relative return")
            return {}

        # ========== Step 5: Calculate weights using mean reversion formula ==========
        # w_i = - (r_i - <r_j>) / Σ|r_k - <r_j>|
        # 
        # Interpretation:
        # - Negative relative return (underperformer) → negative weight → positive signed weight → LONG
        # - Positive relative return (overperformer) → positive weight → negative signed weight → SHORT
        
        positions = {}
        for asset, rel_ret in relative_returns.items():
            # Raw weight including sign (negative of relative return)
            signed_weight = -rel_ret / total_abs_relative_return
            
            # Skip assets with zero or negligible weight
            if abs(signed_weight) < 1e-10:
                continue
            
            # Absolute weight for position sizing
            abs_weight = abs(signed_weight)
            
            # Direction based on sign
            direction = 'long' if signed_weight > 0 else 'short'
            
            positions[asset] = (abs_weight, direction)
            
            logger.debug(
                f"  {asset}: r={asset_period_returns[asset]:.4f}, "
                f"rel_ret={rel_ret:.4f}, signed_weight={signed_weight:.4f}, "
                f"direction={direction}"
            )

        # ========== Step 6: Validate market neutrality ==========
        # Verify that sum of absolute weights equals 1.0
        total_weight = sum(weight for weight, _ in positions.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            logger.warning(
                f"Total absolute weight not 1.0: {total_weight:.6f}. "
                f"This may indicate numerical issues. Assets: {len(positions)}"
            )

        return positions

    def get_required_params(self) -> List[str]:
        return ['lookback_period']


class PercentilePositionStrategy(BasePositionStrategy):
    """
    Percentile-based selection strategy.

    Selects assets based on percentile thresholds rather than fixed counts.
    For example: long top 20% of assets, short bottom 20% of assets.

    Useful when universe size varies over time (dynamic universes).
    """

    def __init__(self):
        super().__init__(strategy_name="PercentileSelection")

    def calculate_positions(
        self,
        factor_snapshot: pd.Series,
        all_data: Dict[str, pd.DataFrame],
        decision_date: pd.Timestamp,
        long_percentile: float = 20.0,
        short_percentile: float = 20.0,
        **kwargs
    ) -> Dict[str, Tuple[float, str]]:
        """
        Calculate positions using percentile thresholds.

        Args:
            factor_snapshot: Factor values for all assets
            all_data: Price data dictionary (not used)
            decision_date: Decision date (not used)
            long_percentile: Percentile threshold for long positions (e.g., 20 = top 20%)
            short_percentile: Percentile threshold for short positions (e.g., 20 = bottom 20%)

        Returns:
            Position dictionary with equal weights
        """
        if factor_snapshot.empty:
            return {}

        universe_size = len(factor_snapshot)
        top_n = round(universe_size * long_percentile / 100)
        bottom_n = round(universe_size * short_percentile / 100)

        # Ensure at least 1 asset if percentile > 0
        if long_percentile > 0:
            top_n = max(1, top_n)
        if short_percentile > 0:
            bottom_n = max(1, bottom_n)

        # Rank assets by factor value
        ranked_assets = factor_snapshot.sort_values(ascending=False)

        # Select based on calculated counts
        long_assets = ranked_assets.head(top_n).index.tolist()
        short_assets = ranked_assets.tail(bottom_n).index.tolist()

        total_positions = len(long_assets) + len(short_assets)
        if total_positions == 0:
            return {}

        equal_weight = 1.0 / total_positions

        positions = {}
        for asset in long_assets:
            positions[asset] = (equal_weight, 'long')

        for asset in short_assets:
            positions[asset] = (equal_weight, 'short')

        return positions

    def get_required_params(self) -> List[str]:
        return ['long_percentile', 'short_percentile']


class RotatingPositionStrategy(BasePositionStrategy):
    """
    Rotating tranche position strategy with overlapping holding periods.

    This strategy divides capital into multiple tranches that rotate over time:
    - Total capital split into (holding_days / rebalance_freq_days) tranches
    - Every rebalance_freq_days, one tranche opens new positions
    - Each tranche holds for holding_days, then closes
    - Multiple tranches can be active simultaneously with overlapping holding periods

    Example 1: holding_days=30, rebalance_freq_days=1
        - 30 tranches (30/1 = 30)
        - Each day, 1/30 of capital enters new positions holding for 30 days
        - At any time, 30 tranches are active with staggered entry dates

    Example 2: holding_days=10, rebalance_freq_days=2
        - 5 tranches (10/2 = 5)
        - Every 2 days, 1/5 of capital enters new positions holding for 10 days
        - At any time, up to 5 tranches are active

    Note: This strategy requires special handling in the backtester to track
    multiple tranches with different entry/exit dates.
    """

    def __init__(self):
        super().__init__(strategy_name="RotatingTranches")
        # Track active tranches: {tranche_id: {'entry_date': date, 'exit_date': date, 'positions': {...}}}
        self.active_tranches = {}
        self.next_tranche_id = 0

    def calculate_positions(
        self,
        factor_snapshot: pd.Series,
        all_data: Dict[str, pd.DataFrame],
        decision_date: pd.Timestamp,
        top_n: int = 5,
        bottom_n: int = 5,
        holding_days: int = 30,
        rebalance_freq_days: int = 1,
        **kwargs
    ) -> Dict[str, Tuple[float, str]]:
        """
        Calculate positions for rotating tranche strategy.

        This method is called by the backtester at each rebalance point.
        It returns positions for a NEW tranche that will be opened.

        Args:
            factor_snapshot: Factor values for all assets at decision time
            all_data: Price data dictionary (not used in basic version)
            decision_date: Date when position decision is made
            top_n: Number of assets to long (highest factor values)
            bottom_n: Number of assets to short (lowest factor values)
            holding_days: Number of days each tranche holds positions
            rebalance_freq_days: Frequency of opening new tranches (in days)

        Returns:
            Position dictionary for the new tranche with fractional weights
            Weight per position = (1 / n_tranches) / (top_n + bottom_n)

            Note: The returned positions are for ONE NEW TRANCHE only.
            The backtester must handle combining multiple active tranches.
        """
        if factor_snapshot.empty:
            return {}

        # Validate parameters
        if holding_days <= 0 or rebalance_freq_days <= 0:
            raise ValueError(
                f"holding_days ({holding_days}) and rebalance_freq_days ({rebalance_freq_days}) "
                "must be positive"
            )

        if holding_days % rebalance_freq_days != 0:
            logger.warning(
                f"holding_days ({holding_days}) is not divisible by rebalance_freq_days ({rebalance_freq_days}). "
                f"This may lead to uneven capital allocation."
            )

        # Calculate number of tranches
        n_tranches = holding_days / rebalance_freq_days

        # Rank assets by factor value (descending)
        ranked_assets = factor_snapshot.sort_values(ascending=False)

        # Select top and bottom assets
        long_assets = ranked_assets.head(top_n).index.tolist()
        short_assets = ranked_assets.tail(bottom_n).index.tolist()

        total_positions = len(long_assets) + len(short_assets)
        if total_positions == 0:
            return {}

        # Each tranche gets 1/n_tranches of total capital
        # Each position within the tranche gets equal weight
        # Final weight per position = (1 / n_tranches) / total_positions
        tranche_weight = 1.0 / n_tranches
        per_position_weight = tranche_weight / total_positions

        positions = {}
        for asset in long_assets:
            positions[asset] = (per_position_weight, 'long')

        for asset in short_assets:
            positions[asset] = (per_position_weight, 'short')

        logger.debug(
            f"RotatingPositionStrategy: decision_date={decision_date.date()}, "
            f"n_tranches={n_tranches:.1f}, tranche_weight={tranche_weight:.4f}, "
            f"positions={len(positions)}, per_position_weight={per_position_weight:.4f}"
        )

        return positions

    def get_required_params(self) -> List[str]:
        return ['top_n', 'bottom_n', 'holding_days', 'rebalance_freq_days']
