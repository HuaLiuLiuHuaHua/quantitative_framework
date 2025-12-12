import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple

# Add project root to Python path to allow module imports
sys.path.append(str(Path(__file__).parents[2]))

from shared.data_loader import load_local_data

# Set up logging
logger = logging.getLogger(__name__)

class NegativeMomentumFactor:
    """
    A simple negative momentum (reversal) factor.
    It calculates the negative percentage price change over a given period.
    High values correspond to assets that have fallen the most (buy signal).
    Low values correspond to assets that have risen the most (sell signal).
    """

    def __init__(self, factor_name: str = "NegativeMomentum"):
        """
        Initializes the NegativeMomentumFactor.

        Args:
            factor_name (str): The name of the factor.
        """
        self.factor_name = factor_name
        self.parameters = {}

    def calculate(
        self,
        price_data_dict: Dict[str, pd.DataFrame],
        lookback_period: int = 20,
        min_volume: float = 0
    ) -> pd.DataFrame:
        """
        Calculate negative momentum (reversal) factor values.

        This factor is the inverse of the percentage price change over the lookback period.
        A high positive value indicates a significant price drop, signaling a potential buy.
        A high negative value indicates a significant price rise, signaling a potential sell.

        Args:
            price_data_dict (dict): Dictionary of price dataframes {symbol: DataFrame}.
            lookback_period (int): Period to calculate momentum over (in trading days).
            min_volume (float): Minimum volume threshold (not used in this version).

        Returns:
            pd.DataFrame: Factor values time series (index=dates, columns=symbols).
        """
        if not price_data_dict:
            return pd.DataFrame()

        all_dates = pd.DatetimeIndex([])
        for df in price_data_dict.values():
            if df is not None and not df.empty:
                all_dates = all_dates.union(df.index)

        if len(all_dates) == 0:
            return pd.DataFrame()

        all_dates = all_dates.sort_values().unique()

        factor_df = pd.DataFrame(index=all_dates, columns=list(price_data_dict.keys()), dtype=float)

        for symbol, df in price_data_dict.items():
            if df is None or df.empty:
                continue

            try:
                # Calculate negative momentum to favor reversal.
                # A positive value means the price went down (a buy signal for reversal).
                # A negative value means the price went up (a sell signal for reversal).
                momentum = -df['close'].pct_change(periods=lookback_period).shift(1)

                factor_df[symbol] = momentum.reindex(all_dates)

            except Exception as e:
                logger.error(f"Error calculating negative momentum for {symbol}: {e}", exc_info=True)
                continue

        factor_df = factor_df.dropna(how='all')

        return factor_df

    def get_parameter_grid(self) -> Dict[str, Tuple]:
        """
        Returns the parameter grid for optimization.
        """
        return {
            'lookback_period': (20, 200, 20), # (start, end, step)
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Returns the default parameters for the factor.
        """
        return {
            'lookback_period': 156,
        }

    def __call__(self, data: Dict[str, pd.DataFrame], **params) -> pd.DataFrame:
        """Makes the factor object callable."""
        return self.calculate(data, **params)

# --- Main execution block for testing ---
if __name__ == "__main__":
    print("=" * 70)
    print("Running Negative Momentum Factor Calculation Test")
    print("=" * 70)

    # --- Configuration for the test ---
    test_tickers = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    start_date = '2024-01-01'
    end_date = '2024-06-30'
    timeframe = '1d'

    # --- Data Loading ---
    print(f"Loading data for {test_tickers} from {start_date} to {end_date}...")
    price_data_dict = {}
    for ticker in test_tickers:
        df = load_local_data(
            symbol=ticker,
            start_date=start_date,
            end_date=end_date,
            data_source=timeframe
        )
        if df is not None and not df.empty:
            price_data_dict[ticker] = df
    
    if not price_data_dict:
        raise ValueError("No data loaded for any test tickers.")

    # --- Factor Calculation ---
    # Initialize the factor
    momentum_factor = NegativeMomentumFactor()
    
    # Get default parameters
    default_params = momentum_factor.get_default_parameters()
    print(f"\nCalculating factor with default parameters: {default_params}")

    # Calculate the factor values
    factor_values = momentum_factor.calculate(price_data_dict, **default_params)

    # --- Output Results ---
    print("\n--- Calculated Factor Values (Last 5 days) ---")
    print(factor_values.tail())
    print("\n" + "=" * 70)
    print("Test finished successfully.")
    print("=" * 70)