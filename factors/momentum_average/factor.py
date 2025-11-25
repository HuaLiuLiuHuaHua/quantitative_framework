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

class MovingAverageCrossFactor:
    """
    Calculates the percentage difference between a short-term and a long-term moving average.
    """

    def __init__(self, factor_name: str = "MovingAverageCross"):
        """
        Initializes the MovingAverageCrossFactor.

        Args:
            factor_name (str): The name of the factor.
        """
        self.factor_name = factor_name
        self.parameters = {}

    def calculate(
        self,
        price_data_dict: Dict[str, pd.DataFrame],
        short_window: int = 20,
        long_window: int = 60,
        min_volume: float = 0
    ) -> pd.DataFrame:
        """
        Calculate factor values based on the percentage difference between short and long term moving averages.

        Args:
            price_data_dict (dict): Dictionary of price dataframes {symbol: DataFrame}
            short_window (int): The window for the short-term moving average.
            long_window (int): The window for the long-term moving average.
            min_volume (float): Minimum volume threshold (not used in this version).

        Returns:
            pd.DataFrame: Factor values time series (index=dates, columns=symbols).
        """
        if not price_data_dict:
            return pd.DataFrame()

        # Collect all unique dates from all assets
        all_dates = pd.DatetimeIndex([])
        for df in price_data_dict.values():
            if df is not None and not df.empty:
                all_dates = all_dates.union(df.index)

        if len(all_dates) == 0:
            return pd.DataFrame()

        all_dates = all_dates.sort_values().unique()

        # Initialize result DataFrame: rows=dates, columns=symbols
        factor_df = pd.DataFrame(index=all_dates, columns=list(price_data_dict.keys()), dtype=float)

        # Calculate factor values for each asset across all dates
        # OPTIMIZATION: Use vectorized operations and avoid repeated reindex
        for symbol, df in price_data_dict.items():
            if df is None or df.empty:
                continue

            try:
                # Calculate moving averages using vectorized pandas operations
                close_prices = df['close']
                short_ma = close_prices.rolling(window=short_window, min_periods=short_window).mean()
                long_ma = close_prices.rolling(window=long_window, min_periods=long_window).mean()

                # Calculate the percentage difference and shift to avoid lookahead bias
                # Using vectorized operations for better performance
                with np.errstate(divide='ignore', invalid='ignore'):
                    factor_value = ((short_ma - long_ma) / long_ma).shift(1)

                # BUGFIX: Handle inf values from division by zero
                factor_value = factor_value.replace([np.inf, -np.inf], np.nan)

                # Assign to result DataFrame (reindex to match all_dates)
                # Only reindex if necessary (reduces overhead)
                if not df.index.equals(all_dates):
                    factor_df[symbol] = factor_value.reindex(all_dates)
                else:
                    factor_df[symbol] = factor_value

            except Exception as e:
                logger.error(f"Error calculating moving average cross for {symbol}: {e}", exc_info=True)
                continue

        # Remove rows where ALL values are NaN
        factor_df = factor_df.dropna(how='all')

        return factor_df

    def get_parameter_grid(self) -> Dict[str, Tuple]:
        """
        Returns the parameter grid for optimization.
        """
        return {
            'short_window': (10, 50, 5),   # (start, end, step)
            'long_window': (50, 200, 20), # (start, end, step)
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Returns the default parameters for the factor.
        """
        return {
            'short_window': 20,
            'long_window': 60,
        }

    def __call__(self, data: Dict[str, pd.DataFrame], **params) -> pd.DataFrame:
        """Makes the factor object callable."""
        return self.calculate(data, **params)

# --- Main execution block for testing ---
if __name__ == "__main__":
    print("=" * 70)
    print("Running Moving Average Cross Factor Calculation Test")
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
    ma_cross_factor = MovingAverageCrossFactor()
    
    # Get default parameters
    default_params = ma_cross_factor.get_default_parameters()
    print(f"\nCalculating factor with default parameters: {default_params}")

    # Calculate the factor values
    factor_values = ma_cross_factor.calculate(price_data_dict, **default_params)

    # --- Output Results ---
    print("\n--- Calculated Factor Values (Last 5 days) ---")
    print(factor_values.tail())
    print("\n" + "=" * 70)
    print("Test finished successfully.")
    print("=" * 70)