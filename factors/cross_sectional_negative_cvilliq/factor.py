import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple

# Add project root to Python path to allow module imports
sys.path.append(str(Path(__file__).parents[2]))

from shared.data_loader import load_local_data
from factors import operators as op

# Set up logging
logger = logging.getLogger(__name__)

class NegativeCVILLIQFactor:
    """
    Negative Coefficient of Variation of Illiquidity (CVILLIQ) Factor.

    This is the inverted version of the CVILLIQ factor.
    A high factor value corresponds to low relative volatility of illiquidity (potential long signal).
    A low factor value corresponds to high relative volatility of illiquidity (potential short signal).

    CVILLIQ = std(ILLIQ) / mean(ILLIQ)
    where ILLIQ = |daily_return| / daily_volume
    This factor returns -CVILLIQ.
    """

    def __init__(self, factor_name: str = "NegativeCVILLIQ"):
        """
        Initializes the NegativeCVILLIQFactor.

        Args:
            factor_name (str): The name of the factor.
        """
        self.factor_name = factor_name
        self.parameters = {}

    def calculate(
        self,
        price_data_dict: Dict[str, pd.DataFrame],
        window: int = 20,
        min_volume: float = 0
    ) -> pd.DataFrame:
        """
        Calculate Negative CVILLIQ factor values.

        Args:
            price_data_dict (dict): Dictionary of price dataframes {symbol: DataFrame}.
                                    Each DataFrame must contain 'close' and 'volume' columns.
            window (int): The rolling window for calculating mean and std dev of illiquidity.
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
            if df is None or df.empty or 'close' not in df or 'volume' not in df:
                continue

            try:
                close = df['close']
                volume = df['volume']

                # Calculate daily returns: r_{i,t}
                returns = close.pct_change(1)

                # Calculate daily illiquidity: ILLIQ_{i,t} = |r| / Volume
                # Add a small epsilon to volume to avoid division by zero
                illiq_daily = op.abs_val(returns) / (volume + 1e-9)

                # Calculate rolling standard deviation of ILLIQ: σ(ILLIQ)
                illiq_std = op.ts_std_dev(illiq_daily, window)

                # Calculate rolling mean of ILLIQ: mean(ILLIQ)
                illiq_mean = op.ts_mean(illiq_daily, window)

                # Calculate CVILLIQ = σ(ILLIQ) / mean(ILLIQ)
                # Add a small epsilon to mean to avoid division by zero
                cvilliq = illiq_std / (illiq_mean + 1e-9)

                # Invert the factor value and shift by 1 to avoid lookahead bias
                factor_df[symbol] = -cvilliq.shift(1).reindex(all_dates)

            except Exception as e:
                logger.error(f"Error calculating Negative CVILLIQ for {symbol}: {e}", exc_info=True)
                continue

        factor_df = factor_df.dropna(how='all')
        return factor_df

    def get_parameter_grid(self) -> Dict[str, Tuple]:
        """
        Returns the parameter grid for optimization.
        The 'window' is the lookback period for calculating the statistics of illiquidity.
        """
        return {
            'window': (10, 60, 5),  # (start, end, step)
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Returns the default parameters for the factor.
        """
        return {
            'window': 20,
        }

    def __call__(self, data: Dict[str, pd.DataFrame], **params) -> pd.DataFrame:
        """Makes the factor object callable."""
        return self.calculate(data, **params)

# --- Main execution block for testing ---
if __name__ == "__main__":
    print("=" * 70)
    print("Running Negative CVILLIQ Factor Calculation Test")
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
    cvilliq_factor = NegativeCVILLIQFactor()
    
    # Get default parameters
    default_params = cvilliq_factor.get_default_parameters()
    print(f"\nCalculating factor with default parameters: {default_params}")

    # Calculate the factor values
    factor_values = cvilliq_factor.calculate(price_data_dict, **default_params)

    # --- Output Results ---
    print("\n--- Calculated Factor Values (Last 5 days) ---")
    print(factor_values.tail())
    print("\n" + "=" * 70)
    print("Test finished successfully.")
    print("=" * 70)