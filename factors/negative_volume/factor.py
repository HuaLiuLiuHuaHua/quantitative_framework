import pandas as pd
from shared.base_factor import BaseFactor

class NegativeVolumeFactor(BaseFactor):
    """
    A factor that is inversely related to trading volume.
    It calculates the rolling average volume.
    The strategy will long assets with low volume and short assets with high volume.
    """
    def calculate(self, price_data: dict, window: int, min_volume: int = 0) -> pd.DataFrame:
        """
        Calculates the rolling average volume and returns its negative.

        Args:
            price_data (dict): A dictionary where keys are symbols and values are pandas DataFrames with price data.
            window (int): The rolling window period for calculating the average volume.
            min_volume (int): A minimum volume filter (not used in this specific logic but kept for compatibility).

        Returns:
            pd.DataFrame: A DataFrame with timestamps as index, symbols as columns, and the negative average volume as values.
        """
        all_factor_values = {}
        for symbol, df in price_data.items():
            if df is not None and not df.empty and 'volume' in df.columns:
                # Calculate rolling average volume
                avg_volume = df['volume'].rolling(window=window, min_periods=1).mean()
                
                # The factor is the negative of the average volume, shifted to prevent lookahead bias.
                # This way, lower volume gets a higher score, leading to a long position.
                factor_value = -avg_volume.shift(1)
                
                all_factor_values[symbol] = factor_value
        
        factor_df = pd.DataFrame(all_factor_values)
        return factor_df
