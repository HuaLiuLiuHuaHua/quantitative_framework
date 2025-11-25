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

class VolatilityReversionFactor:
    """
    Volatility Reversion Factor
    
    This factor identifies potential mean-reversion opportunities based on volatility extremes.
    
    Logic:
    1. Calculate the standard deviation of daily close-to-close returns over x days
    2. For LONG signals:
       - Calculate trigger price: trigger = previous_close * (1 - y * std)
       - Filter A: Yesterday's LOW must be below the trigger price (detected dip)
       - Filter B: Yesterday's close must be above z-day moving average (uptrend context)
       - Rank by how much yesterday's low dipped below trigger price
    
    3. For SHORT signals (inverted logic):
       - Calculate trigger price: trigger = previous_close * (1 + y * std)
       - Filter A: Yesterday's HIGH must be above the trigger price (detected spike)
       - Filter B: Yesterday's close must be BELOW z-day moving average (downtrend context)
       - Rank by how much yesterday's high spiked above trigger price
    
    Weight allocation: Equal weight among selected assets
    If insufficient assets for target count, proportional allocation is used
    
    Parameters:
    - x: lookback period for volatility calculation (days)
    - y: volatility multiplier for trigger threshold
    - z: moving average period for filter B
    - long_pct: percentage of portfolio allocated to long positions (0-1)
    - short_pct: percentage of portfolio allocated to short positions (0-1)
    - leverage: leverage multiplier (>= 1)
    - rebalance_days: days between portfolio rebalances
    - holding_days: days to hold positions
    """

    def __init__(self, factor_name: str = "VolatilityReversion"):
        """
        Initializes the VolatilityReversionFactor.

        Args:
            factor_name (str): The name of the factor.
        """
        self.factor_name = factor_name
        self.parameters = {}

    def calculate(
        self,
        price_data_dict: Dict[str, pd.DataFrame],
        x: int = 20,  # volatility lookback period
        y: float = 2.0,  # volatility multiplier
        z: int = 50,  # moving average period
        long_pct: float = 0.5,  # long allocation percentage
        short_pct: float = 0.5,  # short allocation percentage
        min_volume: float = 0
    ) -> pd.DataFrame:
        """
        Calculate volatility reversion factor values.
        
        Returns factor values where:
        - Positive values indicate long signals (strength proportional to depth below trigger)
        - Negative values indicate short signals (strength proportional to depth above inverse trigger)
        - NaN indicates position doesn't meet selection criteria

        Args:
            price_data_dict (dict): Dictionary of price dataframes {symbol: DataFrame}.
                                    Each DataFrame must contain 'close', 'low', 'high' columns.
            x (int): Lookback period for volatility calculation
            y (float): Volatility multiplier for trigger threshold
            z (int): Moving average period for filter B
            long_pct (float): Percentage of portfolio for long positions
            short_pct (float): Percentage of portfolio for short positions
            min_volume (float): Minimum volume threshold (not used in this version)

        Returns:
            pd.DataFrame: Factor values time series (index=dates, columns=symbols).
                         Values represent signal strength for selection and ranking.
        """
        # 輸入驗證
        if not price_data_dict:
            logger.warning("VolatilityReversionFactor: price_data_dict 為空")
            return pd.DataFrame()

        # 過濾有效數據
        min_required_length = max(x, z) + 10
        valid_data = {k: v for k, v in price_data_dict.items()
                      if v is not None and not v.empty and len(v) >= min_required_length}

        invalid_count = len(price_data_dict) - len(valid_data)
        if invalid_count > 0:
            logger.info(f"VolatilityReversionFactor: 跳過 {invalid_count} 個數據不足的標的 (需要至少 {min_required_length} 行)")

        if not valid_data:
            logger.warning(f"VolatilityReversionFactor: 沒有符合條件的標的數據！")
            return pd.DataFrame()

        logger.info(f"VolatilityReversionFactor: 計算 {len(valid_data)} 個標的的因子值 (x={x}, y={y}, z={z})")

        all_dates = pd.DatetimeIndex([])
        for df in valid_data.values():
            if df is not None and not df.empty:
                all_dates = all_dates.union(df.index)

        if len(all_dates) == 0:
            return pd.DataFrame()

        all_dates = all_dates.sort_values().unique()
        factor_df = pd.DataFrame(index=all_dates, columns=list(price_data_dict.keys()), dtype=float)

        # 計算診斷統計
        total_signals_generated = 0
        signals_with_no_values = 0

        for symbol, df in valid_data.items():
            if df is None or df.empty:
                continue

            try:
                # 驗證必需的列
                required_columns = ['close', 'low', 'high']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    logger.error(f"{symbol}: 缺少必需列: {missing_columns}")
                    continue

                close = df['close']
                low = df['low']
                high = df['high']

                # 驗證數據有效性
                if close.isna().all() or low.isna().all() or high.isna().all():
                    logger.warning(f"{symbol}: 必需列全為 NaN")
                    continue

                # --- LONG SIGNAL CALCULATION ---
                # 1. Calculate daily returns and their standard deviation
                daily_returns = close.pct_change(1)
                volatility = op.ts_std_dev(daily_returns, x)

                # 2. Calculate trigger price for long: 
                #    Based on previous close, adjusted down by volatility
                prev_close = close.shift(1)
                trigger_price_long = prev_close * (1 - y * volatility)

                # 3. Filter A: yesterday's low < trigger_price
                #    This identifies when price dipped below volatility-adjusted level
                prev_low = low.shift(1)
                filter_a_long = prev_low < trigger_price_long

                # 4. Filter B: yesterday's close > z-day moving average
                #    Ensure price is above longer-term average (uptrend context)
                ma_z = op.ts_mean(close, z)
                filter_b = prev_close > ma_z

                # 5. Combined filters for long signal
                long_signal_mask = filter_a_long & filter_b

                # Rank by how much the low dipped below trigger price
                # Deeper dip = stronger signal (positive value)
                depth_below_trigger = trigger_price_long - prev_low
                depth_below_trigger = depth_below_trigger / (prev_close + 1e-9)  # normalize by price
                
                # Long signal: positive where conditions met, strength = normalized depth
                long_signal = depth_below_trigger.copy()
                long_signal[~long_signal_mask] = np.nan

                # --- SHORT SIGNAL CALCULATION (INVERTED LOGIC) ---
                # 1. Calculate inverse trigger for short: 
                #    Based on previous close, adjusted up by volatility
                trigger_price_short = prev_close * (1 + y * volatility)

                # 2. Filter A (inverted): yesterday's high > trigger_price_short
                #    This identifies when price spiked above volatility-adjusted level
                prev_high = high.shift(1)
                filter_a_short = prev_high > trigger_price_short

                # 3. Filter B (inverted for short): yesterday's close < z-day moving average
                #    For short, we want price below longer-term average (downtrend context)
                filter_b_short = prev_close < ma_z

                # 4. Combined filters for short signal
                short_signal_mask = filter_a_short & filter_b_short

                # Rank by how much the high spiked above trigger price
                # Higher spike = stronger signal (negative value for short)
                depth_above_trigger = prev_high - trigger_price_short
                depth_above_trigger = depth_above_trigger / (prev_close + 1e-9)  # normalize by price

                # Short signal: negative where conditions met
                short_signal = -depth_above_trigger.copy()
                short_signal[~short_signal_mask] = np.nan

                # --- COMBINE LONG AND SHORT SIGNALS ---
                # Shift by 1 to avoid lookahead bias
                # Use long signal where available, otherwise use short signal
                combined_signal = long_signal.copy()
                mask_no_long = combined_signal.isna()
                combined_signal[mask_no_long] = short_signal[mask_no_long]

                # 診斷統計
                combined_count = combined_signal.notna().sum()
                total_signals_generated += combined_count
                if combined_count == 0:
                    signals_with_no_values += 1
                    logger.debug(f"{symbol}: 無有效信號 (long_triggers={long_signal_mask.sum()}, short_triggers={short_signal_mask.sum()})")

                factor_df[symbol] = combined_signal.shift(1).reindex(all_dates)

            except Exception as e:
                logger.error(f"Error calculating volatility reversion for {symbol}: {e}", exc_info=True)
                continue

        # 輸出診斷信息
        factor_df = factor_df.dropna(how='all')
        logger.info(f"VolatilityReversionFactor 計算完成: {factor_df.shape}")
        logger.info(f"  總有效信號: {total_signals_generated}")
        logger.info(f"  無有效信號的標的: {signals_with_no_values}/{len(valid_data)}")

        if factor_df.size > 0:
            total_cells = factor_df.size
            non_nan_count = factor_df.notna().sum().sum()
            nan_ratio = 1 - (non_nan_count / total_cells)
            logger.info(f"  非 NaN 比例: {(1-nan_ratio)*100:.1f}% ({non_nan_count}/{total_cells})")

            if nan_ratio > 0.95:
                logger.warning(f"警告: 因子值 95% 以上為 NaN，可能導致無交易")

        return factor_df

    def get_parameter_grid(self) -> Dict[str, Tuple]:
        """
        Returns the parameter grid for optimization.
        
        Returns:
            Dict: Parameter ranges as (start, end, step) tuples
        """
        return {
            'x': (10, 50, 5),  # volatility lookback: 10 to 50 days, step 5
            'y': (1.0, 3.0, 0.5),  # volatility multiplier: 1.0 to 3.0, step 0.5
            'z': (20, 100, 10),  # MA period: 20 to 100 days, step 10
            'long_pct': (0.3, 0.7, 0.1),  # long allocation: 30% to 70%
            'short_pct': (0.0, 0.7, 0.1),  # short allocation: 0% to 70%
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Returns the default parameters for the factor.
        """
        return {
            'x': 20,  # 20-day volatility lookback
            'y': 2.0,  # 2x volatility multiplier
            'z': 50,  # 50-day moving average
            'long_pct': 0.5,  # 50% long allocation
            'short_pct': 0.5,  # 50% short allocation
        }

    def __call__(self, data: Dict[str, pd.DataFrame], **params) -> pd.DataFrame:
        """Makes the factor object callable."""
        return self.calculate(data, **params)


# --- Main execution block for testing ---
if __name__ == "__main__":
    print("=" * 70)
    print("Running Volatility Reversion Factor Calculation Test")
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
    vol_reversion_factor = VolatilityReversionFactor()
    
    # Get default parameters
    default_params = vol_reversion_factor.get_default_parameters()
    print(f"\nCalculating factor with default parameters: {default_params}")

    # Calculate the factor values
    factor_values = vol_reversion_factor.calculate(price_data_dict, **default_params)

    # --- Output Results ---
    print("\n--- Calculated Factor Values (Last 10 days) ---")
    print(factor_values.tail(10))
    
    print("\n--- Factor Statistics ---")
    print(f"Mean: {factor_values.mean().mean():.6f}")
    print(f"Std: {factor_values.std().mean():.6f}")
    print(f"Non-NaN Count per symbol:")
    print(factor_values.notna().sum())
    
    print("\n" + "=" * 70)
    print("Test finished successfully.")
    print("=" * 70)
