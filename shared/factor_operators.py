
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import rankdata

"""
This module provides a collection of factor calculation operators,
adapted from the phandas library to work directly with pandas DataFrames
in a 'wide' format (index=timestamps, columns=symbols).
"""

def neutralize_by_regression(factor_df: pd.DataFrame, risk_factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Neutralizes a factor DataFrame against a risk factor DataFrame using OLS regression.

    This function operates cross-sectionally for each timestamp. For each date, it
    regresses the factor values against the risk factor values and returns the residuals.

    Args:
        factor_df (pd.DataFrame): A 'wide' DataFrame of factor values (index=dates, columns=symbols).
        risk_factor_df (pd.DataFrame): A 'wide' DataFrame of risk factor values to neutralize against.

    Returns:
        pd.DataFrame: A DataFrame of the neutralized factor values (residuals).
    """
    # Align dataframes to ensure they have the same shape and common values
    factor_aligned, risk_aligned = factor_df.align(risk_factor_df, join='inner', axis=1)
    factor_aligned, risk_aligned = factor_aligned.align(risk_aligned, join='inner', axis=0)

    residuals_df = pd.DataFrame(index=factor_aligned.index, columns=factor_aligned.columns, dtype=float)

    # Iterate over each timestamp (row) to perform cross-sectional regression
    for i in range(len(factor_aligned)):
        # Get the cross-sectional slice for the current date
        factor_slice = factor_aligned.iloc[i].dropna()
        risk_slice = risk_aligned.iloc[i].dropna()

        if factor_slice.empty or risk_slice.empty:
            continue

        # Align assets for this specific timestamp
        common_assets = factor_slice.index.intersection(risk_slice.index)
        if len(common_assets) < 2:  # Need at least 2 data points for a meaningful regression
            continue
            
        y = factor_slice.loc[common_assets]
        X = risk_slice.loc[common_assets]

        # Add a constant (intercept) to the independent variable
        X_with_const = sm.add_constant(X)

        # Fit the OLS model
        model = sm.OLS(y, X_with_const).fit()

        # Store the residuals in the corresponding location of the output DataFrame
        residuals_df.iloc[i].loc[common_assets] = model.resid
        
    return residuals_df

def ts_delay(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Calculates the time-series delay (lag) for a wide-format DataFrame.

    Args:
        df (pd.DataFrame): A 'wide' DataFrame (index=dates, columns=symbols).
        window (int): The number of periods to lag.

    Returns:
        pd.DataFrame: The lagged DataFrame.
    """
    if window < 0:
        raise ValueError("Window must be non-negative")
    return df.shift(window)

def ts_rank(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Calculates the rolling time-series rank for a wide-format DataFrame.
    The rank is returned as a percentage (0.0 to 1.0).

    Args:
        df (pd.DataFrame): A 'wide' DataFrame (index=dates, columns=symbols).
        window (int): The rolling window size.

    Returns:
        pd.DataFrame: The DataFrame of rolling percentile ranks.
    """
    if window <= 0:
        raise ValueError("Window must be positive")
    return df.rolling(window, min_periods=window).rank(pct=True)
