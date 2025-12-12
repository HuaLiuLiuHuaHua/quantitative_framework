"""
Walk-forward analysis for Volatility Reversion Factor
Tests the factor across different time periods
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parents[2]))

from shared.data_loader import load_local_data
from factor import VolatilityReversionFactor

def walk_forward_analysis(
    symbols,
    start_date,
    end_date,
    window_days=90,
    step_days=30,
    params=None
):
    """
    Perform walk-forward analysis on factor
    
    Args:
        symbols: List of symbols to analyze
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        window_days: Size of analysis window in days
        step_days: Step size for rolling window in days
        params: Factor parameters dict, or None for defaults
    
    Returns:
        DataFrame with walk-forward results
    """
    factor = VolatilityReversionFactor()
    
    if params is None:
        params = factor.get_default_parameters()
    
    # Load all data first
    all_data = {}
    for symbol in symbols:
        df = load_local_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            data_source='1d'
        )
        if df is not None and not df.empty:
            all_data[symbol] = df
    
    if not all_data:
        print("ERROR: No data loaded")
        return None
    
    # Determine date range
    all_dates = pd.DatetimeIndex([])
    for df in all_data.values():
        all_dates = all_dates.union(df.index)
    
    all_dates = sorted(all_dates)
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    results = []
    
    # Walk-forward windows
    current_date = start_dt
    while current_date + timedelta(days=window_days) <= end_dt:
        window_end = current_date + timedelta(days=window_days)
        
        # Get data for this window
        window_data = {}
        for symbol, df in all_data.items():
            mask = (df.index >= current_date) & (df.index <= window_end)
            window_df = df[mask]
            if not window_df.empty:
                window_data[symbol] = window_df
        
        if not window_data:
            current_date += timedelta(days=step_days)
            continue
        
        # Calculate factor for this window
        try:
            factor_values = factor.calculate(window_data, **params)
            
            # Calculate statistics
            signal_count = factor_values.notna().sum().sum()
            signal_total = factor_values.shape[0] * factor_values.shape[1]
            signal_pct = (signal_count / signal_total) * 100 if signal_total > 0 else 0
            
            long_count = (factor_values > 0).sum().sum()
            short_count = (factor_values < 0).sum().sum()
            
            mean_signal = factor_values.mean().mean()
            std_signal = factor_values.std().mean()
            max_signal = factor_values.max().max()
            min_signal = factor_values.min().min()
            
            results.append({
                'window_start': current_date.strftime('%Y-%m-%d'),
                'window_end': window_end.strftime('%Y-%m-%d'),
                'signal_count': signal_count,
                'signal_pct': signal_pct,
                'long_count': long_count,
                'short_count': short_count,
                'mean': mean_signal,
                'std': std_signal,
                'max': max_signal,
                'min': min_signal,
                'assets_count': len(window_data)
            })
        except Exception as e:
            print(f"Error processing window {current_date}: {e}")
        
        current_date += timedelta(days=step_days)
    
    return pd.DataFrame(results)

def main():
    print("=" * 80)
    print("WALK-FORWARD ANALYSIS: Volatility Reversion Factor")
    print("=" * 80)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LTCUSDT']
    start_date = '2024-01-01'
    end_date = '2024-08-31'
    
    # Test with default parameters
    print(f"\nAnalyzing {symbols} from {start_date} to {end_date}")
    print(f"Window size: 90 days, Step: 30 days\n")
    
    results_df = walk_forward_analysis(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        window_days=90,
        step_days=30,
        params=None  # Use defaults
    )
    
    if results_df is not None and not results_df.empty:
        print("\nWalk-Forward Results:")
        print(results_df.to_string(index=False))
        
        print("\n" + "-" * 80)
        print("Summary Statistics:")
        print(f"Average signals per window:    {results_df['signal_count'].mean():.1f}")
        print(f"Average signal percentage:     {results_df['signal_pct'].mean():.1f}%")
        print(f"Average long signals:          {results_df['long_count'].mean():.1f}")
        print(f"Average short signals:         {results_df['short_count'].mean():.1f}")
        print(f"Average signal mean:           {results_df['mean'].mean():.6f}")
        print(f"Average signal std:            {results_df['std'].mean():.6f}")
        
        # Test parameter variations
        print("\n" + "=" * 80)
        print("PARAMETER VARIATION ANALYSIS")
        print("=" * 80)
        
        param_variations = [
            {'x': 10, 'y': 1.5, 'z': 30},
            {'x': 20, 'y': 2.0, 'z': 50},
            {'x': 30, 'y': 2.5, 'z': 70},
        ]
        
        for params in param_variations:
            print(f"\nParameters: x={params['x']}, y={params['y']}, z={params['z']}")
            var_results = walk_forward_analysis(
                symbols=symbols,
                start_date=start_date,
                end_date='2024-06-30',  # Shorter test period
                window_days=90,
                step_days=30,
                params=params
            )
            
            if var_results is not None and not var_results.empty:
                print(f"  Avg signals: {var_results['signal_count'].mean():.1f}")
                print(f"  Avg signal %: {var_results['signal_pct'].mean():.1f}%")
                print(f"  Long/Short ratio: {var_results['long_count'].sum()}/{var_results['short_count'].sum()}")
        
        print("\n✓ Walk-forward analysis completed successfully")
    else:
        print("ERROR: No results generated")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "=" * 80)
        print("WALK-FORWARD TEST PASSED ✓")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("WALK-FORWARD TEST FAILED ✗")
        print("=" * 80)
        sys.exit(1)
