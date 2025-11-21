#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CVILLIQ Strategy - Equity Curve Diagnostic Tool
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from strategies.cvilliq.strategy import CVILLIQStrategy
from shared.data_loader import load_local_data
from shared.backtest import BacktestEngine


def main():
    print("\n" + "="*80)
    print("  CVILLIQ Strategy - Diagnostic Tool")
    print("="*80)
    
    # Load data
    df = load_local_data(symbol='BTCUSDT', data_source='1h',
                         start_date='2022-11-01', end_date='2024-12-31', verbose=False)
    
    # Generate signals
    strategy = CVILLIQStrategy()
    signals = strategy.generate_signals(df, window=20, long_threshold=1.0,
                                       short_threshold=0.6, leverage=1)
    
    print("\nSignal Distribution:")
    print(f"  Long:  {(signals == 1).sum()}")
    print(f"  Short: {(signals == -1).sum()}")
    print(f"  Flat:  {(signals == 0).sum()}")
    
    # Run backtest
    engine = BacktestEngine(data=df, signals=signals, transaction_cost=0.0002,
                            slippage=0.0001, initial_capital=100000,
                            periods_per_year=8760, leverage=1, stop_loss_fee=0.00055)
    results = engine.run()
    equity = results['equity_curve']
    
    print(f"\nEquity Curve:")
    print(f"  Start:  ${equity.iloc[0]:,.2f}")
    print(f"  End:    ${equity.iloc[-1]:,.2f}")
    print(f"  Unique: {equity.nunique()}")
    
    # Check if flat
    if equity.nunique() <= 2:
        print(f"\nALERT: Equity curve is FLAT!")
    else:
        print(f"\nOK: Equity curve has variation")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
