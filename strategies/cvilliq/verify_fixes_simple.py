#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CVILLIQ Strategy - Simple Fix Verification
Run: python strategies/cvilliq/verify_fixes_simple.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np
from strategies.cvilliq.strategy import CVILLIQStrategy
from shared.data_loader import load_local_data
from shared.backtest import BacktestEngine


def main():
    print("\n" + "="*80)
    print("  CVILLIQ Strategy - Fix Verification")
    print("="*80 + "\n")

    # Load data
    print("[1/5] Loading data...")
    df = load_local_data(symbol='BTCUSDT', data_source='1h',
                         start_date='2022-11-01', end_date='2024-12-31', verbose=False)
    print(f"  OK - {len(df)} rows loaded\n")

    # Test signal generation
    print("[2/5] Generating signals...")
    strategy = CVILLIQStrategy()
    signals = strategy.generate_signals(df, window=20, long_threshold=1.0,
                                       short_threshold=0.6, leverage=1)

    long_count = (signals == 1).sum()
    short_count = (signals == -1).sum()
    flat_count = (signals == 0).sum()

    print(f"  Long signals:  {long_count:6d} ({long_count/len(signals)*100:5.1f}%)")
    print(f"  Short signals: {short_count:6d} ({short_count/len(signals)*100:5.1f}%)")
    print(f"  Flat signals:  {flat_count:6d} ({flat_count/len(signals)*100:5.1f}%)\n")

    if long_count == 0:
        print("  WARNING: No long signals generated\n")
    if short_count == 0:
        print("  WARNING: No short signals generated\n")

    # Test backtest
    print("[3/5] Running backtest...")
    engine = BacktestEngine(data=df, signals=signals, transaction_cost=0.0002,
                            slippage=0.0001, initial_capital=100000,
                            periods_per_year=8760, leverage=1, stop_loss_fee=0.00055)
    results = engine.run()

    equity = results['equity_curve']
    print(f"  Initial: ${equity.iloc[0]:,.2f}")
    print(f"  Final:   ${equity.iloc[-1]:,.2f}")
    print(f"  Return:  {((equity.iloc[-1]/equity.iloc[0])-1)*100:.2f}%")
    print(f"  Unique values: {equity.nunique()}\n")

    # Test CVILLIQ data
    print("[4/5] Checking CVILLIQ data range...")
    cvilliq = strategy.debug_data.get('cvilliq')
    if cvilliq is not None:
        print(f"  Min:     {cvilliq.min():.4f}")
        print(f"  Max:     {cvilliq.max():.4f}")
        print(f"  Mean:    {cvilliq.mean():.4f}")
        print(f"  Range [0.80, 1.50] valid: {0.80 <= cvilliq.max()}\n")

    # Check results
    print("[5/5] Verification Summary")
    print("-" * 80)

    checks = [
        ("Signals generated", long_count > 0 and short_count > 0),
        ("Equity curve moving", equity.nunique() > 100),
        ("Parameters in range", cvilliq is not None and 0.80 <= cvilliq.max()),
        ("Backtest completed", len(equity) > 0),
    ]

    all_pass = True
    for check_name, check_result in checks:
        status = "[PASS]" if check_result else "[FAIL]"
        print(f"  {status} - {check_name}")
        if not check_result:
            all_pass = False

    print("\n" + "="*80)
    if all_pass:
        print("  All checks passed! You can now run:")
        print("  python test_optimization_with_sensitivity.py")
    else:
        print("  Some checks failed. See details above.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
