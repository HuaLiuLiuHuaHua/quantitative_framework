# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a professional quantitative trading framework for developing, testing, and validating algorithmic trading strategies. The framework emphasizes rigorous backtesting, statistical validation, and strict avoidance of lookahead bias.

## Core Architecture

### Two-Engine Backtest System

The framework supports two complementary backtesting approaches:

1. **Vectorized Engine** (`BacktestEngine` in `shared/backtest.py`):
   - Fast, simple signal-based backtesting
   - Strategies implement `generate_signals()` method returning 1/0/-1 signals
   - Used for rapid parameter optimization (grid/random search)
   - Example: `quick_backtest(data, signals, transaction_cost, slippage)`

2. **Event-Driven Engine** (`EventDrivenBacktestEngine` in `shared/backtest.py`):
   - Bar-by-bar simulation with complex exit logic
   - Supports trailing stops, breakeven stops, dynamic position sizing
   - Strategies implement `add_indicators()`, `check_entry_signal()`, `check_exit_signal()`
   - More realistic but slower than vectorized

All strategies inherit from `BaseStrategy` (in `shared/base_strategy.py`) and can support both engines by implementing the appropriate methods.

### Strategy Development Pattern

Each strategy lives in `strategies/{strategy_name}/` with:
- `strategy.py`: Strategy logic inheriting from `BaseStrategy`
- `test_backtest.py`: Basic backtest with fixed parameters
- `test_optimization_random.py`: Randomized parameter search
- `test_mcpt.py`: Monte Carlo Permutation Test for statistical validation
- `test_walkforward_random.py`: Walk-forward analysis with random search
- `results/`: Auto-generated output directory (`.gitkeep` files track empty dirs)

### Critical Design Principle: Lookahead Bias Avoidance

**ALWAYS use `.shift(1)` on all technical indicators** to ensure signals use only historical data:

```python
# ❌ WRONG - uses current bar's indicator
ma = df['close'].rolling(20).mean()
signals[df['close'] > ma] = 1

# ✅ CORRECT - uses previous bar's indicator
ma = df['close'].rolling(20).mean().shift(1)
signals[df['close'] > ma] = 1
```

This is enforced throughout the codebase. Signals generated at bar close are executed at next bar's open.

### Shared Modules (`shared/`)

Core functionality used across all strategies:

- **backtest.py**: Both vectorized and event-driven backtest engines
- **base_strategy.py**: Abstract base class with `Position` dataclass
- **optimizer.py**: Parallel grid/random search supporting both engines
- **walkforward.py**: Rolling window out-of-sample testing
- **mcpt.py**: Monte Carlo Permutation Testing (bar permutation logic from MCPT-Main)
- **permutation.py**: Advanced/simple permutation functions for MCPT
- **metrics.py**: Performance metrics (Sharpe, profit factor, Calmar, max drawdown, etc.)
- **visualization.py**: Plotting functions for equity curves, optimization results, MCPT distributions
- **data_loader.py**: Load local data with fallback mechanisms
- **cross_sectional_backtest.py**: Cross-sectional portfolio backtesting
- **position_strategies.py**: Position sizing strategies (Strategy Pattern)
- **factor_operators.py**: Factor combination operators
- **ic_analysis.py**: Information coefficient analysis
- **sensitivity.py**: Parameter sensitivity analysis

### Position Strategy System (`shared/position_strategies.py`)

The framework uses the **Strategy Pattern** to decouple position sizing logic from backtesting:

**Available Strategies:**

1. **`TopBottomPositionStrategy`** (default)
   - Selects top N assets (highest factor values) for long positions
   - Selects bottom N assets (lowest factor values) for short positions
   - Equal weight for all positions
   - Used by most factors (momentum, volume, CVILLIQ)

2. **`ExcessReturnPositionStrategy`**
   - Market-neutral reversal strategy
   - Calculates excess returns relative to market average
   - Weights positions by |excess_return| / Σ|excess_returns|
   - All assets with non-zero excess returns participate
   - Used by momentum_reverse factor

3. **`PercentilePositionStrategy`**
   - Selects assets based on percentile thresholds (e.g., top 20%, bottom 20%)
   - Useful for dynamic universes with varying sizes
   - Equal weight within each group

**Usage Examples:**

```python
# Default behavior (TopBottom strategy)
from shared.cross_sectional_backtest import CrossSectionalBacktester

backtester = CrossSectionalBacktester(
    factor_calculator=my_factor,
    top_n=5,
    bottom_n=5,
    # position_strategy not specified - uses TopBottomPositionStrategy
)

# Custom strategy (ExcessReturn)
from shared.position_strategies import ExcessReturnPositionStrategy

position_strategy = ExcessReturnPositionStrategy()
backtester = CrossSectionalBacktester(
    factor_calculator=my_factor,
    position_strategy=position_strategy,
    # top_n/bottom_n not needed - strategy controls position sizing
)

# Percentile-based strategy
from shared.position_strategies import PercentilePositionStrategy

position_strategy = PercentilePositionStrategy()
backtester = CrossSectionalBacktester(
    factor_calculator=my_factor,
    position_strategy=position_strategy,
)

# Run backtest with strategy-specific parameters
results = backtester.run(
    factor_df=factor_df,
    lookback_period=20,  # Passed to ExcessReturnPositionStrategy
    long_percentile=20,  # Passed to PercentilePositionStrategy
    short_percentile=20
)
```

**Creating Custom Position Strategies:**

```python
from shared.position_strategies import BasePositionStrategy
import pandas as pd

class MyCustomStrategy(BasePositionStrategy):
    def __init__(self):
        super().__init__(strategy_name="MyCustomStrategy")

    def calculate_positions(self, factor_snapshot, all_data, decision_date, **kwargs):
        """
        Implement custom position logic.

        Returns:
            dict: {asset: (weight, direction)}
            - weight: float (0.0 to 1.0)
            - direction: 'long' or 'short'
        """
        positions = {}
        # Your custom logic here
        return positions

    def get_required_params(self):
        return ['param1', 'param2']  # Parameters your strategy needs
```

### Factor System (`factors/`)

For cross-sectional strategies operating on multiple assets:

- **factors/momentum/factor.py**: Example momentum factor
- Factors calculate time-series values across a universe of assets
- Returns DataFrame with dates as index, symbols as columns
- See `dynamic_universe_loader.py` for loading dynamic asset universes

### Data Management (`data_fetchers/`)

Scripts for fetching historical data from Bybit:
- Pattern: `bybit_{symbol}_1{d|h}_fetcher.py`
- Saves to `data/{SYMBOL}USDT_{timeframe}_{start}_{end}.csv`
- All fetchers inherit from common base class

## Common Commands

### Running Tests/Backtests

```bash
# Basic backtest with fixed parameters
cd strategies/bb_atr
python test_backtest.py

# Parameter optimization (random search, 200 iterations)
python test_optimization_random.py

# Statistical validation via MCPT
python test_mcpt.py

# Walk-forward analysis (out-of-sample testing)
python test_walkforward_random.py

# Walk-forward + MCPT on each window
python test_walkforward_random_mcpt.py
```

### Data Fetching

```python
from data_fetchers.bybit_btc_1h_fetcher import fetch_bybit_btc_1h_data

df = fetch_bybit_btc_1h_data(
    start_date="2022-01-01",
    end_date="2025-10-02",
    save=True,
    verbose=True
)
```

### Installing Dependencies

```bash
pip install -r requirements.txt
```

Dependencies: pandas, numpy, matplotlib, seaborn, ta (technical analysis), numba (performance)

## Strategy Testing Workflow

The recommended sequence for validating a strategy:

1. **Basic Backtest** (`test_backtest.py`): Verify strategy logic with fixed parameters
2. **Parameter Optimization** (`test_optimization_random.py`): Find optimal parameters via random search
3. **MCPT Validation** (`test_mcpt.py`): Check if results are statistically significant (p < 0.05)
4. **Walk-Forward Analysis** (`test_walkforward_random.py`): Test robustness with rolling windows
5. **Walk-Forward MCPT** (`test_walkforward_random_mcpt.py`): MCPT on each out-of-sample window

Only deploy strategies that pass all tests with:
- MCPT p-value < 0.05 (statistically significant)
- Walk-forward OOS/IS ratio > 0.7
- Walk-forward consistency > 60%

## Key Implementation Details

### Transaction Costs

All backtests account for:
- `transaction_cost`: Default 0.0006 (0.06%, includes maker/taker fees)
- `slippage`: Default 0.0001 (0.01%, execution slippage)

### Time Periods

Set `periods_per_year` based on timeframe:
- 1-hour data: 8760 (24 * 365)
- Daily data: 365

This affects annualized metrics like Sharpe ratio.

### Parallel Processing

Optimizers use `n_jobs=-1` by default (all CPU cores). Reduce if memory constrained:
```python
optimizer = ParameterOptimizer(..., n_jobs=4)  # Use 4 cores
```

### Random Search Parameters

Random search uses `n_iter` iterations (default 200) sampling from parameter grids:
```python
param_grid = {
    'bb_window': list(range(20, 201, 5)),  # Grid of possible values
    'bb_std': list(np.arange(1.5, 3.1, 0.1))
}
```

Optimizer randomly samples `n_iter` combinations from the Cartesian product.

### Result Storage

All test scripts automatically save results to `strategies/{name}/results/`:
- Backtest: equity curves, trade logs, performance metrics (JSON/CSV/PNG)
- Optimization: parameter grids, best params, sensitivity charts
- MCPT: permutation distributions, p-values, statistical summaries
- Walk-forward: rolling window results, OOS metrics, consistency analysis

### Plotting

All plots use `matplotlib.use('Agg')` to disable auto-display and save to disk only.

## Code Style Conventions

- Follow PEP 8 style guide
- Use `numba` decorators for computationally intensive loops
- Module-level strategy wrapper functions required for multiprocessing:
  ```python
  def strategy_func(data, param1, param2):
      strategy = MyStrategy()
      return strategy.generate_signals(data, param1=param1, param2=param2)
  ```
- Document purposes at file headers (Chinese docstrings are standard in this codebase)
- Use `load_local_data()` from `shared/data_loader.py` to load data, not manual CSV reads

## Testing Against Overfitting

The framework incorporates statistical rigor via:

1. **Bar Permutation (MCPT)**: Randomly shuffles trade entry signals while keeping bars in order. If original strategy significantly outperforms 1000 permutations (p < 0.05), it's not just curve-fitting.

2. **Walk-Forward Analysis**: Optimizes on training window, tests on unseen future window, rolls forward. Prevents using future data for parameter selection.

3. **Combined MCPT + Walk-Forward**: Runs MCPT on each walk-forward window to ensure each OOS period is statistically valid.

## Google Colab Support

`Google_Colab_Runner.ipynb` enables running the framework on Colab with GPU/TPU acceleration for faster optimization.

## Important Notes

- Git tracks empty `results/` directories with `.gitkeep` files
- Data files are .gitignored (see `.gitignore`)
- All strategies use event-driven engine with sophisticated exit logic (trailing stops, breakeven, time-based exits)
- Factor-based strategies in `factors/` operate on cross-sectional data across multiple assets
- `dynamic_universe_loader.py` demonstrates loading rebalancing universes from CSV
