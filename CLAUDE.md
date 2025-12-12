# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **professional quantitative trading research framework** for cryptocurrency strategy development. It emphasizes avoiding lookahead bias, rigorous backtesting, robust parameter optimization, and statistical validation.

### Core Philosophy
- **No lookahead bias**: All indicators use `shift(1)` to ensure only historical data is used
- **Rigorous validation**: Walk-Forward, Monte Carlo Permutation Testing (MCPT), and sensitivity analysis
- **Modular design**: Pluggable strategies, flexible backtesting engines, extensible factor system
- **Production-ready**: Transaction costs, slippage modeling, comprehensive performance metrics

---

## Architecture Overview

### Directory Structure

```
quantitative_framework/
├── data/                          # Historical OHLCV data (CSV)
├── data_fetchers/                 # Bybit API data collection
├── shared/                        # Core framework modules
├── strategies/                    # Strategy implementations
├── factors/                       # Cross-sectional factors
└── ml_tools/                      # Machine learning utilities
```

### Core Modules in `shared/`

| Module | Purpose |
|--------|---------|
| `backtest.py` | Two backtesting engines: vectorized (fast) and event-driven (complex logic) |
| `base_strategy.py` | Abstract base class for all strategies |
| `optimizer.py` | Grid/random search parameter optimization with parallel processing |
| `metrics.py` | 13 comprehensive performance metrics (Sharpe, Calmar, Profit Factor, etc.) |
| `mcpt.py` | Monte Carlo Permutation Testing for statistical significance |
| `walkforward.py` | Walk-Forward analyzer for out-of-sample validation |
| `sensitivity.py` | Parameter sensitivity analysis and visualization |
| `data_loader.py` | Local data loading with lookback window handling |
| `base_factor.py` | Abstract base for cross-sectional factors |
| `visualization.py` | Equity curves, drawdown analysis, heatmaps, monthly returns |

### Strategy Architecture

All strategies inherit from `BaseStrategy` and must implement:
- `get_lookback()`: Returns required lookback period for indicators
- Either `generate_signals()` (for vectorized engine) or `check_entry_signal()` + `check_exit_signal()` (for event-driven)

**Strategy Types**:
- **Traditional**: `ma_cross/`, `momentum/` - Simple technical indicators
- **ML-based**: `lstm_feature/`, `tree_combined/` - LSTM/XGBoost prediction models
- **Cross-sectional**: `factors/` directory - Asset universe ranking factors

---

## Common Development Tasks

### Running a Strategy Backtest
```bash
cd strategies/[strategy_name]
python test_backtest.py
```

### Parameter Optimization
```bash
cd strategies/[strategy_name]
python test_optimization_with_sensitivity.py
```

### Walk-Forward Out-of-Sample Testing
```bash
cd strategies/[strategy_name]
python test_walkforward_with_sensitivity.py
```

### Fetching Fresh Data
```bash
# Example: Bybit BTC 1-hour data
from data_fetchers.bybit_btc_1h_fetcher import fetch_bybit_btc_1h_data
df = fetch_bybit_btc_1h_data(start_date="2024-01-01", end_date="2024-12-31", save=True)
```

### Creating a New Strategy

1. **Create directory**: `mkdir strategies/your_strategy`
2. **Implement strategy.py** inheriting from `BaseStrategy`
3. **Create test files**: `test_backtest.py`, `test_optimization_with_sensitivity.py`, `test_walkforward_with_sensitivity.py`
4. Reference existing strategies (`ma_cross/`, `lstm_feature/`) for patterns

### Creating a New Data Source

Copy and modify an existing fetcher (e.g., `bybit_btc_1h_fetcher.py`):
- Change `symbol` (e.g., "ETHUSDT")
- Update function name
- Modify output filename

---

## Key Design Patterns

### Lookahead Bias Prevention

**Critical**: All indicators must use `shift(1)`:
```python
# ❌ Wrong - uses current bar's data
ma = df['close'].rolling(20).mean()
signals = df['close'] > ma  # Lookahead!

# ✅ Correct - uses previous bar's data
ma = df['close'].rolling(20).mean().shift(1)
signals = df['close'] > ma  # Only historical data
```

This ensures signals generated at bar close are executed next bar open (no future peeking).

### Two Backtesting Engines

**Vectorized Engine** (`BacktestEngine`):
- Fast, all-at-once calculation
- Suitable for simple entry/exit logic
- Use `quick_backtest()` for rapid prototyping

**Event-Driven Engine** (`EventDrivenBacktestEngine`):
- Bar-by-bar simulation
- Supports complex logic: trailing stops, position sizing, dynamic exits
- Slower but more realistic

### Parameter Optimization

```python
from shared.optimizer import ParameterOptimizer

param_grid = {'window': [10, 20, 30], 'threshold': [0.01, 0.02]}
optimizer = ParameterOptimizer(
    strategy_func=strategy.generate_signals,
    data=df,
    param_grid=param_grid,
    objective='sharpe_ratio',  # or 'profit_factor', 'calmar_ratio', etc.
    n_jobs=4  # Parallel processing cores
)
best_params = optimizer.optimize()
```

### Cross-Sectional Factors

Factors rank assets in a universe. Each factor implements:
```python
class MyFactor(BaseFactor):
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Return DataFrame with index=timestamps, columns=symbols, values=factor scores
        return factor_scores
```

Used in ensemble strategies to select top/bottom assets.

---

## Performance Metrics

The framework calculates 13 metrics:

**Portfolio Level**:
- Total Return, Annualized Return, Volatility, Sharpe Ratio
- Maximum Drawdown, Calmar Ratio
- Win Rate, Profit Factor
- Total Trades, Winning Trades, Losing Trades, Avg Win/Loss

**Transaction Costs**: Fees (bps) + Slippage (bps) modeled per trade

---

## Testing and Validation Workflow

1. **test_backtest.py**: Validate strategy logic with fixed parameters
2. **test_optimization_with_sensitivity.py**: Find optimal parameters, analyze sensitivity
3. **test_walkforward_with_sensitivity.py**: Realistic out-of-sample testing
4. **test_mcpt.py** (optional): Statistical significance via permutation testing

Results are saved in `results/` with:
- Performance metrics (JSON)
- Equity curve plots
- Parameter sensitivity heatmaps
- Monthly returns tables

---

## Important Notes

### Data Handling
- Data stored as CSV in `data/` directory
- Supports lookback extension for indicator warmup
- `data_loader.py` handles date range and lookback automatically

### Dependencies
- **Core**: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn
- **ML**: tensorflow/keras (LSTM), xgboost
- **Utilities**: joblib (parallel), tqdm (progress), requests (API)

### Code Review and Debugging

Per project instructions:
- Use code reviewer for all code modifications
- Use debugger for debugging issues

### Optimization Objectives

Choose based on strategy goals:
- **Sharpe Ratio**: Stable, risk-adjusted returns
- **Profit Factor**: High absolute profitability
- **Calmar Ratio**: Risk-efficient growth
- **Max Return**: Maximum absolute gain
- **Max Drawdown**: Minimize peak-to-trough loss

---

## Recent Architecture Decisions

- **Lookahead Bias**: Enforced via `shift(1)` on all indicators
- **Dual Engine Design**: Vectorized for speed, event-driven for complexity
- **Parallel Optimization**: ProcessPoolExecutor for multi-core parameter search
- **Walk-Forward First**: Out-of-sample testing over in-sample optimization
- **ML Integration**: LSTM/XGBoost as factors within strategy framework
- **Cross-Sectional Support**: Unified factor system for asset ranking

---

## File Reading/Modification Patterns

When modifying code:
1. Always use code reviewer for code changes (per CLAUDE.md instructions)
2. Read files first to understand context before editing
3. Avoid over-engineering: only change what's necessary
4. Use existing patterns from similar strategies/factors as templates
5. Test modified code by running the appropriate test file

---

## Useful References

- **BaseStrategy**: `shared/base_strategy.py` - Interface all strategies implement
- **Example Strategies**: `strategies/ma_cross/`, `strategies/lstm_feature/`
- **Metrics**: `shared/metrics.py` - All available performance metrics
- **Validation**: `shared/mcpt.py`, `shared/walkforward.py` - Statistical testing
