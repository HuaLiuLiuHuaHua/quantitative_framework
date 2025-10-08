# Gemini Code Assistant Context: quantitative_framework

## Project Overview

This project is a professional quantitative trading framework written in Python. It is designed for the development, backtesting, optimization, and robust analysis of algorithmic trading strategies. The framework places a strong emphasis on rigorous testing and the avoidance of common pitfalls like lookahead bias.

The architecture is modular, consisting of three main parts:
1.  **Data Fetchers (`data_fetchers/`)**: A collection of scripts responsible for downloading historical market data (e.g., from the Bybit exchange) and saving it locally in CSV format.
2.  **Shared Library (`shared/`)**: The core engine of the framework. It provides reusable components for:
    *   **Data Loading (`data_loader.py`)**: A unified interface to load local data, with a fallback to fetch new data if it's missing.
    *   **Backtesting (`backtest.py`)**: A sophisticated engine that simulates strategy performance, accounting for transaction costs and slippage, and calculates a comprehensive set of performance metrics.
    *   **Optimization (`optimizer.py`)**: A parallelized grid-search optimizer to find the best parameters for a strategy based on a specified objective function (e.g., Sharpe Ratio, Profit Factor).
    *   **Walk-Forward Analysis (`walkforward.py`)**: A tool for robust out-of-sample testing that simulates a real-world scenario of periodically re-optimizing a strategy.
    *   **Metrics & Visualization**: Modules for calculating performance metrics and plotting results.
3.  **Strategies (`strategies/`)**: Individual trading strategy implementations. Each strategy resides in its own directory, containing the core logic and a suite of test scripts for execution.

## Building and Running

### 1. Install Dependencies

The project dependencies are managed via `pip`. The `README.md` indicates a `requirements.txt` file should exist.

```bash
# Navigate to the project root
cd C:\Users\liual\OneDrive\桌面\quantitative_framework

# Install required Python packages
pip install -r requirements.txt
```
*(TODO: A `requirements.txt` file was mentioned in the README but not found in the file listing. It should be created to lock dependencies.)*

### 2. Fetch Market Data

Before running any tests, you must download the necessary historical data. This is done by running the appropriate script from the `data_fetchers` directory.

```bash
# Example: To run a fetcher script directly (if it has a __main__ block)
python data_fetchers/bybit_sol_1h_fetcher.py
```

### 3. Run a Strategy Test

The primary entry points for running the framework are the `test_*.py` files located within each strategy's sub-directory (e.g., `strategies/bb_atr/`).

**To run a simple backtest:**
```bash
# Navigate to the strategy directory
cd strategies/bb_atr

# Run the backtest script
python test_backtest.py
```

**To run a parameter optimization:**
```bash
# Navigate to the strategy directory
cd strategies/bb_atr

# Run the optimization script
python test_optimization.py
```

**To run a walk-forward analysis:**
```bash
# Navigate to the strategy directory
cd strategies/bb_atr

# Run the walk-forward script
python test_walkforward.py
```
*Note: Configuration for each test (e.g., date range, strategy parameters) is managed within the `if __name__ == "__main__":` block of the respective test script.*

## Development Conventions

### Adding a New Strategy

1.  **Create Directory**: Create a new sub-directory inside `strategies/` (e.g., `strategies/my_new_strategy/`).
2.  **Implement Logic**: Inside the new directory, create a `strategy.py` file. This file should contain a class with a `generate_signals(self, df: pd.DataFrame, **params) -> pd.Series` method that returns a pandas Series of signals (1 for long, -1 for short, 0 for neutral).
3.  **Create Test Scripts**: Copy the `test_backtest.py`, `test_optimization.py`, and `test_walkforward.py` scripts from an existing strategy (like `bb_atr`) into your new directory.
4.  **Adapt Scripts**: Modify the copied test scripts to import and instantiate your new strategy class and adjust the parameter grids and default settings as needed.

### Lookahead Bias Prevention

A core principle of this framework is the strict avoidance of lookahead bias. When generating signals, any indicator or data point used for a decision on a given candle **must** be from a previous candle. The established convention is to use `.shift(1)` on indicator data before comparing it to the current price.

**Correct Implementation Example:**
```python
# Calculate the indicator on historical data, then shift it
indicator = df['close'].rolling(20).mean().shift(1)

# Now, compare the current price to the *previous* indicator value
signals[df['close'] > indicator] = 1
```

### Code Structure

*   **Reusable Logic**: Any function or class that can be used across multiple strategies (e.g., a new performance metric) should be added to the `shared/` library.
*   **Data**: All market data should be stored in the `data/` directory with a consistent naming scheme (e.g., `SYMBOL_TIMEFRAME_START_END.csv`).
*   **Results**: All output from tests (CSV files, JSON summaries, PNG charts) is automatically saved into a `results/` sub-directory within the corresponding strategy's folder.
