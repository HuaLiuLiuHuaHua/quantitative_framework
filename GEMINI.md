
# Project Overview

This project is a Python-based quantitative trading framework designed for the development, backtesting, and analysis of algorithmic trading strategies. It provides a robust set of tools to handle tasks such as data fetching, strategy implementation, performance evaluation, and statistical validation. The framework is built with a strong emphasis on preventing lookahead bias and ensuring the statistical significance of backtest results.

## Key Technologies

*   **Python**: The core programming language.
*   **Pandas**: For data manipulation and analysis.
*   **NumPy**: For numerical operations.
*   **Matplotlib & Seaborn**: For data visualization.
*   **ta**: A library for technical analysis, used to calculate indicators.
*   **Numba**: For accelerating Python code, particularly in signal calculation.

## Architecture

The framework is organized into several key directories:

*   `data/`: Stores historical price data for various assets.
*   `data_fetchers/`: Contains scripts for downloading data from exchanges like Bybit.
*   `factors/`: Provides a library of "factor operators" for creating custom trading signals and indicators.
*   `shared/`: Contains the core components of the framework, including:
    *   `backtest.py`: Vectorized and event-driven backtesting engines.
    *   `optimizer.py`: Tools for strategy parameter optimization.
    *   `walkforward.py`: For conducting walk-forward analysis to test strategy robustness.
    *   `mcpt.py`: Implements the Monte Carlo Permutation Test for statistical significance.
*   `strategies/`: Contains implementations of different trading strategies.

# Building and Running

## 1. Install Dependencies

To set up the environment, install the required Python packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## 2. Fetch Data

The framework includes scripts to fetch historical data. For example, to get BTC/USDT 1-hour data from Bybit, you can run:

```python
from data_fetchers.bybit_btc_1h_fetcher import fetch_bybit_btc_1h_data

df = fetch_bybit_btc_1h_data(
    start_date="2024-01-01",
    end_date="2024-12-31",
    save=True
)
```

## 3. Run a Backtest

Each strategy directory contains a `test_backtest.py` script to run a backtest for that strategy. For example, to backtest the `bb_atr` strategy:

```bash
cd strategies/bb_atr
python test_backtest.py
```

## 4. Run a Parameter Optimization

To find the best parameters for a strategy, you can run the optimization script:

```bash
cd strategies/bb_atr
python test_optimization_random.py
```

# Development Conventions

## Creating a New Strategy

To create a new trading strategy, follow these steps:

1.  **Create a new directory** under `strategies/`.
2.  **Implement the strategy logic** in a `strategy.py` file. The strategy class should include:
    *   A `generate_signals` method that takes a DataFrame and returns a Series of trading signals (1 for long, -1 for short, 0 for flat).
    *   Methods to define default parameters and a parameter grid for optimization.
3.  **Create a `test_backtest.py` script** to run a backtest of the strategy.
4.  **Create a `test_optimization_random.py` script** to run a parameter optimization.

## Avoiding Lookahead Bias

A core principle of the framework is to avoid lookahead bias. When generating signals, always use shifted data to ensure that decisions are made only with information available at that time. For example, when using a moving average, use `.shift(1)` to access the previous bar's value:

```python
# Correct: Use the previous bar's moving average
ma = df['close'].rolling(20).mean().shift(1)
signals[df['close'] > ma] = 1
```
