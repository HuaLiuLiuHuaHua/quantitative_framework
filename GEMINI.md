# Project Overview

This project is a professional quantitative trading framework written in Python. It is designed for the development, backtesting, and optimization of trading strategies, with a strong emphasis on avoiding lookahead bias and ensuring statistical robustness.

The framework is inspired by `MCPT-Main`, `Medium`, and `trading_framework`, and it integrates the best features of each.

## Core Features

-   **Strict Lookahead Bias Control**: All technical indicators use `shift(1)` to ensure that only historical data is used for signal generation.
-   **Comprehensive Backtesting Engine**: The framework includes both a vectorized and an event-driven backtesting engine. It provides accurate calculation of transaction costs (fees, slippage) and a full suite of performance metrics (Sharpe Ratio, Max Drawdown, Profit Factor, etc.).
-   **Efficient Parameter Optimization**: The framework supports grid search and random search for parameter optimization, with parallel processing capabilities to speed up the process.
-   **Statistical Validation**: It includes statistical tests like Bar Permutation from `MCPT-Main` to validate the significance of a strategy.
-   **Flexible Data Management**: The framework provides tools for automatically fetching and validating data from sources like Bybit.

# Building and Running

## 1. Install Dependencies

There is no `requirements.txt` file in the project. Based on the libraries used in the code (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `joblib`, `tqdm`), you can install the dependencies with the following command:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels joblib tqdm
```

## 2. Fetching Data

The `data_fetchers` directory contains scripts to download historical data. For example, to fetch 1-hour BTC data from Bybit, you can use the `bybit_btc_1h_fetcher.py` script.

## 3. Running a Strategy Backtest

The `strategies` directory contains different trading strategies. Each strategy has a `test_backtest.py` file to run a backtest. For example, to backtest the `donchian` strategy, you would run:

```bash
cd strategies/donchian
python test_backtest.py
```

## 4. Parameter Optimization

Each strategy also has a `test_optimization.py` file for parameter optimization. For example, to optimize the `donchian` strategy, you would run:

```bash
cd strategies/donchian
python test_optimization.py
```

# Development Conventions

## Creating a New Data Source

To create a new data source, you can copy an existing data fetcher and modify it with the new symbol and function name.

## Creating a New Strategy

To create a new strategy, you need to:

1.  Create a new directory under `strategies`.
2.  Create a `strategy.py` file that contains a class that inherits from `BaseStrategy` and implements the `generate_signals` method.
3.  Create `test_backtest.py` and `test_optimization.py` files to test the new strategy.

## Avoiding Lookahead Bias

A core principle of this framework is to avoid lookahead bias. This is achieved by using `shift(1)` on all technical indicators. This ensures that the signal for a given time period is generated using only data from previous time periods.
