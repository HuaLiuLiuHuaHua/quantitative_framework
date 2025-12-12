"""
Base Strategy Class
All strategies must inherit from this class.

Design Principle:
- Compatible with two backtest engines:
  1. Vectorized Engine: Implement generate_signals()
  2. Event-Driven Engine: Implement add_indicators(), check_entry_signal(), check_exit_signal()
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import dataclasses


@dataclasses.dataclass
class Position:
    """Represents a trading position."""
    entry_price: float
    quantity: float
    direction: int  # 1 for long, -1 for short
    entry_timestamp: pd.Timestamp
    stop_loss_price: float
    take_profit_price: float


class BaseStrategy(ABC):
    """
    Base class for all strategies.

    All strategies should inherit from this class.
    Implement the corresponding methods based on the chosen backtest engine.
    """

    def __init__(self, **params: Any):
        """
        Initializes the strategy and sets its parameters.

        Args:
            **params: A dictionary of parameters for the strategy.
        """
        self.params = params

    @abstractmethod
    def get_required_lookback(self, **params) -> int:
        """
        (Required for Backtest Engines) Returns the lookback period required by the strategy.

        This method should calculate and return the maximum number of bars required
        for all indicators to warm up properly, based on the given parameters.
        This is crucial for providing a "warm-up" period for walk-forward OOS tests.

        Args:
            **params: The specific parameter set being used.

        Returns:
            int: The lookback period.
        """
        raise NotImplementedError("Strategies must implement get_required_lookback(**params)")

    # --- For Event-Driven Engine ---

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        (Optional for Event-Driven Engine) Adds all necessary technical indicators to the data.
        
        Args:
            df: The OHLCV data.

        Returns:
            A new DataFrame with the indicators.
        """
        return df

    def check_entry_signal(self, df: pd.DataFrame, i: int) -> int:
        """
        (Required for Event-Driven Engine) Checks for an entry signal at the given index.

        Args:
            df: The complete data with indicators.
            i: The index of the current bar.

        Returns:
            int: 1 for long, -1 for short, 0 for no signal.
        """
        raise NotImplementedError("Event-driven strategies must implement check_entry_signal()")

    def check_exit_signal(self, df: pd.DataFrame, i: int, position: Position) -> bool:
        """
        (Optional for Event-Driven Engine) Checks for an exit signal at the given index.

        Args:
            df: The complete data with indicators.
            i: The index of the current bar.
            position: The current position object.

        Returns:
            bool: True to close the position, False to hold.
        """
        return False

    # --- For Vectorized Engine ---

    def generate_signals(self, df: pd.DataFrame, **params: Any) -> pd.Series:
        """
        (Required for Vectorized Engine) Generates a complete series of trade signals.
        
        Args:
            df: The OHLCV data.
            **params: Strategy parameters.

        Returns:
            pd.Series: A series of trade signals (1, -1, 0).
        """
        raise NotImplementedError("Vectorized strategies must implement generate_signals()")
