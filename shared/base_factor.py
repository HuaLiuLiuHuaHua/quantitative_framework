from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class BaseFactor(ABC):
    """
    Base class for all factors.
    All factors should inherit from this class and implement the calculate method.
    """

    @abstractmethod
    def calculate(self, price_data: Dict[str, pd.DataFrame], **kwargs: Any) -> pd.DataFrame:
        """
        Calculates the factor values.

        Args:
            price_data (Dict[str, pd.DataFrame]): A dictionary where keys are symbols and values are pandas DataFrames with price data.
            **kwargs: Additional parameters for factor calculation.

        Returns:
            pd.DataFrame: A DataFrame with timestamps as index, symbols as columns, and factor values.
        """
        raise NotImplementedError("Factors must implement the calculate() method.")
