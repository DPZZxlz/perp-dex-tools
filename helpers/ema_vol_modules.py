import asyncio
import datetime as dt
import time
from abc import ABC, abstractmethod
from collections import deque, namedtuple
from decimal import Decimal as D
from enum import Enum
from typing import Dict, List
import math
import numpy as np
import logging
logger = logging.getLogger(__name__)

class RollingAnnualizedVolatility:
    """
    A class for calculating rolling annualized volatility.

    This class maintains a fixed-size window of price data and their corresponding
    timestamps. It provides methods to update the data and calculate the
    annualized volatility based on the stored price history.

    Attributes:
        prices (deque): A fixed-size deque storing historical prices.
        timestamps (deque): A fixed-size deque storing timestamps corresponding to the prices.
        logger (Logger): A logger instance for this class.
    """
    def __init__(self, window_size: int):
        self.prices = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

    def update(self, new_price: D, new_timestamp: float):
        """Update with a new price and its timestamp."""
        if len(self.prices) > 0 and new_price == self.prices[-1]:
            return
        self.prices.append(new_price)
        self.timestamps.append(D(new_timestamp))

    def get_value(self):
        """Calculate and return the annualized volatility based on stored prices."""
        if len(self.prices) < 10:
            return D(0)  # Not enough data to calculate volatility

        prices_array = np.array(self.prices, dtype=np.float64)
        log_returns_array = np.diff(np.log(prices_array))

        # Get timestamps and calculate time differences
        timestamps_array = np.array(self.timestamps, dtype=np.float64)
        time_diffs = np.diff(timestamps_array)

        # Create a mask for non-zero time differences
        non_zero_mask = time_diffs != 0

        # Filter log returns and time diffs to only include non-zero time diffs
        filtered_log_returns = log_returns_array[non_zero_mask]
        filtered_time_diffs = time_diffs[non_zero_mask]

        # If all time diffs were zero, return 0 (or handle this case as appropriate)
        if len(filtered_time_diffs) == 0:
            return D(0)

        # Normalize log returns by the square root of the time differences
        normalized_log_returns = filtered_log_returns / np.sqrt(filtered_time_diffs)

        # Calculate sample variance of normalized log returns
        variance = np.var(normalized_log_returns, ddof=1)  # ddof=1 for sample variance

        # Calculate daily volatility
        normalized_volatility = np.sqrt(variance)
        
        # Annualize the volatility
        avg_interval_ms = np.mean(filtered_time_diffs)
        ms_per_year = 365 * 24 * 60 * 60 * 1000  # milliseconds
        annualized_volatility = normalized_volatility * np.sqrt(ms_per_year / avg_interval_ms)

        return D(annualized_volatility)
    
    
    def get_mean(self) -> D:
        if not self.prices:
            return D('0')
        return sum(self.prices) / D(len(self.prices))