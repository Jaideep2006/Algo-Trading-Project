# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 10:29:10 2025

@author: Jaideep
"""

from src.backtester import Order
from typing import List
import numpy as np

class Trader:
    def __init__(self):
        self.mid_prices = []
        self.fast_period = 100
        self.slow_period = 200
        self.prev_fast_ema = None
        self.prev_slow_ema = None

    def calculate_ema(self, prices, period, prev_ema=None):
        alpha = 2 / (period + 1)
        if prev_ema is None:
            # Initialize EMA with the SMA of the first 'period' prices
            if len(prices) < period:
                return None
            return np.mean(prices[-period:])
        else:
            return alpha * prices[-1] + (1 - alpha) * prev_ema
 
        if self.prev_fast_ema is not None and self.prev_slow_ema is not None:
            # Buy Signal: Fast EMA crosses above Slow EMA
 b                orders.append(Order("PRODUCT", price=best_ask, quantity=best_bid_vol))

            # Sell Signal: Fast EMA crosses below Slow EMA
            elif fast_ema < slow_ema:
                orders.append(Order("PRODUCT", price=best_bid, quantity=-best_ask_vol))

        # Update previous EMAs
        self.prev_fast_ema = fast_ema
        self.prev_slow_ema = slow_ema

        result["PRODUCT"] = orders
        return result

