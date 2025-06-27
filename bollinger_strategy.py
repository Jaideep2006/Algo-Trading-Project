# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 11:47:51 2025

@author: Jaideep
"""

from src.backtester import Order
from typing import List
import numpy as np

class Trader:
    def __init__(self):
        self.mid_prices = []
        self.period = 100
        self.std_multiplier = 2

    def calculate_bollinger_bands(self, prices):
        if len(prices) < self.period:
            return None, None, None
        
        window = np.array(prices[-self.period:])
        sma = np.mean(window)
        std = np.std(window, ddof=1)
        upper_band = sma + self.std_multiplier * std
        lower_band = sma - self.std_multiplier * std
        return upper_band, sma, lower_band

    def run(self, state, position):
        result = {}
        orders: List[Order] = []
        order_depth = state.order_depth

        # Ensure there is liquidity on both sides
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return result

        # Extract best prices and volumes
        best_ask, best_ask_vol = list(order_depth.sell_orders.items())[0]
        best_bid, best_bid_vol = sorted(order_depth.buy_orders.items(), reverse=True)[0]
        mid_price = (best_ask + best_bid) / 2

        self.mid_prices.append(mid_price)

        # Compute Bollinger Bands
        upper_band, middle_band, lower_band = self.calculate_bollinger_bands(self.mid_prices)
        if upper_band is None:
            return result  # Not enough data

        # Signal-based order placement
        if mid_price <= lower_band:
            # Buy at best ask
            orders.append(Order("PRODUCT", price=best_ask, quantity = best_bid_vol))
        elif mid_price >= upper_band:
            # Sell at best bid
            orders.append(Order("PRODUCT", price=best_bid,quantity = -best_ask_vol))

        result["PRODUCT"] = orders
        return result
