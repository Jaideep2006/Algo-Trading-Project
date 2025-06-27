# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 09:52:40 2025

@author: Jaideep
"""
from src.backtester import Order
from typing import List
import numpy as np

class Trader:
    def __init__(self):
        self.mid_prices = []
        self.lookback = 100  # Lookback window for breakout strategy

    def calculate_breakout_bands(self, prices):
        if len(prices) <= self.lookback:
            return None, None

        window = prices[-(self.lookback+1):-1]  # exclude today
        upper_band = max(window)
        lower_band = min(window)
        return upper_band, lower_band

    def run(self, state, position):
        result = {}
        orders: List[Order] = []
        order_depth = state.order_depth

        if not order_depth.sell_orders or not order_depth.buy_orders:
            return result

        # Extract best prices and volumes
        best_ask, best_ask_vol = list(order_depth.sell_orders.items())[0]
        best_bid, best_bid_vol = sorted(order_depth.buy_orders.items(), reverse=True)[0]
        mid_price = (best_ask + best_bid) / 2

        self.mid_prices.append(mid_price)

        # Compute breakout bands
        upper_band, lower_band = self.calculate_breakout_bands(self.mid_prices)
        if upper_band is None:
            return result  # Not enough data yet

        # Signal-based order placement
        if mid_price < lower_band:
            # Buy signal: price breaks below lower band
            orders.append(Order("PRODUCT", price=best_ask, quantity=min(10, best_ask_vol)))
        elif mid_price > upper_band:
            # Sell signal: price breaks above upper band
            orders.append(Order("PRODUCT", price=best_bid, quantity=-min(10, best_bid_vol)))

        result["PRODUCT"] = orders
        return result
