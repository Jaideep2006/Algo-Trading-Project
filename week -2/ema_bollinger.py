# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 10:17:26 2025

@author: Jaideep
"""

from src.backtester import Order, OrderBook
from typing import List
import numpy as np

class Trader:
    def __init__(self):
        self.mid_prices = []
        self.period = 200
        self.std_multiplier = 2.0

    

    def ema(self, prices):

        if len(prices) < self.period:
            return None

        prices = np.array(prices[-(self.period + 1):]) 

        alpha = 2 / (self.period + 1)
        ema_current = prices[0]  # Start with the first price

        for price in prices[1:]:
            ema_current = alpha * price + (1 - alpha) * ema_current

        return ema_current
    
    def calculate_bollinger_bands(self, prices):
        if len(prices) < self.period:
            return None, None
        
        window = np.array(prices[-self.period:])
        sma = np.mean(window)
        std = np.std(window, ddof=1)
        upper_band = sma + self.std_multiplier * std
        lower_band = sma - self.std_multiplier * std
        return upper_band, lower_band
    
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
        upper_band, lower_band = self.calculate_bollinger_bands(self.mid_prices)
        ema_val = self.ema(self.mid_prices)
        if upper_band is None or ema_val is None:
            return result  # Not enough data

        # Signal-based order placement
        if mid_price < lower_band and mid_price >= ema_val :
            # Buy at best ask
            orders.append(Order("PRODUCT", price=best_ask, quantity = min(10, best_bid_vol)))                   
        elif mid_price > upper_band and mid_price <= ema_val :
            # Sell at best bid
            orders.append(Order("PRODUCT", price=best_bid,quantity = -min(10, best_ask_vol)))

        result["PRODUCT"] = orders
        return result


   