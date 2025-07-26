# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 10:10:58 2025

@author: Jaideep
"""

from src.backtester import Order
from typing import List
import numpy as np

class Trader:
    def __init__(self):
        self.mid_prices = []
        self.lookback = 20
        self.stop_loss_threshold = 5  # price units
        self.trailing_stop_buffer = 3
        self.entry_price = None
        self.trailing_stop_price = None

    def calculate_breakout_bands(self, prices):
        if len(prices) <= self.lookback:
            return None, None
        window = prices[-(self.lookback+1):-1]  # exclude current day
        upper_band = max(window)
        lower_band = min(window)
        return upper_band, lower_band

    def run(self, state, position):
        result = {}
        orders: List[Order] = []
        order_depth = state.order_depth

        if not order_depth.sell_orders or not order_depth.buy_orders:
            return result

        best_ask, best_ask_vol = list(order_depth.sell_orders.items())[0]
        best_bid, best_bid_vol = sorted(order_depth.buy_orders.items(), reverse=True)[0]
        mid_price = (best_ask + best_bid) / 2

        self.mid_prices.append(mid_price)

        upper_band, lower_band = self.calculate_breakout_bands(self.mid_prices)
        if upper_band is None:
            return result  # Not enough data

        # === ENTRY LOGIC ===
        if position == 0:
            if mid_price < lower_band:
                # Enter long
                orders.append(Order("PRODUCT", price=best_ask, quantity=min(10, best_ask_vol)))
                self.entry_price = mid_price
                self.trailing_stop_price = mid_price - self.trailing_stop_buffer
            elif mid_price > upper_band:
                # Enter short
                orders.append(Order("PRODUCT", price=best_bid, quantity=-min(10, best_bid_vol)))
                self.entry_price = mid_price
                self.trailing_stop_price = mid_price + self.trailing_stop_buffer

        # === EXIT LOGIC FOR LONG ===
        elif position > 0 and self.entry_price is not None:
            if mid_price <= self.entry_price - self.stop_loss_threshold:
                orders.append(Order("PRODUCT", price=best_bid, quantity=-position))
                self.entry_price = None
                self.trailing_stop_price = None
            elif mid_price > self.entry_price:
                self.trailing_stop_price = max(self.trailing_stop_price or mid_price - self.trailing_stop_buffer,
                                               mid_price - self.trailing_stop_buffer)

                if self.trailing_stop_price is not None and mid_price <= self.trailing_stop_price:
                    orders.append(Order("PRODUCT", price=best_bid, quantity=-position))
                    self.entry_price = None
                    self.trailing_stop_price = None
                elif mid_price > upper_band:
                    orders.append(Order("PRODUCT", price=best_bid, quantity=-position))
                    self.entry_price = None
                    self.trailing_stop_price = None

        # === EXIT LOGIC FOR SHORT ===
        elif position < 0 and self.entry_price is not None:
            if mid_price >= self.entry_price + self.stop_loss_threshold:
                orders.append(Order("PRODUCT", price=best_ask, quantity=-position))
                self.entry_price = None
                self.trailing_stop_price = None
            elif mid_price < self.entry_price:
                self.trailing_stop_price = min(self.trailing_stop_price or mid_price + self.trailing_stop_buffer,
                                               mid_price + self.trailing_stop_buffer)

                if self.trailing_stop_price is not None and mid_price >= self.trailing_stop_price:
                    orders.append(Order("PRODUCT", price=best_ask, quantity=-position))
                    self.entry_price = None
                    self.trailing_stop_price = None
                elif mid_price < lower_band:
                    orders.append(Order("PRODUCT", price=best_ask, quantity=-position))
                    self.entry_price = None
                    self.trailing_stop_price = None

        result["PRODUCT"] = orders
        return result
