# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 18:42:41 2025

@author: Jaideep
"""

from src.backtester import Order, OrderBook
from typing import List
import numpy as np

class Trader:
    def __init__(self):
        self.mid_prices = []
        self.period = 50

    
    def sma(self, arr):
        window = np.array(arr[-self.period:])
        return np.mean(window)
        
    def calculate_rsi(self, prices):
        if len(prices) < self.period:
            return None

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = self.sma(gains)
        avg_loss = self.sma(losses)

        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def run(self, state, position):
        result = {}
        orders: List[Order] = []
        order_depth: OrderBook = state.order_depth

        if not order_depth.sell_orders or not order_depth.buy_orders:
            return result

        # Get best bid and best ask
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        # Calculate mid-price
        mid_price = (best_bid + best_ask) / 2
        self.mid_prices.append(mid_price)

        # Only proceed if we have enough data for RSI
        if len(self.mid_prices) < self.period:
            return result

        rsi = self.calculate_rsi(self.mid_prices)

        # Strategy: Buy low RSI, sell high RSI
        if rsi is not None:
            if rsi < 30:
                # Oversold → Buy
                buy_volume = min(10, order_depth.sell_orders[best_ask])  # limit volume
                orders.append(Order( "PRODUCT", price=best_ask, quantity=buy_volume))
            elif rsi > 70:
                # Overbought → Sell
                sell_volume = min(10, order_depth.buy_orders[best_bid])
                orders.append(Order( "PRODUCT", price=best_bid, quantity=-sell_volume))

        result["PRODUCT"] = orders
        return result

