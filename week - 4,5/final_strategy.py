# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 15:26:22 2025

@author: Jaideep
"""

from src.backtester import Order, OrderBook
from typing import List
import pandas as pd
import numpy as np
import statistics
import math

# Base Class
class BaseClass:
    def __init__(self, product_name, max_position):
        self.product_name = product_name
        self.max_position = max_position
    
    def get_orders(self, state, orderbook, position):
        """Override this method in product-specific strategies"""
        return []

class SudowoodoStrategy(BaseClass): # Inherit from Base Class
    def __init__(self):
        super().__init__("SUDOWOODO", 50) # Initialize using the init dunder from its Parent class
        self.fair_value = 10000
    
    def get_orders(self, state, orderbook, position):
        orders = []
        
        if not orderbook.buy_orders and not orderbook.sell_orders:
            return orders
        # LOGIC FROM THE NOTEBOOK SHARED ON NOTION FOR SUDOWOODO
        orders.append(Order(self.product_name, self.fair_value + 2, -10))
        orders.append(Order(self.product_name, self.fair_value - 2, 10))

        return orders

class DrowzeeStrategy(BaseClass):
    def __init__(self):
        super().__init__("DROWZEE", 50)
        self.lookback = 50
        self.z_threshold = 3.75
        self.prices = []
    
    def get_orders(self, state, orderbook, position):
        orders = []
        if not orderbook.buy_orders and not orderbook.sell_orders:
            return orders
        
        best_ask = min(orderbook.sell_orders.keys()) if orderbook.sell_orders else None
        best_bid = max(orderbook.buy_orders.keys()) if orderbook.buy_orders else None
        mid_price = (best_ask + best_bid) // 2
        self.prices.append(mid_price)

        if len(self.prices) > self.lookback:
            mean_price = statistics.mean(self.prices[-self.lookback:])
            stddev_price = statistics.stdev(self.prices[-self.lookback:])
            z_score = (mid_price - mean_price) / stddev_price
            if z_score > self.z_threshold:
                orders.append(Order(self.product_name, best_bid, -self.max_position + position))
            elif z_score < -self.z_threshold:
                orders.append(Order(self.product_name, best_ask, self.max_position - position))
            else:
                return self.market_make(mid_price)
        elif len(self.prices) <= self.lookback:
            return self.market_make(mid_price)
        return orders
    
    def market_make(self, price):
        orders = []
        orders.append(Order(self.product_name, price - 1, 25))
        orders.append(Order(self.product_name, price + 1, -25))
        return orders

class AbraStrategy(BaseClass):
    def __init__(self):
        super().__init__("ABRA", 50)
        self.prices = []
        self.lookback = 200
        self.z_threshold = 2.0
        self.z_mm_threshold = 0.3
        self.skew_factor = 0.1
    def get_orders(self, state, orderbook, position):
        orders = []

        if not orderbook.buy_orders and not orderbook.sell_orders:
            return orders

        best_ask = min(orderbook.sell_orders.keys()) if orderbook.sell_orders else None
        best_bid = max(orderbook.buy_orders.keys()) if orderbook.buy_orders else None
        mid_price = (best_ask + best_bid) // 2
        self.prices.append(mid_price)

        if len(self.prices) > self.lookback:
            mean_price = statistics.mean(self.prices[-self.lookback:])
            stddev_price = statistics.stdev(self.prices[-self.lookback:])
            z_score = (mid_price - mean_price) / stddev_price
            if z_score > self.z_threshold:
                orders.append(Order(self.product_name, best_bid, -7))
            elif z_score < -self.z_threshold:
                orders.append(Order(self.product_name, best_ask, 7))
            elif abs(z_score) < self.z_mm_threshold:
                return self.market_make(mid_price, position)
        elif len(self.prices) <= self.lookback:
            return self.market_make(mid_price, position)
        return orders

    def market_make(self, mid_price, position):
        orders = []
        adjusted_mid_price = mid_price + self.skew_factor*position
        orders.append(Order(self.product_name, adjusted_mid_price - 2, 7))
        orders.append(Order(self.product_name, adjusted_mid_price + 2, -7))
        return orders

class JolteonStrategy(BaseClass):
    def __init__(self):
        super().__init__("JOLTEON", 350)
        self.lookback = 200
        self.prices = []
        self.fast_period = 300
        self.slow_period = 100
        self.last_buy_price = None   # ← track latest buy

    def get_orders(self, state, orderbook, position):
        orders = []
        # no market data yet
        if not orderbook.buy_orders and not orderbook.sell_orders:
            return orders

        best_ask = min(orderbook.sell_orders.keys()) if orderbook.sell_orders else None
        best_bid = max(orderbook.buy_orders.keys()) if orderbook.buy_orders else None
        mid_price = (best_ask + best_bid) // 2
        self.prices.append(mid_price)

        # warm‑up until lookback
        if len(self.prices) <= self.lookback:
            return self._market_make_with_check(mid_price)

        # trend‑based entry/exit
        fast_ema = self.ema(self.prices, self.fast_period)
        slow_ema = self.ema(self.prices, self.slow_period)

        # go long below
        if fast_ema is not None and slow_ema is not None and fast_ema > slow_ema:
            orders.append(Order(self.product_name, best_bid, 10))
            self.last_buy_price = best_bid

        # go short (sell) only if price > last buy
        elif fast_ema is not None and slow_ema is not None and fast_ema < slow_ema:
            if self.last_buy_price is not None and best_ask > self.last_buy_price:
                orders.append(Order(self.product_name, best_ask, -10))
            else:
                # if we can't sell above cost, just keep market‐making
                return self._market_make_with_check(mid_price)
        else:
            # neutral: keep quoting
            return self._market_make_with_check(mid_price)

        return orders

    def _market_make_with_check(self, price):
        """Market‐make but only sell above last_buy_price."""
        mm_orders = []
        bid_price = price - 1
        ask_price = price + 1

        # always place bid
        mm_orders.append(Order(self.product_name, bid_price, 25))

        # only place ask if above last buy
        if self.last_buy_price is None or ask_price > self.last_buy_price:
            mm_orders.append(Order(self.product_name, ask_price, -25))

        return mm_orders

    def ema(self, price_list, period):
        if len(price_list) < period + 1:
            return None
        data = np.array(price_list[-(period + 1):])
        alpha = 2 / (period + 1)
        ema_current = data[0]
        for p in data[1:]:
            ema_current = alpha * p + (1 - alpha) * ema_current
        return ema_current
    
class ShinxStrategy(BaseClass):
    def __init__(self):
        super().__init__("SHINX", 60)
        self.lookback = 200
        self.prices = []
        self.fast_period = 300
        self.slow_period = 100
        self.last_buy_price = None   # ← track latest buy

    def get_orders(self, state, orderbook, position):
        orders = []
        # no market data yet
        if not orderbook.buy_orders and not orderbook.sell_orders:
            return orders

        best_ask = min(orderbook.sell_orders.keys()) if orderbook.sell_orders else None
        best_bid = max(orderbook.buy_orders.keys()) if orderbook.buy_orders else None
        mid_price = (best_ask + best_bid) // 2
        self.prices.append(mid_price)

        # warm‑up until lookback
        if len(self.prices) <= self.lookback:
            return self._market_make_with_check(mid_price)

        # trend‑based entry/exit
        fast_ema = self.ema(self.prices, self.fast_period)
        slow_ema = self.ema(self.prices, self.slow_period)

        # go long below
        if fast_ema is not None and slow_ema is not None and fast_ema > slow_ema:
            orders.append(Order(self.product_name, best_bid, self.max_position -position))
            self.last_buy_price = best_bid

        # go short (sell) only if price > last buy
        elif fast_ema is not None and slow_ema is not None and fast_ema < slow_ema:
            if self.last_buy_price is not None and best_ask > self.last_buy_price:
                orders.append(Order(self.product_name, best_ask, -10))
            else:
                # if we can't sell above cost, just keep market‐making
                return self._market_make_with_check(mid_price)
        else:
            # neutral: keep quoting
            return self._market_make_with_check(mid_price)

        return orders

    def _market_make_with_check(self, price):
        """Market‐make but only sell above last_buy_price."""
        mm_orders = []
        bid_price = price - 1
        ask_price = price + 1

        # always place bid
        mm_orders.append(Order(self.product_name, bid_price, 25))

        # only place ask if above last buy
        if self.last_buy_price is None or ask_price > self.last_buy_price:
            mm_orders.append(Order(self.product_name, ask_price, -25))

        return mm_orders

    def ema(self, price_list, period):
        if len(price_list) < period + 1:
            return None
        data = np.array(price_list[-(period + 1):])
        alpha = 2 / (period + 1)
        ema_current = data[0]
        for p in data[1:]:
            ema_current = alpha * p + (1 - alpha) * ema_current
        return ema_current

class LuxrayStrategy(BaseClass):
    def __init__(self):
        super().__init__("LUXRAY", 250)
        self.lookback = 200
        self.prices = []
        self.rsi_period = 50
        self.overbought = 90
        self.oversold = 10
        self.last_buy_price = None   # track last buy for market-making

    def get_orders(self, state, orderbook, position):
        orders = []
        # no market data yet
        if not orderbook.buy_orders and not orderbook.sell_orders:
            return orders

        best_ask = min(orderbook.sell_orders.keys()) if orderbook.sell_orders else None
        best_bid = max(orderbook.buy_orders.keys()) if orderbook.buy_orders else None
        mid_price = (best_ask + best_bid) / 2
        self.prices.append(mid_price)

        # maintain lookback window
        if len(self.prices) > self.lookback:
            self.prices.pop(0)

        # until enough data, market-make
        if len(self.prices) < self.rsi_period + 1:
            return self._market_make_with_check(mid_price)

        # compute RSI
        rsi = self._compute_rsi(self.prices, self.rsi_period)

        # entry: oversold -> long, overbought -> short
        if rsi is not None and rsi <= self.oversold and position < self.max_position:
            # long entry
            orders.append(Order(self.product_name, best_ask, 10))
            self.last_buy_price = best_ask

        elif rsi is not None and rsi >= self.overbought and position > -self.max_position:
            # short entry
            orders.append(Order(self.product_name, best_bid, -self.max_position + position))

        # exit: back to neutral RSI
        elif rsi is not None and self.oversold < rsi < self.overbought and position != 0:
            # unwind full position
            
            return self._market_make_with_check(mid_price)
            """
            orders.append(Order(self.product_name,
                                best_bid if position > 0 else best_ask,
                                -position))"""

        # fallback market-making
        if not orders:
            return self._market_make_with_check(mid_price)

        return orders

    def _market_make_with_check(self, price):
        """Market-make but only sell above last_buy_price."""
        mm_orders = []
        bid_price = price - 1
        ask_price = price + 1

        # always place bid
        mm_orders.append(Order(self.product_name, bid_price, 25))

        # only place ask if above last buy
        if self.last_buy_price is None or ask_price > self.last_buy_price:
            mm_orders.append(Order(self.product_name, ask_price, -25))

        return mm_orders

    def _compute_rsi(self, prices: List[float], period: int) -> float:
        # convert to pandas series
        series = pd.Series(prices)
        delta = series.diff().dropna()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = -delta.clip(upper=0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else None


class AshStrategy(BaseClass):
    def __init__(self):
        super().__init__("ASH", 60)
        self.lookback = 100
        self.prices = []
        self.last_buy_price = None
        self.k = 2  # Bollinger Band width multiplier
        self.stop_loss_buffer = 10   # points below lower band to trigger stop loss
        self.extra_band_buffer = 5 # points above upper band to allow shorting

    def get_orders(self, state, orderbook, position):
        orders = []

        if not orderbook.buy_orders and not orderbook.sell_orders:
            return orders

        best_ask = min(orderbook.sell_orders.keys()) if orderbook.sell_orders else None
        best_bid = max(orderbook.buy_orders.keys()) if orderbook.buy_orders else None
        mid_price = (best_ask + best_bid) // 2
        self.prices.append(mid_price)

        if len(self.prices) <= self.lookback:
            return self._market_make_with_check(mid_price)

        # Bollinger Bands
        price_series = pd.Series(self.prices[-self.lookback:])
        ma = price_series.mean()
        std = price_series.std()
        upper_band = ma + self.k * std
        lower_band = ma - self.k * std

        # ✅ Stop-Loss logic: if price has dropped far below lower band
        
        if self.last_buy_price is not None and mid_price < (lower_band - self.stop_loss_buffer):
            orders.append(Order(self.product_name, best_bid, -self.max_position + position))
            self.last_buy_price = None  # reset after stop-loss
            return orders
        
        # ✅ BUY when price is below lower band
        if mid_price < lower_band:
            orders.append(Order(self.product_name, best_bid, 10))
            self.last_buy_price = best_bid

        # ✅ SELL when price is above upper band and profitable
        elif mid_price > upper_band:
            if self.last_buy_price is not None and best_ask > self.last_buy_price:
                orders.append(Order(self.product_name, best_ask, -self.max_position + position))
            # ✅ Extra: Aggressive shorting if price way above upper band
            elif best_ask > (upper_band + self.extra_band_buffer):
                orders.append(Order(self.product_name, best_ask, -10))

            else:
                return self._market_make_with_check(mid_price)

        else:
            return self._market_make_with_check(mid_price)

        return orders

    def _market_make_with_check(self, price):
        """Market-make but only sell above last_buy_price."""
        mm_orders = []
        bid_price = price - 1
        ask_price = price + 1

        mm_orders.append(Order(self.product_name, bid_price, 10))

        if self.last_buy_price is None or ask_price > self.last_buy_price:
            mm_orders.append(Order(self.product_name, ask_price, -10))

        return mm_orders

class MistyStrategy(BaseClass):
    def __init__(self):
        super().__init__("PRODUCT", 100)
        self.lookback = 100
        self.prices = []
        self.last_buy_price = None
        self.k = 2  # Bollinger Band width multiplier
        self.stop_loss_buffer = 10   # points below lower band to trigger stop loss
        self.extra_band_buffer = 5 # points above upper band to allow shorting

    def get_orders(self, state, orderbook, position):
        orders = []

        if not orderbook.buy_orders and not orderbook.sell_orders:
            return orders

        best_ask = min(orderbook.sell_orders.keys()) if orderbook.sell_orders else None
        best_bid = max(orderbook.buy_orders.keys()) if orderbook.buy_orders else None
        mid_price = (best_ask + best_bid) // 2
        self.prices.append(mid_price)

        if len(self.prices) <= self.lookback:
            return self._market_make_with_check(mid_price)

        # Bollinger Bands
        price_series = pd.Series(self.prices[-self.lookback:])
        ma = price_series.mean()
        std = price_series.std()
        upper_band = ma + self.k * std
        lower_band = ma - self.k * std

        # ✅ Stop-Loss logic: if price has dropped far below lower band
        
        if self.last_buy_price is not None and mid_price < (lower_band - self.stop_loss_buffer):
            orders.append(Order(self.product_name, best_bid, -self.max_position + position))
            self.last_buy_price = None  # reset after stop-loss
            return orders
        
        # ✅ BUY when price is below lower band
        if mid_price < lower_band:
            orders.append(Order(self.product_name, best_bid, 10))
            self.last_buy_price = best_bid

        # ✅ SELL when price is above upper band and profitable
        elif mid_price > upper_band:
            if self.last_buy_price is not None and best_ask > self.last_buy_price:
                orders.append(Order(self.product_name, best_ask, -self.max_position + position))
            # ✅ Extra: Aggressive shorting if price way above upper band
            elif best_ask > (upper_band + self.extra_band_buffer):
                orders.append(Order(self.product_name, best_ask, -10))

            else:
                return self._market_make_with_check(mid_price)

        else:
            return self._market_make_with_check(mid_price)

        return orders

    def _market_make_with_check(self, price):
        """Market-make but only sell above last_buy_price."""
        mm_orders = []
        bid_price = price - 1
        ask_price = price + 1

        mm_orders.append(Order(self.product_name, bid_price, 10))

        if self.last_buy_price is None or ask_price > self.last_buy_price:
            mm_orders.append(Order(self.product_name, ask_price, -10))

        return mm_orders

class Trader:
    MAX_LIMIT = 0 # for single product mode only, don't remove
    def __init__(self):
        self.strategies = {
            "SUDOWOODO": SudowoodoStrategy(),
            "DROWZEE": DrowzeeStrategy(), 
            "ABRA": AbraStrategy(),
            "JOLTEON": JolteonStrategy(),
            "SHINX": ShinxStrategy(),
            "LUXRAY": LuxrayStrategy(),
            "ASH": AshStrategy(),
            "MISTY": MistyStrategy()
        }
    
    def run(self, state):
        result = {}
        positions = getattr(state, 'positions', {})
        if len(self.strategies) == 1: self.MAX_LIMIT= self.strategies["PRODUCT"].max_position # for single product mode only, don't remove

        for product, orderbook in state.order_depth.items():
            current_position = positions.get(product, 0)
            product_orders = self.strategies[product].get_orders(state, orderbook, current_position)
            result[product] = product_orders
        
        return result, self.MAX_LIMIT