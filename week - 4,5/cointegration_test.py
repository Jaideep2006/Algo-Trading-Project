# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 16:02:18 2025

@author: Jaideep
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

ash_prices = pd.read_csv('prices_ash.csv')
luxray_prices = pd.read_csv('prices_luxray.csv')

ash_midprice =  ( ash_prices['bid_price_1'] + ash_prices['ask_price_1'] )/2
luxray_midprice =  ( luxray_prices['bid_price_1'] + luxray_prices['ask_price_1'] )/2


def cointegration_test(ash_prices, luxray_prices, verbose=True):
    # Step 1: Convert to named Series
    ash = pd.Series(ash_prices, name='ASH')
    luxray = pd.Series(luxray_prices, name='LUXRAY')

    # Step 2: Linear regression: ASH ~ LUXRAY
    X = sm.add_constant(luxray)
    model = sm.OLS(ash, X).fit()
    
    # Access parameters using correct names
    hedge_ratio = model.params['LUXRAY']
    intercept = model.params['const']

    # Step 3: Residual = actual - predicted
    residuals = ash - (hedge_ratio * luxray + intercept)

    # Step 4: ADF test
    adf_result = adfuller(residuals)

    if verbose:
        print("Engle-Granger Cointegration Test:")
        print(f"ADF Statistic: {adf_result[0]}")
        print(f"p-value: {adf_result[1]}")
        for key, value in adf_result[4].items():
            print(f"Critical Value ({key}): {value}")
        if adf_result[1] < 0.05:
            print("✅ Residuals are stationary → Series are cointegrated.")
        else:
            print("❌ Residuals are NOT stationary → No cointegration.")

    return {
        "hedge_ratio": hedge_ratio,
        "intercept": intercept,
        "adf_statistic": adf_result[0],
        "p_value": adf_result[1],
        "critical_values": adf_result[4],
        "cointegrated": adf_result[1] < 0.05
    }

cointegration_test(ash_midprice, luxray_midprice)
