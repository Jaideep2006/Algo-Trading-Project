# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 14:56:03 2025

@author: Jaideep
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


abra = pd.read_csv('abra_price.csv')
drowzee = pd.read_csv('drowzee_prices.csv')
sudowoodo = pd.read_csv('sudowoodo_prices.csv')
ash = pd.read_csv('prices_ash.csv')
misty = pd.read_csv('prices_misty.csv')
jolteon = pd.read_csv('prices_jolteon.csv')
luxray = pd.read_csv('prices_luxray.csv')
shinx = pd.read_csv('prices_shinx.csv')

abra_mid = (abra['bid_price_1'] + abra['ask_price_1'] )/2
drowzee_mid = (drowzee['bid_price_1'] + drowzee['ask_price_1'] )/2
sudowoodo_mid = (sudowoodo['bid_price_1'] + sudowoodo['ask_price_1'] )/2
ash_mid = (ash['bid_price_1'] + ash['ask_price_1'] )/2
misty_mid = (misty['bid_price_1'] + misty['ask_price_1'] )/2
jolteon_mid = (jolteon['bid_price_1'] + jolteon['ask_price_1'] )/2
luxray_mid = (luxray['bid_price_1'] + luxray['ask_price_1'] )/2
shinx_mid = (shinx['bid_price_1'] + shinx['ask_price_1'] )/2

df = pd.DataFrame({
    'abra': abra_mid,
    'drowzee': drowzee_mid,
    'sudowoodo': sudowoodo_mid,
    'jolteon': jolteon_mid,
    'luxray': luxray_mid,
    'shinx': shinx_mid,
    'ash' : ash_mid,
    'misty' : misty_mid
})

df.corr()

plt.figure(figsize = (12,8))
sns.set(font_scale = 1.4)
sns.heatmap(df.corr(), cmap = 'Reds' ,annot = True, annot_kws={"size" :15}, vmax = 0.6)
plt.show()