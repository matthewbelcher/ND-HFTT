import os
import pandas as pd
from datetime import datetime

df = pd.read_csv(f'data/market_ticks/40_min/01_29_25.csv')
df = df[df['SYM_ROOT'] == 'SPY']
df['timestamp'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME_M'])
df = df.reset_index(drop=True)
print(df)
df['bid_diff'] = df['BID'].shift(1, fill_value=df['BID'].iloc[0]) - df['BID']
df['ask_diff'] = df['ASK'].shift(1, fill_value=df['ASK'].iloc[0]) - df['ASK']
df['spread'] = df['ASK'] - df['BID']

print(df['bid_diff'].min(), df['ask_diff'].min())
print(df['bid_diff'].max(), df['ask_diff'].max())
print(df['bid_diff'].mean(), df['ask_diff'].mean())

df = df[(abs(df['bid_diff']) < 1) & (abs(df['ask_diff']) < 1)]
print(df)

bid_median = df['BID'].median()
ask_median = df['ASK'].median()

df = df[(abs(df['BID'] - bid_median) < 0.25 * bid_median) & (abs(df['ASK'] - ask_median) < 0.25 * ask_median)]

print(df)
df = df[(df['timestamp'] > datetime(2025, 1, 29, 14, 0, 0)) & (df['timestamp'] < datetime(2025, 1, 29, 14, 0, 1))]
print(df)
print(df['ASK'].min(), df['BID'].min())
print(df['ASK'].max(), df['BID'].max())
print(df['ASK'].mean(), df['BID'].mean())
print(df['ASK'].std(), df['BID'].std())
