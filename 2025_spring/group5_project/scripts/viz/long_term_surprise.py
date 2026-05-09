import os
import re
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

fomc_release_dates = sorted([re.search(r'monetary(.*)a.htm', x).group(1) for x in os.listdir('data/fomc_statements')])
fomc_release_dates = [datetime(year=int(date[0:4]), month=int(date[4:6]), day=int(date[6:8])) for date in fomc_release_dates]
fomc_release_dates = list(filter(lambda x: x >= datetime(2022,4 , 1), fomc_release_dates))
fomc_release_dates = sorted(fomc_release_dates, reverse=True)[1:]
print([x.strftime("%m-%d-%Y") for x in fomc_release_dates])

rate_df = pd.read_csv(f'data/effr/effr.csv', parse_dates=['date'])

change_probs = []
for date in fomc_release_dates:
    prob_df = pd.read_csv(f'data/federal_funds/historical_probabilities/{date.month:02}_{date.day:02}_{date.year}.csv', parse_dates=['Date'])   

    old_rate = rate_df[rate_df['date'] == date]['target_high'].values[0]
    new_rate = rate_df[rate_df['date'] == date + timedelta(days=1)]['target_high'].values[0]
    change = new_rate - old_rate
    change_bp = int(change * 100)

    wmid = 0
    for change in prob_df.columns:
        if change == 'Date':
            continue
        change_num = int(change.replace('bp', ''))
        wmid += change_num * prob_df[prob_df['Date'] == date - timedelta(days=1)][change].values[0]

    if np.isnan(wmid):
        print(f"Missing WMID data for date: {date}")
        print(prob_df)

    change_probs.append((date, int(new_rate * 100), change_bp, wmid, wmid - change_bp))

change_probs_df = pd.DataFrame(change_probs, columns=['date', 'new_rate', 'change_bp', 'change_wmid', 'surprise'])

symbol_changes = {}

for symbol in ['VOO', 'SPY', 'VFH']:
    symbol_changes[symbol] = []

for symbol in ['VOO', 'SPY', 'VFH']:
    for date in fomc_release_dates:
        df = pd.read_csv(f'data/market_ticks/clean_sampled/{symbol}/{date.month:02}_{date.day:02}_{str(date.year)[2:]}.csv')
        df['timestamp'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME_M'])
        df.drop(columns=['DATE', 'SYM_SUFFIX'], inplace=True)

        market_df = df[(df['SYM_ROOT'] == symbol)]
        market_df['WMID'] = (market_df['BID'] * market_df['BIDSIZ'] + market_df['ASK'] * market_df['ASKSIZ']) / (market_df['BIDSIZ'] + market_df['ASKSIZ'])

        price_at_release = market_df[market_df['timestamp'] < datetime(date.year, date.month, date.day, 14, 0,0,0)].iloc[-1]['WMID']
        price_at_one_ms = market_df[market_df['timestamp'] < datetime(date.year, date.month, date.day, 14, 0,0,100)].iloc[-1]['WMID']
        price_at_one_s = market_df[market_df['timestamp'] < datetime(date.year, date.month, date.day, 14, 0,1,0)].iloc[-1]['WMID']
        price_at_one_m = market_df[market_df['timestamp'] < datetime(date.year, date.month, date.day, 14, 1,0,0)].iloc[-1]['WMID']
        price_at_five_m = market_df[market_df['timestamp'] < datetime(date.year, date.month, date.day, 14, 5,0,0)].iloc[-1]['WMID']
        price_at_ten_m = market_df[market_df['timestamp'] < datetime(date.year, date.month, date.day, 14, 10,0,0)].iloc[-1]['WMID']
        price_at_thirty_m = market_df[market_df['timestamp'] < datetime(date.year, date.month, date.day, 14, 30,0,0)].iloc[-1]['WMID']
    
        symbol_changes[symbol].append((date, price_at_release, (price_at_one_ms - price_at_release) / price_at_release, (price_at_one_s - price_at_release) / price_at_release, (price_at_one_m - price_at_release) / price_at_release, (price_at_five_m - price_at_release) / price_at_release, (price_at_ten_m - price_at_release) / price_at_release, (price_at_thirty_m - price_at_release) / price_at_release))

    symbol_change_dfs = {}
    for symbol, data in symbol_changes.items():
        symbol_change_dfs[symbol] = pd.DataFrame(data, columns=['date', 'price_at_release', 'percent_change_1ms', 'percent_change_1s', 'percent_change_1m', 'percent_change_5m', 'percent_change_10m', 'percent_change_30m'])

for symbol in ['VOO', 'SPY', 'VFH']:
    print(symbol_change_dfs[symbol])
# fig, ax1 = plt.subplots()

# color = 'tab:red'
# ax1.set_xlabel('FOMC Release Date')
# ax1.set_ylabel('Difference in Forecast and Actual', color=color)
# ax1.plot(change_probs_df['date'], change_probs_df['surprise'], color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('VOO Change (%)', color=color)  # we already handled the x-label with ax1
# ax2.plot(voo_changes_df['date'], voo_changes_df['percent_change_1ms'], label='1 ms')
# ax2.plot(voo_changes_df['date'], voo_changes_df['percent_change_1s'], label='1 s')
# ax2.plot(voo_changes_df['date'], voo_changes_df['percent_change_1m'], label='1 min')
# ax2.plot(voo_changes_df['date'], voo_changes_df['percent_change_5m'], label='5 min', color='tab:cyan')
# ax2.legend()
# ax2.tick_params(axis='y', labelcolor=color)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# fig.legend()
# plt.show()

# for symbol, changes_df in symbol_change_dfs.items():
#     surprises = []
#     for date in fomc_release_dates:
#         surprises.append((date, change_probs_df[change_probs_df['date'] == date]['surprise'].values[0], changes_df[changes_df['date'] == date]['percent_change_1ms'].values[0], changes_df[changes_df['date'] == date]['percent_change_1s'].values[0], changes_df[changes_df['date'] == date]['percent_change_1m'].values[0], changes_df[changes_df['date'] == date]['percent_change_5m'].values[0], changes_df[changes_df['date'] == date]['percent_change_10m'].values[0], changes_df[changes_df['date'] == date]['percent_change_30m'].values[0]))

#     surprises_df = pd.DataFrame(surprises, columns=['date', 'surprise', 'percent_change_1ms', 'percent_change_1s', 'percent_change_1m', 'percent_change_5m', 'percent_change_10m', 'percent_change_30m'])
#     plt.plot(surprises_df['surprise'], surprises_df['percent_change_1ms'], 'o', label='1 ms')
#     plt.plot(surprises_df['surprise'], surprises_df['percent_change_1s'], 'o', label='1 s')
#     plt.plot(surprises_df['surprise'], surprises_df['percent_change_1m'], 'o', label='1 min')
#     plt.plot(surprises_df['surprise'], surprises_df['percent_change_5m'], 'o', label='5 min')
#     plt.plot(surprises_df['surprise'], surprises_df['percent_change_10m'], 'o', label='10 min')
#     plt.plot(surprises_df['surprise'], surprises_df['percent_change_30m'], 'o', label='30 min')
#     plt.legend()
#     plt.xlabel('Surprise (bp)')
#     plt.ylabel(f'{symbol} Change (%)')
#     plt.show()

for symbol, df in symbol_change_dfs.items():
    relationship = []
    for date in fomc_release_dates:
        print(change_probs_df[change_probs_df['date'] == date])
        relationship.append((float(change_probs_df[change_probs_df['date'] == date]['surprise'].values[0]), float(df[df['date'] == date]['percent_change_30m'].values[0])))

    x, y = list(zip(*relationship))
    print(x)
    print(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(f'Symbol: {symbol}, Slope: {slope}, Intercept: {intercept}, R-squared: {r_value**2}, P-value: {p_value}, Std Err: {std_err}')

    lin_reg = lambda x: slope * x + intercept
    plt.plot(x, y, 'o')
    plt.plot(x, [lin_reg(i) for i in x], color='red')
    plt.xlabel('Surprise (bp)')
    plt.ylabel(f'{symbol} Change (%)')
    plt.title(f'Surprise vs {symbol} Change')
    plt.show()
    
