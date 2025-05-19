from datetime import datetime, timedelta
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

fomc_release_dates = sorted([re.search(r'monetary(.*)a.htm', x).group(1) for x in os.listdir('data/fomc_statements')])
fomc_release_dates = [datetime(year=int(date[0:4]), month=int(date[4:6]), day=int(date[6:8])) for date in fomc_release_dates]
fomc_release_dates = list(filter(lambda x: x >= datetime(2022,4 , 1), fomc_release_dates))
fomc_release_dates = sorted(fomc_release_dates, reverse=True)[1:]

rate_df = pd.read_csv(f'data/effr/effr.csv', parse_dates=['date'])
change_probs = []
for date in fomc_release_dates:
    prob_df = pd.read_csv(f'data/federal_funds/historical_probabilities/{date.month:02}_{date.day:02}_{date.year}.csv', parse_dates=['Date'])   

    old_rate = rate_df[rate_df['date'] == date]['target_high'].values[0]
    new_rate = rate_df[rate_df['date'] == date + timedelta(days=1)]['target_high'].values[0]
    change = new_rate - old_rate
    change_bp = int(change * 100)

    wmid = 0
    for col_change in prob_df.columns:
        if col_change == 'Date':
            continue
        change_num = int(col_change.replace('bp', ''))
        wmid += change_num * prob_df[prob_df['Date'] == date - timedelta(days=1)][col_change].values[0]
    
    change_prob = prob_df[prob_df['Date'] == date - timedelta(days=1)][f"{change_bp}bp"].values[0]

    if np.isnan(wmid):
        print(f"Missing WMID data for date: {date}")
        print(prob_df)

    change_probs.append((date, int(new_rate * 100), change_bp, wmid, wmid - change_bp, change_prob))

change_probs_df = pd.DataFrame(change_probs, columns=['date', 'new_rate', 'change_bp', 'change_wmid', 'surprise', 'change_prob'])

symbol = 'SPY'
diffs = []
for date in fomc_release_dates:
    df = pd.read_csv(f'data/market_ticks/clean_sampled/{symbol}/{date.month:02}_{date.day:02}_{str(date.year)[2:]}.csv')
    surprise =  change_probs_df[change_probs_df['date'] == date]['surprise'].values[0]
    change = change_probs_df[change_probs_df['date'] == date]['change_bp'].values[0]
    change_prob = change_probs_df[change_probs_df['date'] == date]['change_prob'].values[0]
    diffs.append((date, change, change_prob, surprise,  (df.iloc[-1]['BID'] * df.iloc[-1]['BIDSIZ'] + df.iloc[-1]['ASK'] * df.iloc[-1]['ASKSIZ']) / (df.iloc[-1]['BIDSIZ'] + df.iloc[-1]['ASKSIZ']) -  (df.iloc[0]['BID'] * df.iloc[0]['BIDSIZ'] + df.iloc[0]['ASK'] * df.iloc[0]['ASKSIZ']) / (df.iloc[0]['BIDSIZ'] + df.iloc[0]['ASKSIZ'])))

diff_df = pd.DataFrame(diffs, columns=['date', 'change', 'change_prob', 'surprise', 'market_diff'])

train_df = diff_df.sample(frac=0.5)
X = diff_df[['change', 'change_prob', 'surprise']]
Y = diff_df['market_diff']

model = LinearRegression()
reg = model.fit(X, Y)
print(reg.coef_)

orders = []
for date in fomc_release_dates:
    df = pd.read_csv(f'data/market_ticks/clean_sampled/{symbol}/{date.month:02}_{date.day:02}_{str(date.year)[2:]}.csv')
    surprise = change_probs_df[change_probs_df['date'] == date]['surprise'].values[0]
    change = change_probs_df[change_probs_df['date'] == date]['change_bp'].values[0]
    change_prob = change_probs_df[change_probs_df['date'] == date]['change_prob'].values[0]
    swing = reg.predict([[change, change_prob, surprise]])[0]
    
    print(change)
    if swing < 0:
        orders.append((date, 'SELL', symbol, surprise, change, change_prob, df.iloc[0]['BID'], df.iloc[-1]['ASK'], reg.predict([[change, change_prob, surprise]])[0]))
        
    elif swing > 0:
        orders.append((date, 'BUY', symbol, surprise, change, change_prob, df.iloc[0]['ASK'], df.iloc[-1]['BID'], reg.predict([[change, change_prob, surprise]])[0]))
        


# plt.plot(change_probs_df['date'], change_probs_df['market_diff'], 'o')
# plt.plot(change_probs_df['date'], reg.predict(X), color='red')
# plt.xlabel('Date')
# plt.ylabel('Market Diff')
# plt.title('Market Diff Over Time')
# plt.show()
    
orders_df = pd.DataFrame(orders, columns=['date', 'action', 'symbol', 'surprise', 'change_bp', 'change_prob', 'entry_price', 'exit_price', 'predict'])
orders_df['profit'] = orders_df.apply(lambda row: row['exit_price'] - row['entry_price'] if row['action'] == 'BUY' else row['entry_price'] - row['exit_price'] - 0.31, axis=1)
orders_df.sort_values(by='date', inplace=True)
print(orders_df)
print("Total Profit:", orders_df['profit'].sum())
print("Average Profit:", orders_df['profit'].mean())
print("Entry Price:", orders_df['entry_price'].median())

plt.plot(orders_df['date'], orders_df['profit'].cumsum())
plt.xlabel('Date')
plt.ylabel('PnL ($)')
plt.title('Strategy #2 PnL')
plt.show()

# plt.plot(orders_df['surprise'], orders_df['profit'], 'o')
# plt.xlabel('Surprise (bp)')
# plt.ylabel('Profit')
# plt.title('Profit vs Surprise')
# plt.show()
