import os
import re
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

fomc_release_dates = sorted([re.search(r'monetary(.*)a.htm', x).group(1) for x in os.listdir('data/fomc_statements')])
fomc_release_dates = [datetime(year=int(date[0:4]), month=int(date[4:6]), day=int(date[6:8])) for date in fomc_release_dates]
fomc_release_dates = list(filter(lambda x: x >= datetime(2022, 2, 1), fomc_release_dates))
fomc_release_dates = sorted(fomc_release_dates, reverse=True)[1:]

rate_df = pd.read_csv(f'data/effr/effr.csv', parse_dates=['date'])
print(rate_df.head())

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
        wmid += change_num * prob_df[prob_df['Date'] == date - timedelta(days=7)][change].values[0]

    change_probs.append((date, int(new_rate * 100), change_bp, wmid))

change_probs_df = pd.DataFrame(change_probs, columns=['date', 'new_rate', 'change_bp', 'change_wmid'])

plt.plot(change_probs_df['date'], change_probs_df['change_wmid'], color='tab:red', label='WMID of Forecasted Change')
plt.plot(change_probs_df['date'], change_probs_df['change_bp'], color='tab:blue', label='Actual Change')
plt.legend()
plt.xlabel('FOMC Release Date')
plt.ylabel('Change (bp)')
plt.show()