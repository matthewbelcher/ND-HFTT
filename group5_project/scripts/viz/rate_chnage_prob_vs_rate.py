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
    change_prob = prob_df[prob_df['Date'] == date - timedelta(days=1)][f'{change_bp}bp'].values[0]

    change_probs.append((date, int(new_rate * 100), change_bp, change_prob))

change_probs_df = pd.DataFrame(change_probs, columns=['date', 'new_rate', 'change_bp', 'change_prob'])
print(change_probs_df)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('FOMC Release Date')
ax1.set_ylabel('Probabilitiy of Actual Change Occurring', color=color)
ax1.plot(change_probs_df['date'], change_probs_df['change_prob'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Rate (bp)', color=color)  # we already handled the x-label with ax1
ax2.plot(change_probs_df['date'], change_probs_df['new_rate'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()