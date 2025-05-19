import os
import re
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def all_dates():
    fomc_release_dates = sorted([re.search(r'monetary(.*)a.htm', x).group(1) for x in os.listdir('data/fomc_statements')])
    fomc_release_dates = [datetime(year=int(date[0:4]), month=int(date[4:6]), day=int(date[6:8])) for date in fomc_release_dates]
    fomc_release_dates = list(filter(lambda x: x >= datetime(2022, 2, 1), fomc_release_dates))
    fomc_release_dates = sorted(fomc_release_dates, reverse=True)[1:]

    rate_df = pd.read_csv(f'data/effr/effr.csv', parse_dates=['date'])
    # print(rate_df.head())

    for date in fomc_release_dates:
        prob_df = pd.read_csv(f'data/federal_funds/historical_probabilities/{date.month:02}_{date.day:02}_{date.year}.csv', parse_dates=['Date'])   
        prev_day = date - pd.Timedelta(days=1)

        old_rate = rate_df[rate_df['date'] == date]['target_high'].values[0]
        new_rate = rate_df[rate_df['date'] == date + timedelta(days=1)]['target_high'].values[0]
        change = new_rate - old_rate

        change_bp = int(change * 100)

        num_days_acc = 0
        prev_day_pred = prob_df[prob_df['Date'] == prev_day][f'{change_bp}bp'].values[0]

        wmids = []

        if date == pd.to_datetime('2022-03-16'):
            continue

        while True:
            num_days_acc += 1
            if num_days_acc >= len(prob_df) - 2:
                break
            curr_pred = prob_df[prob_df['Date'] == prev_day - timedelta(days=num_days_acc)][f'{change_bp}bp'].values[0]
            if abs(curr_pred - prev_day_pred) > 0.1:
                break
        print(date)
        print(f'number of days with prediction less that 1% away from day before release: {num_days_acc}')

        wmid = 0
        for i in range(1, 31):
            for change in prob_df.columns:
                if change == 'Date':
                    continue
                change_num = int(change.replace('bp', ''))
                wmid += change_num * prob_df[prob_df['Date'] == date - timedelta(days=i)][change].values[0]
            wmids.append(wmid)
        print(f"mean: {np.mean(wmids)}")
        print(f"max: {np.max(wmids)} at {wmids.index(np.max(wmids)) + 1} days before release")
        print(f"min: {np.min(wmids)} at {wmids.index(np.min(wmids)) + 1} days before release")
        print(f'range: {np.max(wmids) - np.min(wmids)}')
        print()

def plot_fluctuations(date):
    date = pd.to_datetime(date)
    rate_df = pd.read_csv(f'data/effr/effr.csv', parse_dates=['date'])
    print(rate_df.head())

    prob_df = pd.read_csv(f'data/federal_funds/historical_probabilities/{date.month:02}_{date.day:02}_{date.year}.csv', parse_dates=['Date'])   
    prev_day = date - pd.Timedelta(days=1)

    old_rate = rate_df[rate_df['date'] == date]['target_high'].values[0]
    new_rate = rate_df[rate_df['date'] == date + timedelta(days=1)]['target_high'].values[0]
    change = new_rate - old_rate

    change_bp = int(change * 100)

    wmids = []

    wmid = 0
    for i in range(1, 31):
        for change in prob_df.columns:
            if change == 'Date':
                continue
            change_num = int(change.replace('bp', ''))
            wmid += change_num * prob_df[prob_df['Date'] == date - timedelta(days=i)][change].values[0]
        wmids.append(wmid)
    wmids.reverse()
    plt.plot(range(-30, 0), wmids)
    plt.title(f"Days to release vs Weighted Mid Prediction on {date}")
    plt.xlabel("Days")
    plt.ylabel("Weighted Mid Prediction")
    plt.axhline(y=np.mean(wmids), color='r', linestyle='--', label=f'Mean: {np.mean(wmids):.2f}')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_fluctuations(sys.argv[1])
    else:
        all_dates()

