import os
import pandas as pd
import re
from datetime import datetime
from matplotlib import pyplot as plt

def get_probability(release_date: datetime, relative_date: datetime) -> pd.DataFrame | None:
    df = pd.read_csv(f'data/federal_funds/historical_probabilities_3/{release_date.strftime("%m_%d_%Y")}.csv', parse_dates=['Date'])
    try:
        return df[df["Date"] <= relative_date].iloc[-1]
    except IndexError:
        return None

target_rates = pd.read_csv('data/target_rates/target_rates_4.csv', parse_dates=['date'])
target_rates['target_low'] = target_rates['target_low'].astype(float)
target_rates['target_high'] = target_rates['target_high'].astype(float)


fomc_release_dates = sorted([re.search(r'monetary(.*)a.htm', x).group(1) for x in os.listdir('data/fomc_statements')])
fomc_release_dates = [datetime(year=int(date[0:4]), month=int(date[4:6]), day=int(date[6:8])) for date in fomc_release_dates]

fomc_release_dates = list(filter(lambda x: x >= datetime(year=2008, month=12, day=16) and x <= datetime(year=2025, month=1, day=1), fomc_release_dates))

probs = []
for release in fomc_release_dates:
    print(release)
    try:
        prob_1d = get_probability(release, release - pd.Timedelta(days=1))
        prob_7d = get_probability(release, release - pd.Timedelta(days=7))
        prob_30d = get_probability(release, release - pd.Timedelta(days=30))
        prob_60d = get_probability(release, release - pd.Timedelta(days=60))
        
        actual = int((target_rates[target_rates['date'] == release]['target_high'].values[0] - target_rates[target_rates['date'] < release]['target_high'].values[-1]) * 100)
        row = [release]

        if prob_1d is not None:
            row.append(prob_1d[f'{actual}bp'])
        else:
            row.append(None)
        
        if prob_7d is not None:
            row.append(prob_7d[f'{actual}bp'])
        else:
            row.append(None)

        if prob_30d is not None:
            row.append(prob_30d[f'{actual}bp'])
        else:
            row.append(None)

        if prob_60d is not None:
            row.append(prob_60d[f'{actual}bp'])
        else:
            row.append(None)
        probs.append(row)
    except Exception as e:
        print(f"Error processing {release}: {e}")
        continue

print(probs)

plt.plot([x[0] for x in probs], [x[1] * 100 if x[1] is not None else None for x in probs], 'o')
plt.plot([x[0] for x in probs], [x[2] * 100 if x[2] is not None else None for x in probs], 'o')
plt.plot([x[0] for x in probs], [x[3] * 100 if x[3] is not None else None for x in probs], 'o')
plt.plot([x[0] for x in probs], [x[4] * 100 if x[4] is not None else None for x in probs], 'o')
plt.legend(['1 Day', '7 Days', '30 Days', '60 Days'])
plt.grid()

plt.title('Probability of Rate Change on FOMC Release Dates')
plt.xlabel('FOMC Release Date')
plt.ylabel('Probability of Rate Change (%)')
plt.xticks(rotation=45)
plt.show()
