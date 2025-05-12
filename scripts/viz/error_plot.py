import os
import pandas as pd
import re
from datetime import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

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

fomc_release_dates = list(filter(lambda x: x >= datetime(year=2009, month=3, day=18) and x <= datetime(year=2025, month=1, day=1), fomc_release_dates))

avg_errors = []
all_errors = []
for days_prior in tqdm(range(1,60)):
    errors = []
    for release in tqdm(fomc_release_dates, leave=False):
        actual = int((target_rates[target_rates['date'] == release]['target_high'].values[0] - target_rates[target_rates['date'] < release]['target_high'].values[-1]) * 100)
        probs = get_probability(release, release - pd.Timedelta(days=days_prior))

        if probs is not None:
            weighted_avg = -200 * probs['-200bp'] +  -175 * probs['-175bp'] + \
                -150 * probs['-150bp'] + -125 * probs['-125bp'] + \
                -100 * probs['-100bp'] + -75 * probs['-75bp'] + \
                -50 * probs['-50bp'] + -25 * probs['-25bp'] + \
                0 * probs['0bp'] + 25 * probs['25bp'] + \
                50 * probs['50bp'] + 75 * probs['75bp'] + \
                100 * probs['100bp'] + 125 * probs['125bp'] + \
                150 * probs['150bp'] + 175 * probs['175bp'] + \
                200 * probs['200bp']

            errors.append(np.abs(probs[f'{actual}bp']) )
        
        if probs is not None and days_prior == 1:
            max_prob = -1
            max_prob_change = None
            for change in range(-200, 201, 25):
                prob = probs[f'{change}bp']
                if prob > max_prob:
                    max_prob = prob
                    max_prob_change = change

            all_errors.append((max_prob, max_prob_change == actual))
    print(errors)
    avg_errors.append(np.percentile(errors, 10) if errors else 0)

plt.plot(range(1, 60), avg_errors)
plt.xlabel('Days Prior to FOMC Release')
plt.ylabel('Probability of Actual Change (%)')
plt.title('10th Percentile Probability of Actual Change by Days Prior to FOMC Release')
plt.xticks(range(1, 60, 5))
plt.grid()
plt.show()

bucket_accuracy = []
for i in range(0, 101, 5):
    count = 0
    num_correct = 0
    for prob, is_correct in all_errors:
        if prob >= i / 100 and prob < (i + 5) / 100:
            count += 1
            if is_correct:
                num_correct += 1
    
    bucket_accuracy.append((i, num_correct / count if count > 0 else None, count))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter([x[0] for x in bucket_accuracy], [x[1] for x in bucket_accuracy], [x[2] for x in bucket_accuracy], 'o')
ax.set_xlabel('Probability Bucket (%)')
ax.set_ylabel('Accuracy (%)')
plt.title('Accuracy of Rate Change Predictions by Probability Bucket')
plt.xticks(range(0, 101, 5))
plt.grid()
plt.show()
print(all_errors)