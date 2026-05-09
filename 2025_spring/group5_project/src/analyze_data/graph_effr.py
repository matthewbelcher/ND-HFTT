import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates

df = pd.read_csv('data/effr/effr.csv')
df['date'] = pd.to_datetime(df['date'])
df.index = df['date']
df.plot(y=['rate', 'target_low', 'target_high'])
plt.ylabel('rate (%)')
plt.legend(['EFFR', 'target low', 'target high'])
plt.show()