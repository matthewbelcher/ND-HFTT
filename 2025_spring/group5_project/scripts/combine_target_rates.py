from math import comb
import pandas as pd
import re
import os
from datetime import datetime

"""
Creates the target_rates data
"""


lower_df = pd.read_csv("data/DFEDTARL.csv", parse_dates=["observation_date"])
upper_df = pd.read_csv("data/DFEDTARU.csv", parse_dates=["observation_date"])
combined_df = pd.merge(
    lower_df, upper_df, on="observation_date", suffixes=("_lower", "_upper")
)
combined_df["observation_date"] = pd.to_datetime(
    combined_df["observation_date"]
) - pd.DateOffset(days=1)
combined_df.rename(
    columns={
        "observation_date": "date",
        "DFEDTARL": "target_low",
        "DFEDTARU": "target_high",
    },
    inplace=True,
)
combined_df.set_index("date", inplace=True)
print(combined_df.head())

# old_df = pd.read_csv('data/target_rates/target_rates_3.csv', parse_dates=['date'])
# old_df.set_index('date', inplace=True)
# print(old_df.head())

fomc_release_dates = sorted(
    [
        re.search(r"monetary(.*)a.htm", x).group(1)
        for x in os.listdir("data/fomc_statements")
    ]
)
fomc_release_dates = [
    datetime(year=int(date[0:4]), month=int(date[4:6]), day=int(date[6:8]))
    for date in fomc_release_dates
]

fomc_release_dates = list(
    filter(lambda x: x >= datetime(year=2008, month=12, day=16), fomc_release_dates)
)

combined_df = combined_df[combined_df.index.isin(fomc_release_dates)]
print(combined_df.head())

combined_df.to_csv("data/target_rates/target_rates_4.csv", index=True)
