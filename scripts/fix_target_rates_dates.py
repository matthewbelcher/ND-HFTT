import csv 
import os
import re

with open("data/target_rates/target_rates.csv", "r") as fh:
    reader = csv.reader(fh)
    rows = list(reader)

dates = [re.findall(r'monetary(20[0-9]*)a.htm', x)[0] for x in sorted(os.listdir("data/fomc_statements"))]
dates = dates[len(dates) - len(rows):]

for i in range(1, len(rows)):
    rows[i][0] = dates[i]

with open("data/target_rates/target_rates_3.csv", "w") as fh:
    writer = csv.writer(fh)
    writer.writerows(rows)