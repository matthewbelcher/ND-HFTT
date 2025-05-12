import os
import pandas as pd
from datetime import datetime, timedelta

def fed_fund_futures_to_csv(file_name: str) -> None:
    with open(f"{file_name}", 'r') as file:
        lines = file.readlines()

    rows = []
    for row_index in range(len(lines) // 9):
        row = []
        for i in range(9):
            row.append(lines[row_index * 9 + i].strip())

        row[0] = datetime.strptime(row[0], "%m/%d/%Y")
        row[-1] = row[-1].replace(',', '')
        row[-2] = row[-2].replace(',', '')
        rows.append(row)
        

    num_days = (rows[0][0]  - rows[-1][0]).days

    market_days = {}
    for row in rows:
        market_days[row[0]] = row

    first_day = rows[0][0]
    for prev_days in range(num_days + 1):
        day = first_day - timedelta(days=prev_days)
        if day in market_days.keys():
            continue
        else:
            for days_before in range(1, 10):
                test_day = day - timedelta(days=days_before)
                if test_day in market_days.keys():
                    new_row = [x for x in market_days[test_day]]
                    new_row[0] = day
                    rows.append(new_row)
                    break

    rows = sorted(rows, key=lambda x: x[0])


    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "last", "change", "percent_change", "volume", "open_int"])
    df.drop(columns=['change', 'percent_change'], inplace=True)
    df.drop_duplicates(inplace=True)
    csv_file_name = f"{file_name.replace('.txt', '.csv')}"
    df.to_csv(csv_file_name, index=False)

fed_fund_futures_to_csv("zqf23.txt")