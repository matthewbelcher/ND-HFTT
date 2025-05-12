import os
import pandas as pd

files = os.listdir("data/federal_funds/daily")

for file in files:
    print(f"Processing {file}...")
    if not file.endswith(".csv"):
        print(f"Skipping {file}, not a CSV file.")
        continue
    name = file.split('.')[0]
    if len(file.split('.')[0]) < 5:
        year_digit = name[-1]
        if int(year_digit) < 5:
            year = '2' + year_digit
        elif int(year_digit) > 5:
            year = '1' + year_digit
        else:
            year = '2' + year_digit
        name = name[:3] + year
        print(f"Renaming {file} to {name}")

    with open(f"data/federal_funds/daily/{file}", "r") as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines[2:]]

    data = []
    for line in lines:
        values = line.split(",", maxsplit=4)
        if len(values) == 4:
            date, bid, ask, price = values
        elif len(values) == 5:
            date, bid, ask, price, _ = values
        if price == "#N/A" or date == "#N/A":
            continue
        if bid == "#N/A":
            bid = None
        if ask == "#N/A":
            ask = None
        data.append((date, bid, ask, price))

    df = pd.DataFrame(data, columns=["date", "bid", "ask", "price"])
    df.to_csv(f"data/federal_funds/formatted_daily/{name}.csv", index=False)


