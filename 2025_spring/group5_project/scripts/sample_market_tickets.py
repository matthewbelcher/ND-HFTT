import os
from datetime import datetime
from textwrap import fill
import pandas as pd


"""
This script samples market tick data for VOO, SPY, and VFH around FOMC release dates.
Creates the clean_sampled directory if it doesn't exist.
"""

files = os.listdir("data/market_ticks/40_min")

for file in files:
    if file[2] != "_":
        continue
    df = pd.read_csv(f"data/market_ticks/40_min/{file}")
    df["timestamp"] = pd.to_datetime(df["DATE"] + " " + df["TIME_M"])
    date = pd.to_datetime(df["DATE"].iloc[0])

    for symbol in ["VOO", "SPY", "VFH"]:
        df_symbol = df[df["SYM_ROOT"] == symbol]
        df_symbol = df_symbol.reset_index(drop=True)

        df_symbol["bid_diff"] = (
            df_symbol["BID"].shift(1, fill_value=df_symbol["BID"].iloc[0])
            - df_symbol["BID"]
        )
        df_symbol["ask_diff"] = (
            df_symbol["ASK"].shift(1, fill_value=df_symbol["ASK"].iloc[0])
            - df_symbol["ASK"]
        )

        print(df_symbol["bid_diff"].min(), df_symbol["ask_diff"].min())
        print(df_symbol["bid_diff"].max(), df_symbol["ask_diff"].max())
        print(df_symbol["bid_diff"].mean(), df_symbol["ask_diff"].mean())

        df_symbol = df_symbol[
            (abs(df_symbol["bid_diff"]) < 1) & (abs(df_symbol["ask_diff"]) < 1)
        ]
        print(df_symbol)

        bid_median = df_symbol["BID"].median()
        ask_median = df_symbol["ASK"].median()

        df_symbol = df_symbol[
            (abs(df_symbol["BID"] - bid_median) < 0.25 * bid_median)
            & (abs(df_symbol["ASK"] - ask_median) < 0.25 * ask_median)
        ]

        rows = []
        price_at_release = df_symbol[
            df_symbol["timestamp"]
            < datetime(date.year, date.month, date.day, 14, 0, 0, 0)
        ].iloc[-1]
        price_at_one_ms = df_symbol[
            df_symbol["timestamp"]
            < datetime(date.year, date.month, date.day, 14, 0, 0, 100)
        ].iloc[-1]
        price_at_one_s = df_symbol[
            df_symbol["timestamp"]
            < datetime(date.year, date.month, date.day, 14, 0, 1, 0)
        ].iloc[-1]
        price_at_one_m = df_symbol[
            df_symbol["timestamp"]
            < datetime(date.year, date.month, date.day, 14, 1, 0, 0)
        ].iloc[-1]
        price_at_five_m = df_symbol[
            df_symbol["timestamp"]
            < datetime(date.year, date.month, date.day, 14, 5, 0, 0)
        ].iloc[-1]
        price_at_ten_m = df_symbol[
            df_symbol["timestamp"]
            < datetime(date.year, date.month, date.day, 14, 10, 0, 0)
        ].iloc[-1]
        price_at_thirty_m = df_symbol[
            df_symbol["timestamp"]
            < datetime(date.year, date.month, date.day, 14, 30, 0, 0)
        ].iloc[-1]

        rows = [
            price_at_release,
            price_at_one_ms,
            price_at_one_s,
            price_at_one_m,
            price_at_five_m,
            price_at_ten_m,
            price_at_thirty_m,
        ]

        new_df = pd.DataFrame(rows)
        print(new_df)

        # Ensure the directory exists before saving the file
        output_dir = f"data/market_ticks/clean_sampled/{symbol}"
        os.makedirs(output_dir, exist_ok=True)

        new_df.to_csv(f"{output_dir}/{file}", index=False)
