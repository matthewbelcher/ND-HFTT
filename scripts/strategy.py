import matplotlib.pyplot as plt
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression


"""
1) surprise_regression_signal: Calculates the "surprise" between market-expected Fed rate changes and actual changes
   (positive = Fed more dovish than expected, negative = more hawkish)
2) order_book_imbalance: Measures shifts in order book dynamics (buy vs sell pressure) in the minute before/after 
   FOMC announcements at 14:00
xxx         implied_probability_shift: Measures how Fed Funds futures pricing (implied rates) change after announcements
3) cross_asset_confirmation: Verifies if multiple related assets (SPY, VOO, VFH) move in the same direction
   after announcements

Should be combined with preditive weights: weights = dict(zip(features, reg.coef_))
"""


def get_fomc_release_dates(
    statements_dir="data/fomc_statements", min_date=datetime(2022, 4, 1)
):
    dates = sorted(
        [
            re.search(r"monetary(.*)a.htm", x).group(1)
            for x in os.listdir(statements_dir)
        ]
    )
    dates = [
        datetime(year=int(date[0:4]), month=int(date[4:6]), day=int(date[6:8]))
        for date in dates
    ]
    dates = list(filter(lambda x: x >= min_date, dates))
    return sorted(dates, reverse=True)[1:]


def surprise_regression_signal(
    target_rates_path="data/target_rates/target_rates_4.csv",
    prob_dir="data/federal_funds/historical_probabilities_3",
    min_date=datetime(2009, 1, 1),
    # min_date=datetime(2022, 1, 1),
):
    # Surprise = difference between actual and expected (WMID) rate change
    target_df = pd.read_csv(target_rates_path, parse_dates=["date"])
    # fomc_dates = get_fomc_release_dates(min_date=min_date)
    fomc_dates = []
    for date in target_df["date"]:
        if date >= min_date and os.path.exists(
            f"{prob_dir}/{date.month:02}_{date.day:02}_{date.year}.csv"
        ):
            fomc_dates.append(date)
    dates = [date.to_pydatetime() for date in fomc_dates]
    dates = list(filter(lambda x: x >= min_date, dates))
    fomc_dates = sorted(dates, reverse=True)[1:]
    signals = []
    for date in fomc_dates:
        # print(f"{prob_dir}/{date.month:02}_{date.day:02}_{date.year}.csv")
        try:
            prob_df = pd.read_csv(
                f"{prob_dir}/{date.month:02}_{date.day:02}_{date.year}.csv",
                parse_dates=["Date"],
            )
            old_rate = target_df[target_df["date"] == date]["target_high"].values[0]
            next_date_mask = target_df["date"] > date
            if not next_date_mask.any():
                # No future dates available, skip this date
                raise Exception(f"No future target rate available for {date}")

            next_date = target_df.loc[next_date_mask, "date"].min()
            new_rate = target_df[target_df["date"] == next_date]["target_high"].values[
                0
            ]
            change = new_rate - old_rate
            change_bp = int(change * 100)
            predicted = 0
            for col in prob_df.columns:
                if col == "Date":
                    continue
                change_num = int(col.replace("bp", ""))
                predicted += (
                    change_num
                    * prob_df[prob_df["Date"] == date - timedelta(days=1)][col].values[
                        0
                    ]
                )
            surprise = predicted - change_bp
            signals.append(
                {
                    "date": date,
                    "actual_change_bp": change_bp,
                    "predicted": predicted,
                    "surprise": surprise,
                }
            )
        except Exception as e:
            print(f"Error processing date {date}: {e}")
            continue
    return pd.DataFrame(signals)


def order_book_imbalance(
    market_ticks_dir="data/market_ticks/40_min",
    symbol="SPY",
    # min_date=datetime(2022, 4, 1),
    min_date=datetime(2009, 1, 1),
):
    fomc_dates = get_fomc_release_dates(min_date=min_date)

    imbalances = []
    for date in fomc_dates:
        try:
            df = pd.read_csv(
                f"{market_ticks_dir}/{date.month:02}_{date.day:02}_{str(date.year)[2:]}.csv"
            )
            df = df[df["SYM_ROOT"] == symbol]
            df["timestamp"] = pd.to_datetime(df["DATE"] + " " + df["TIME_M"])
            # 1 minute before and after 14:00:00
            pre = df[
                (
                    df["timestamp"]
                    >= datetime(date.year, date.month, date.day, 13, 59, 0)
                )
                & (
                    df["timestamp"]
                    < datetime(date.year, date.month, date.day, 14, 0, 0)
                )
            ]
            post = df[
                (df["timestamp"] >= datetime(date.year, date.month, date.day, 14, 0, 0))
                & (
                    df["timestamp"]
                    < datetime(date.year, date.month, date.day, 14, 1, 0)
                )
            ]

            def imbalance(sub):
                buy = (sub["BID"] * sub["BIDSIZ"]).sum()
                sell = (sub["ASK"] * sub["ASKSIZ"]).sum()
                return (buy - sell) / (buy + sell) if (buy + sell) != 0 else np.nan

            pre_imb = imbalance(pre)
            post_imb = imbalance(post)
            imbalances.append(
                {
                    "date": date,
                    "pre_imbalance": pre_imb,
                    "post_imbalance": post_imb,
                    "imbalance_shift": (
                        post_imb - pre_imb
                        if pre_imb is not np.nan and post_imb is not np.nan
                        else np.nan
                    ),
                }
            )
        except Exception as e:
            continue
    return pd.DataFrame(imbalances)


def implied_probability_shift(
    futures_dir="data/federal_funds/...",
    get_futures_code=None,
    fomc_dates=None,
):
    # get_futures_code should be a function(date: datetime) -> str
    # If not provided, use the one from implied_rate_change_odds.py
    if get_futures_code is None:

        def get_futures_code(date):
            months = {
                1: "F",
                2: "G",
                3: "H",
                4: "J",
                5: "K",
                6: "M",
                7: "N",
                8: "Q",
                9: "U",
                10: "V",
                11: "X",
                12: "Z",
            }
            return months[date.month].lower() + str(date.year)[-2:]

    if fomc_dates is None:
        fomc_dates = get_fomc_release_dates()
    shifts = []
    for date in fomc_dates:
        try:
            code = get_futures_code(date)
            path = f"{futures_dir}/ff{code}.csv"
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, parse_dates=["date"])
            # Find price at 13:59 and 14:01
            pre = df[df["date"] <= datetime(date.year, date.month, date.day, 13, 59, 0)]
            post = df[df["date"] >= datetime(date.year, date.month, date.day, 14, 1, 0)]
            if pre.empty or post.empty:
                continue
            pre_price = pre.iloc[-1]["price"]
            post_price = post.iloc[0]["price"]
            pre_rate = 100 - pre_price
            post_rate = 100 - post_price
            shifts.append(
                {
                    "date": date,
                    "pre_implied_rate": pre_rate,
                    "post_implied_rate": post_rate,
                    "implied_rate_shift": post_rate - pre_rate,
                }
            )
        except Exception as e:
            continue
    return pd.DataFrame(shifts)


def cross_asset_confirmation(
    market_ticks_dir="data/market_ticks/40_min",
    symbols=["SPY", "VOO", "VFH"],
    # min_date=datetime(2022, 4, 1),
    min_date=datetime(2009, 1, 1),
):
    fomc_dates = get_fomc_release_dates(min_date=min_date)
    confirmations = []
    for date in fomc_dates:
        try:
            df = pd.read_csv(
                f"{market_ticks_dir}/{date.month:02}_{date.day:02}_{str(date.year)[2:]}.csv"
            )
            df["timestamp"] = pd.to_datetime(df["DATE"] + " " + df["TIME_M"])
            asset_moves = {}
            for symbol in symbols:
                sub = df[df["SYM_ROOT"] == symbol]
                if sub.empty:
                    asset_moves[symbol] = np.nan
                    continue
                # Use WMID at 13:59:59 and 14:01:00
                pre = sub[
                    sub["timestamp"]
                    <= datetime(date.year, date.month, date.day, 13, 59, 59)
                ]
                post = sub[
                    sub["timestamp"]
                    >= datetime(date.year, date.month, date.day, 14, 1, 0)
                ]
                if pre.empty or post.empty:
                    asset_moves[symbol] = np.nan
                    continue
                pre_row = pre.iloc[-1]
                post_row = post.iloc[0]
                pre_wmid = (
                    pre_row["BID"] * pre_row["BIDSIZ"]
                    + pre_row["ASK"] * pre_row["ASKSIZ"]
                ) / (pre_row["BIDSIZ"] + pre_row["ASKSIZ"])
                post_wmid = (
                    post_row["BID"] * post_row["BIDSIZ"]
                    + post_row["ASK"] * post_row["ASKSIZ"]
                ) / (post_row["BIDSIZ"] + post_row["ASKSIZ"])
                asset_moves[symbol] = post_wmid - pre_wmid
            # Confirm if all move in same direction (sign)
            moves = [v for v in asset_moves.values() if not np.isnan(v)]
            confirmation = np.sign(moves).sum() == len(moves) or np.sign(
                moves
            ).sum() == -len(moves)
            confirmations.append(
                {"date": date, **asset_moves, "cross_asset_confirmation": confirmation}
            )
        except Exception as e:
            continue
    return pd.DataFrame(confirmations)


def test_all_signals():

    # surprise_df = surprise_regression_signal()
    # imbalance_df = order_book_imbalance()
    # # probshift_df = implied_probability_shift()
    # crossasset_df = cross_asset_confirmation()
    # os.makedirs("output", exist_ok=True)
    # surprise_df.to_csv("output/surprise_signals.csv", index=False)
    # imbalance_df.to_csv("output/order_book_imbalance.csv", index=False)
    # # probshift_df.to_csv("output/implied_probability_shift.csv", index=False)
    # crossasset_df.to_csv("output/cross_asset_confirmation.csv", index=False)
    # return

    # Load DataFrames from the output directory
    surprise_df = pd.read_csv("output/surprise_signals.csv", parse_dates=["date"])
    imbalance_df = pd.read_csv("output/order_book_imbalance.csv", parse_dates=["date"])
    # probshift_df = pd.read_csv(
    #     "output/implied_probability_shift.csv", parse_dates=["date"]
    # )
    crossasset_df = pd.read_csv(
        "output/cross_asset_confirmation.csv", parse_dates=["date"]
    )

    # --- Merge on date ---
    merged = surprise_df.merge(imbalance_df, on="date", how="inner")
    # merged = merged.merge(probshift_df, on="date", how="inner")
    merged = merged.merge(crossasset_df, on="date", how="inner")

    # --- Calculate market move (target) ---
    # Example: Use SPY WMID move as target
    merged["spy_move"] = merged["SPY"]

    # --- Prepare features and target ---
    features = [
        "surprise",  # from surprise_regression_signal
        "imbalance_shift",  # from order_book_imbalance
        # "implied_rate_shift",  # from implied_probability_shift
        "cross_asset_confirmation",  # from cross_asset_confirmation (bool/int)
    ]
    X = merged[features].astype(float)
    y = merged["spy_move"].astype(float)

    # --- Fit regression ---
    reg = LinearRegression()
    reg.fit(X, y)
    weights = dict(zip(features, reg.coef_))

    # --- Predict and evaluate ---
    merged["predicted"] = reg.predict(X)
    merged["signal"] = np.sign(merged["predicted"])
    merged["actual_direction"] = np.sign(merged["spy_move"])
    merged["correct"] = merged["signal"] == merged["actual_direction"]
    merged["pnl"] = merged["signal"] * merged["spy_move"]

    # --- Output weights ---
    print("Suggested weights for linear combination of signals:")
    for k, v in weights.items():
        print(f"{k}: {v:.4f}")

    # --- Output summary stats ---
    print(f"Accuracy: {merged['correct'].mean():.2%}")
    print(f"Total PnL: {merged['pnl'].sum():.4f}")
    print(f"Average PnL per trade: {merged['pnl'].mean():.4f}")

    # --- Plot cumulative PnL ---
    plt.figure(figsize=(10, 5))
    plt.plot(merged["date"], merged["pnl"].cumsum(), marker="o")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL (proxy)")
    plt.title("Cumulative PnL using Combined Signal")
    plt.grid()
    plt.show()

    # --- Plot feature importances ---
    plt.figure(figsize=(8, 4))
    plt.bar(weights.keys(), weights.values())
    plt.title("Signal Weights (Linear Regression Coefficients)")
    plt.ylabel("Weight")
    plt.show()

    # --- Scatter actual vs predicted ---
    plt.figure(figsize=(6, 6))
    plt.scatter(merged["predicted"], merged["spy_move"])
    plt.xlabel("Predicted Move")
    plt.ylabel("Actual SPY Move")
    plt.title("Predicted vs Actual SPY Move")
    plt.grid()
    plt.show()

    # --- Return merged DataFrame for further analysis ---
    return merged


# Example usage:
results_df = test_all_signals()
# df = surprise_regression_signal()
print(results_df.head())
os.makedirs("output", exist_ok=True)
results_df.to_csv("output/signals.csv", index=False)
