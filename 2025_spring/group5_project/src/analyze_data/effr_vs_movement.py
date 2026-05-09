import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# read in all available market data
def load_market_data(base_dir="./data/market_ticks"):
    all_days = {}
    for filename in os.listdir(base_dir):
        if filename.endswith(".csv"):
            date_str = filename.split(".")[0]
            date = pd.to_datetime(date_str, format="%m_%d_%y")
            filepath = os.path.join(base_dir, filename)
            try:
                data = pd.read_csv(filepath)
                all_days[date] = data
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    return all_days


def get_means(path, time_from_release=10):
    if time_from_release < 0 or time_from_release > 10:
        raise ValueError("time_from_release must be between 0 and 10 seconds")
    means_spy_df = pd.DataFrame(
        columns=[
            "date",
            "pre-release bid mean",
            "pre-release ask mean",
            "post-release bid mean",
            "post-release ask mean",
            "pre-release weighted mid",
            "post-release weighted mid",
        ]
    )

    files = os.listdir(path)
    for file in files:
        df = pd.read_csv(path + file)
        df["TIME"] = pd.to_datetime(df["TIME_M"], format="%H:%M:%S.%f")
        df["TIME_IN_SEC"] = (
            df["TIME"].dt.nanosecond * 1e-9
            + df["TIME"].dt.microsecond * 1e-6
            + df["TIME"].dt.second
            + df["TIME"].dt.minute * 60
            + df["TIME"].dt.hour * 3600
        )
        df["TIME_FROM_RELEASE"] = df["TIME_IN_SEC"] - 14 * 3600.0000000
        df = df[
            (df["TIME_FROM_RELEASE"] >= -time_from_release)
            & (df["TIME_FROM_RELEASE"] <= time_from_release)
        ]
        # output_filepath = f"./{file.split('.')[0]}_filtered.csv"
        # os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        # df.to_csv(output_filepath, index=False)

        df["ASK_CLEAN"] = df["ASK"][df["ASK"] > 5]
        df["BID_CLEAN"] = df["BID"][df["BID"] > 5]
        pre_weighted_mid_mean = (
            (df["ASK_CLEAN"] * df["ASKSIZ"] + df["BID_CLEAN"] * df["BIDSIZ"])
            / (df["ASKSIZ"] + df["BIDSIZ"])
        )[(df["TIME_FROM_RELEASE"] < 0) & (df["SYM_ROOT"] == "SPY")].mean()
        pre_ask_mean = df["ASK_CLEAN"][
            (df["TIME_FROM_RELEASE"] < 0) & (df["SYM_ROOT"] == "SPY")
        ].mean()
        pre_bid_mean = df["BID_CLEAN"][
            (df["TIME_FROM_RELEASE"] < 0) & (df["SYM_ROOT"] == "SPY")
        ].mean()

        post_weighted_mid_mean = (
            (df["ASK_CLEAN"] * df["ASKSIZ"] + df["BID_CLEAN"] * df["BIDSIZ"])
            / (df["ASKSIZ"] + df["BIDSIZ"])
        )[(df["TIME_FROM_RELEASE"] > 0) & (df["SYM_ROOT"] == "SPY")].mean()
        post_ask_mean = df["ASK_CLEAN"][
            (df["TIME_FROM_RELEASE"] > 0) & (df["SYM_ROOT"] == "SPY")
        ].mean()
        post_bid_mean = df["BID_CLEAN"][
            (df["TIME_FROM_RELEASE"] > 0) & (df["SYM_ROOT"] == "SPY")
        ].mean()
        date = pd.to_datetime(file.split(".")[0], format="%m_%d_%y")
        means_spy_df.loc[len(means_spy_df)] = [
            date,
            pre_bid_mean,
            pre_ask_mean,
            post_bid_mean,
            post_ask_mean,
            pre_weighted_mid_mean,
            post_weighted_mid_mean,
        ]
    return means_spy_df


def get_effective_fed_funds_rate(path="./data/effr/effr.csv"):
    eff_fed_funds = pd.read_csv(path)
    eff_fed_funds["date"] = pd.to_datetime(eff_fed_funds["date"], format="%Y-%m-%d")
    eff_fed_funds["rate"] = eff_fed_funds["rate"].astype(float)
    return eff_fed_funds


if __name__ == "__main__":
    means = get_means("./data/market_ticks/", 10)
    # print(
    #     means["pre-release weighted mid"],
    #     means["pre-release ask mean"],
    #     means["pre-release bid mean"],
    #     means["post-release ask mean"],
    #     means["post-release bid mean"],
    # )
    # exit()
    bid_movement = means["post-release bid mean"] - means["pre-release bid mean"]
    ask_movement = means["post-release ask mean"] - means["pre-release ask mean"]
    mid_movement = (
        means["post-release ask mean"] + means["post-release bid mean"]
    ) / 2 - (means["pre-release ask mean"] + means["pre-release bid mean"]) / 2
    weigthed_mid_movement = (
        means["post-release weighted mid"] - means["pre-release weighted mid"]
    )
    er_df = get_effective_fed_funds_rate()
    delta_rate = []
    for date in means["date"]:
        if date in er_df["date"].values:
            delta_rate.append(
                er_df["rate"][er_df["date"] == date].values[0]
                - er_df["rate"][er_df["date"] == date + pd.Timedelta(days=1)].values[0]
            )
        else:
            delta_rate.append(np.nan)
    movement_df = pd.DataFrame(
        {
            "date": means["date"],
            "bid_movement": bid_movement,
            "ask_movement": ask_movement,
            "mid_movement": mid_movement,
            "weighted_mid_movement": weigthed_mid_movement,
            "rate_change": delta_rate,
        }
    )
    movement_df = movement_df.sort_values("date")
    # print(movement_df)
    plt.figure(figsize=(10, 6))
    # plt.plot(
    #     movement_df["date"],
    #     movement_df["bid_movement"],
    #     label="Bid Movement",
    #     color="blue",
    # )
    # plt.plot(
    #     movement_df["date"],
    #     movement_df["ask_movement"],
    #     label="Ask Movement",
    #     color="green",
    # )
    plt.plot(
        movement_df["date"],
        movement_df["mid_movement"],
        label="Unweighted Mid Movement +/- 10s",
        color="orange",
    )
    plt.plot(
        movement_df["date"],
        movement_df["weighted_mid_movement"],
        label="Weighted Mid Movement +/- 10s",
        color="brown",
    )
    plt.plot(
        movement_df["date"],
        movement_df["rate_change"],
        label="Effective Rate Change",
        color="red",
    )

    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Market Movements and Rate Changes Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
