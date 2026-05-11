#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ALL_FEATURES = [
    "volatility",
    "buy_volume",
    "sell_volume",
    "order_flow_imbalance",
    "total_volume",
    "momentum",
    "vol_adj_momentum",
    "prev_up_down",
    "rsi",
    "ma_10_dist",
    "rsi_centered",
    "mom_zscore",
    "macd",
    "bb_dist",
    "vol_change",
    "ofi_scaled",
    "mom_lag_1",
    "mom_lag_2",
]

TARGET_COL = "next_up_down"


def load_raw(filepath: str) -> pd.DataFrame:
    """
    Load the raw Coinbase BTC trade CSV and coerce column types.

    Expected columns: trade_id, time, price, size, side.
    trade_id is dropped - it carries no signal.

    Returns
    -------
    pd.DataFrame
        Columns: time (datetime64), price (float64), size (float64),
                 side (str). Sorted chronologically. No NaN in key cols.
    """
    df = pd.read_csv(filepath)
    df = df.drop(columns=["trade_id"], errors="ignore")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df = df.dropna(subset=["time", "price", "size"])
    df = df.sort_values("time").reset_index(drop=True)
    print(
        f"Loaded   {len(df):,} trades  |  " f"{df['time'].min()} -> {df['time'].max()}"
    )
    return df


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index using simple rolling mean for gains/losses.

    Matches the notebook's compute_rsi() function exactly.
    Values range 0-100. >70 = overbought, <30 = oversold.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _add_bucket_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Floor every trade timestamp to its 5-minute bucket, then compute
    group-level microstructure statistics and broadcast them back to
    every row in the group via groupby.transform.

    This matches the notebook's approach exactly: features are computed
    on the full trade DataFrame before deduplication so that every bucket
    has the correct aggregate stats regardless of how many trades it
    contains.

    Adds columns
    ------------
    volatility           std of trade prices within the bucket
    buy_volume           total BTC size of BUY-side trades
    sell_volume          total BTC size of SELL-side trades
    order_flow_imbalance buy_volume - sell_volume
    total_volume         buy_volume + sell_volume
    momentum             last price of this bucket - last price of previous
    vol_adj_momentum     momentum / volatility
    """
    df = df.copy()

    # Snap to 5-minute boundary
    df["time"] = df["time"].dt.floor("5min")

    # Within-bucket price volatility
    df["volatility"] = df.groupby("time")["price"].transform("std")

    # Buy / sell volume split using .where() then groupby sum
    df["buy_volume"] = (
        df["size"].where(df["side"] == "BUY", 0).groupby(df["time"]).transform("sum")
    )
    df["sell_volume"] = (
        df["size"].where(df["side"] == "SELL", 0).groupby(df["time"]).transform("sum")
    )

    df["order_flow_imbalance"] = df["buy_volume"] - df["sell_volume"]
    df["total_volume"] = df["buy_volume"] + df["sell_volume"]

    # Momentum: last traded price per bucket, differenced across buckets
    bucket_prices = df.groupby("time")["price"].last()
    bucket_momentum = bucket_prices.diff()
    df["momentum"] = df["time"].map(bucket_momentum)

    # Volatility-adjusted momentum
    df["vol_adj_momentum"] = df["momentum"] / df["volatility"].replace(0, np.nan)

    return df


def _add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the prediction target and one lagged label.

    Label construction (matches the notebook exactly):

        bucket_first_price[t]  - first traded price in bucket t.
                                  Polymarket's "price to beat" for window t.

        current_up_down[t]     - 1 if bucket_first_price[t+1] >
                                  bucket_first_price[t], else 0.
                                  "Did the market open higher next window?"

        next_up_down[t]        - current_up_down[t-1] shifted forward.
                                  This is the TARGET: what will the NEXT
                                  window's direction be?

        prev_up_down[t]        - current_up_down[t-1].
                                  The PREVIOUS window's direction, used
                                  as a lagged label feature.

    Both are mapped back to the full trade-level DataFrame so they
    survive the groupby-first deduplication in the next step.
    """
    df = df.copy()

    bucket_first_price = df.groupby("time")["price"].first()

    # Was the next bucket's open higher than this bucket's open?
    current_up_down = (bucket_first_price.shift(-1) > bucket_first_price).astype(int)

    # Target: direction of the window after next
    df[TARGET_COL] = df["time"].map(current_up_down.shift(-1))

    # Feature: direction of the previous window
    df["prev_up_down"] = df["time"].map(current_up_down.shift(1))

    return df


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to one row per 5-minute bucket by taking the first trade.

    After this step:
      - `price` is the opening trade price of each bucket
      - All bucket-level features already attached are preserved
      - The index is reset to 0…N-1

    NaN rows (from label shift edges) are dropped first so that the
    deduplication does not create buckets with missing targets.
    """
    df = df.dropna()
    deduped = df.groupby("time").first().reset_index()
    print(f"Deduped  {len(deduped):,} 5-minute buckets")
    return deduped


def _add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators that require a clean, equally-spaced price
    series. All are computed on the `price` column (opening trade of
    each 5-minute bucket) after deduplication.

    Indicators added
    ----------------
    ma_10_dist    (price - 10-period MA) / 10-period MA.
                  Positive = price above its recent average.

    rsi_centered  RSI - 50. Centres the indicator around 0 for linear
                  models that work best with zero-centred features.

    mom_zscore    (momentum - 20-period rolling mean) / 20-period rolling
                  std. How unusual is the current momentum vs recent
                  history?

    macd          12-period EMA - 26-period EMA. Positive = short-term
                  trend above long-term trend (bullish momentum).

    bb_dist       (price - 20-period mean) / 20-period std. How many
                  standard deviations is price from its Bollinger midline?

    vol_change    Percentage change in volatility vs previous bucket.
                  Rising volatility often precedes larger moves.

    ofi_scaled    order_flow_imbalance / total_volume. Scale-invariant
                  measure of net buying pressure, ranges -1 to +1.

    mom_lag_1     Momentum from 1 bucket ago.
    mom_lag_2     Momentum from 2 buckets ago.
    """
    df = df.copy()
    price = df["price"]

    # Distance from 10-period moving average (normalised)
    ma_10 = price.rolling(10).mean()
    df["ma_10_dist"] = (price - ma_10) / ma_10.replace(0, np.nan)

    # RSI centred at 0
    df["rsi_centered"] = _compute_rsi(price) - 50

    # Momentum z-score
    mom_mean = df["momentum"].rolling(window=20).mean()
    mom_std = df["momentum"].rolling(window=20).std()
    df["mom_zscore"] = (df["momentum"] - mom_mean) / mom_std.replace(0, np.nan)

    # MACD
    ema_fast = price.ewm(span=12, adjust=False).mean()
    ema_slow = price.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow

    # Bollinger Band distance
    roll_mean = price.rolling(window=20).mean()
    roll_std = price.rolling(window=20).std()
    df["bb_dist"] = (price - roll_mean) / roll_std.replace(0, np.nan)

    # Volatility rate of change
    df["vol_change"] = df["volatility"].pct_change()

    # Normalised order flow imbalance
    df["ofi_scaled"] = df["order_flow_imbalance"] / df["total_volume"].replace(
        0, np.nan
    )

    # Lagged momentum
    df["mom_lag_1"] = df["momentum"].shift(1)
    df["mom_lag_2"] = df["momentum"].shift(2)

    return df


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline on raw Coinbase trade data.

    Execution order matches the notebook exactly:
        1  _add_bucket_features   microstructure on all trades
        2  _add_labels            next/prev_up_down on all trades
        3  _deduplicate           one row per bucket, NaN edges dropped
        4  _add_technicals        RSI, MACD, BB, MA, OFI, lags
        5  dropna                 remove NaN rows from rolling windows
        6  rsi column             computed after the clean dropna pass
        7  final dropna + reset   remove RSI warmup NaNs

    Parameters
    ----------
    raw : pd.DataFrame
        Output of load_raw(). Columns: time, price, size, side.

    Returns
    -------
    pd.DataFrame
        One row per 5-minute bucket containing ALL_FEATURES + TARGET_COL.
        No NaN values. Index reset to 0…N.
    """
    df = _add_bucket_features(raw)
    df = _add_labels(df)
    df = _deduplicate(df)
    df = _add_technicals(df)

    # First dropna - clears rolling-window warmup rows
    df = df.dropna().reset_index(drop=True)

    # RSI is computed after the clean series is established
    # (matches the notebook's second RSI block)
    df["rsi"] = _compute_rsi(df["price"])

    # Final dropna - clears RSI warmup rows
    df = df.dropna().reset_index(drop=True)

    print(f"Final    {len(df):,} windows x {len(ALL_FEATURES)} features")
    print(f"Up rate  {df[TARGET_COL].mean():.1%}")
    return df


def build_features_from_path(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: load_raw -> build_features -> optionally write CSV.

    Parameters
    ----------
    input_path : str or Path
        Path to a Coinbase BTC trade CSV (output of bitcoin_datacollector.py).
    output_path : str or Path, optional
        If provided, the engineered feature DataFrame is written here as CSV.

    Returns
    -------
    pd.DataFrame
        One row per 5-minute bucket, ALL_FEATURES + TARGET_COL columns.
    """
    raw = load_raw(str(input_path))
    df = build_features(raw)
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved -> {out}")
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Engineer features from a Coinbase BTC trade CSV.",
    )
    parser.add_argument(
        "--data",
        required=True,
        metavar="FILE",
        help="Path to the raw Coinbase BTC trade CSV.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Where to write the engineered features (CSV). "
        "If omitted, the features are computed but not saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_features_from_path(args.data, args.output)
