from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    ARRIVAL_RATE_ROLLING_MINUTES,
    FINAL_ACCELERATION_LOOKBACK_MINUTES,
    OFI_IMPACT_WINDOWS_SECONDS,
    PRE_WINDOW_MINUTES,
    RETURN_HORIZONS_MINUTES,
    TARGET_MOVE_PCTS,
)
from data_loader import load_trade_days, load_trade_file


@dataclass(frozen=True)
class WindowSlices:
    pre: pd.DataFrame
    impact: pd.DataFrame
    post: pd.DataFrame


def take_side_multiplier(is_buyer_maker: pd.Series) -> pd.Series:
    return np.where(is_buyer_maker, -1.0, 1.0)


def compute_ofi(df: pd.DataFrame) -> float:
    if df.empty:
        return math.nan
    signed_qty = df["qty"] * take_side_multiplier(df["is_buyer_maker"])
    total_qty = df["qty"].sum()
    if total_qty == 0:
        return 0.0
    return float(signed_qty.sum() / total_qty)


def nearest_price(df: pd.DataFrame, target_ts: pd.Timestamp) -> float:
    index = (df["time_et_dt"] - target_ts).abs().idxmin()
    return float(df.loc[index, "price"])


def get_windows(trades: pd.DataFrame, release_time_et: pd.Timestamp) -> WindowSlices:
    pre_start = release_time_et - pd.Timedelta(minutes=PRE_WINDOW_MINUTES)
    post_end = release_time_et + pd.Timedelta(minutes=120)
    impact_end = release_time_et + pd.Timedelta(minutes=5)

    pre = trades[(trades["time_et_dt"] >= pre_start) & (trades["time_et_dt"] < release_time_et)].copy()
    impact = trades[(trades["time_et_dt"] >= release_time_et) & (trades["time_et_dt"] <= impact_end)].copy()
    post = trades[(trades["time_et_dt"] > impact_end) & (trades["time_et_dt"] <= post_end)].copy()
    return WindowSlices(pre=pre, impact=impact, post=post)


def compute_threshold_hit_seconds(relative_returns: pd.Series, threshold: float) -> float:
    hit_up = relative_returns[relative_returns >= threshold]
    hit_down = relative_returns[relative_returns <= -threshold]
    candidates = []
    if not hit_up.empty:
        candidates.append(float(hit_up.index[0]))
    if not hit_down.empty:
        candidates.append(float(hit_down.index[0]))
    if not candidates:
        return math.nan
    return min(candidates)


def compute_day_features(trades: pd.DataFrame, release_time_et: pd.Timestamp) -> dict[str, float | str]:
    windows = get_windows(trades, release_time_et)
    pre = windows.pre
    impact = windows.impact

    if pre.empty or impact.empty:
        raise ValueError("Expected non-empty pre and impact windows.")

    p0 = nearest_price(trades, release_time_et)
    p_minus_120 = nearest_price(trades, release_time_et - pd.Timedelta(minutes=120))
    p_minus_1 = nearest_price(trades, release_time_et - pd.Timedelta(minutes=1))

    pre_release_drift = (p_minus_1 / p_minus_120) - 1.0 if p_minus_120 else math.nan

    pre_5m = pre.set_index("time_et_dt").resample("5min").agg(qty=("qty", "sum"), signed_qty=("is_buyer_maker", "size"))
    pre_5m["signed_qty"] = (
        pre.set_index("time_et_dt")
        .assign(signed_qty=pre["qty"] * take_side_multiplier(pre["is_buyer_maker"]))
        .resample("5min")["signed_qty"]
        .sum()
    )
    pre_5m["ofi"] = np.where(pre_5m["qty"] > 0, pre_5m["signed_qty"] / pre_5m["qty"], np.nan)
    pre_ofi_mean_5m = float(pre_5m["ofi"].mean())

    arrivals_1m = pre.set_index("time_et_dt").resample("1min").size().rename("trades_per_min")
    trades_per_second = arrivals_1m / 60.0
    arrival_rate_mean = float(trades_per_second.mean())
    final_10 = trades_per_second[trades_per_second.index >= (release_time_et - pd.Timedelta(minutes=FINAL_ACCELERATION_LOOKBACK_MINUTES))]
    arrival_rate_final10_mean = float(final_10.mean()) if not final_10.empty else math.nan
    arrival_rate_accel = arrival_rate_final10_mean - arrival_rate_mean

    first_trade = impact.iloc[0]
    first_trade_direction = "buy" if not bool(first_trade["is_buyer_maker"]) else "sell"

    impact_seconds = ((impact["time_et_dt"] - release_time_et).dt.total_seconds()).clip(lower=0)
    impact = impact.assign(elapsed_s=impact_seconds.values, rel_ret=(impact["price"] / p0) - 1.0)

    impact_ofi = {}
    for window_s in OFI_IMPACT_WINDOWS_SECONDS:
        w = impact[impact["elapsed_s"] <= window_s]
        impact_ofi[f"ofi_0_{window_s}s"] = compute_ofi(w)

    max_upside_5m = float(impact["rel_ret"].max())
    max_downside_5m = float(impact["rel_ret"].min())

    time_indexed = impact.set_index("elapsed_s")["rel_ret"].sort_index()
    threshold_times = {
        f"time_to_move_{int(th * 10000)}bp_s": compute_threshold_hit_seconds(time_indexed, th)
        for th in TARGET_MOVE_PCTS
    }

    r30 = float(nearest_price(trades, release_time_et + pd.Timedelta(seconds=30)) / p0 - 1.0)
    r5 = float(nearest_price(trades, release_time_et + pd.Timedelta(minutes=5)) / p0 - 1.0)
    move_ratio_30s_to_5m = r30 / r5 if r5 != 0 else math.nan

    returns = {}
    for horizon in RETURN_HORIZONS_MINUTES:
        px = nearest_price(trades, release_time_et + pd.Timedelta(minutes=horizon))
        returns[f"ret_{horizon}m"] = float(px / p0 - 1.0)

    continuation_5m_to_120m = int(np.sign(returns["ret_5m"]) == np.sign(returns["ret_120m"]))

    post_with_lr = trades.copy()
    post_with_lr["log_ret"] = np.log(post_with_lr["price"]).diff()
    rv_5m = (
        post_with_lr.set_index("time_et_dt")["log_ret"].rolling("5min").std().dropna()
    )
    realized_vol_5m_mean = float(rv_5m.mean()) if not rv_5m.empty else math.nan

    pre_vol_per_min = float(pre.set_index("time_et_dt").resample("1min")["qty"].sum().mean())
    vol_30m = float(
        trades[
            (trades["time_et_dt"] >= release_time_et)
            & (trades["time_et_dt"] <= release_time_et + pd.Timedelta(minutes=30))
        ]["qty"].sum()
    )
    baseline_30m = pre_vol_per_min * 30.0
    volume_elevated_30m_ratio = vol_30m / baseline_30m if baseline_30m > 0 else math.nan

    high_120 = nearest_price(trades, release_time_et + pd.Timedelta(minutes=120))
    low_120 = float(
        trades[
            (trades["time_et_dt"] >= release_time_et)
            & (trades["time_et_dt"] <= release_time_et + pd.Timedelta(minutes=120))
        ]["price"].min()
    )
    high_120 = float(
        trades[
            (trades["time_et_dt"] >= release_time_et)
            & (trades["time_et_dt"] <= release_time_et + pd.Timedelta(minutes=120))
        ]["price"].max()
    )

    return {
        "price_t0": p0,
        "pre_ofi_mean_5m": pre_ofi_mean_5m,
        "arrival_rate_mean_tps": arrival_rate_mean,
        "arrival_rate_final10m_tps": arrival_rate_final10_mean,
        "arrival_rate_acceleration_tps": arrival_rate_accel,
        "pre_release_drift": pre_release_drift,
        "first_trade_direction": first_trade_direction,
        **impact_ofi,
        "max_upside_5m": max_upside_5m,
        "max_downside_5m": max_downside_5m,
        **threshold_times,
        "ret_30s": r30,
        "ret_5m": r5,
        "ratio_30s_to_5m": move_ratio_30s_to_5m,
        **returns,
        "continuation_5m_to_120m": continuation_5m_to_120m,
        "realized_vol_5m_mean": realized_vol_5m_mean,
        "volume_elevated_30m_ratio": volume_elevated_30m_ratio,
        "range_0_120m": (high_120 - low_120) / p0 if p0 else math.nan,
    }


def build_feature_table() -> pd.DataFrame:
    trade_days = load_trade_days()
    rows: list[dict[str, object]] = []

    for _, row in trade_days.iterrows():
        trades = load_trade_file(row["trade_file"])
        feature_row = compute_day_features(trades=trades, release_time_et=row["release_time_et"])
        rows.append(
            {
                "reference_month": row["reference_month"],
                "release_date_key": row["release_date_key"],
                "release_time_et": row["release_time_et"],
                "actual_cpi": row["actual_cpi"],
                "forecast_cpi": row["forecast_cpi"],
                "surprise": row["surprise"],
                "surprise_category": row["surprise_category"],
                **feature_row,
            }
        )

    return pd.DataFrame(rows).sort_values("release_time_et").reset_index(drop=True)

