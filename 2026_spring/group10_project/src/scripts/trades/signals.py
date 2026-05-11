from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config import (
    SIGNAL_A_OFI_THRESHOLD,
    SIGNAL_C_RATIO_REVERSAL_THRESHOLD,
    SIGNAL_C_RATIO_TREND_THRESHOLD,
    SIGNAL_HORIZONS_MINUTES,
)


def surprise_direction(surprise: float) -> int:
    if surprise > 0:
        return 1
    if surprise < 0:
        return -1
    return 0


def signal_a_direction(row: pd.Series, ofi_threshold: float = SIGNAL_A_OFI_THRESHOLD) -> int:
    ofi = float(row["ofi_0_60s"])
    if math.isnan(ofi) or abs(ofi) < ofi_threshold:
        return 0
    ofi_dir = int(np.sign(ofi))
    s_dir = surprise_direction(float(row["surprise"]))
    if ofi_dir == s_dir:
        return ofi_dir
    return 0


def classify_signal_c_regime(ratio_30s_to_5m: float) -> str:
    if math.isnan(ratio_30s_to_5m):
        return "unknown"
    if ratio_30s_to_5m >= SIGNAL_C_RATIO_TREND_THRESHOLD:
        return "trend"
    if ratio_30s_to_5m <= SIGNAL_C_RATIO_REVERSAL_THRESHOLD:
        return "reversal"
    return "mixed"


def apply_signals(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()

    df["signal_a_direction"] = df.apply(signal_a_direction, axis=1)
    for horizon in SIGNAL_HORIZONS_MINUTES:
        ret_col = f"ret_{horizon}m"
        pred_col = f"signal_a_correct_{horizon}m"
        pnl_col = f"signal_a_pnl_{horizon}m"
        df[pred_col] = np.where(
            df["signal_a_direction"] == 0,
            np.nan,
            (np.sign(df[ret_col]) == df["signal_a_direction"]).astype(float),
        )
        df[pnl_col] = df["signal_a_direction"] * df[ret_col]

    df["abs_surprise"] = df["surprise"].abs()
    df["abs_ofi_60s"] = df["ofi_0_60s"].abs()

    df["signal_b_target_vol"] = df["range_0_120m"]

    df["signal_c_regime"] = df["ratio_30s_to_5m"].apply(classify_signal_c_regime)
    df["signal_c_entry_delay_min"] = np.select(
        [
            df["signal_c_regime"] == "trend",
            df["signal_c_regime"] == "reversal",
        ],
        [0, 5],
        default=2,
    )
    df["signal_c_direction"] = np.sign(df["ret_5m"]).replace(0, np.nan)
    df["signal_c_pnl_120m"] = np.where(
        df["signal_c_direction"].isna(),
        np.nan,
        df["signal_c_direction"] * df["ret_120m"],
    )

    return df

