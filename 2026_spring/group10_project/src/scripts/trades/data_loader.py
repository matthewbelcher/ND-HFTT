from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import EASTERN_TZ, RAW_CLEANED_TRADES_DIR, RAW_EVENTS_PATH, RELEASE_HOUR_ET, RELEASE_MINUTE_ET


def normalize_events_table(events_df: pd.DataFrame) -> pd.DataFrame:
    df = events_df.copy()
    df.columns = df.columns.str.strip().str.lower()

    required = {"reference_month", "release_date", "release_time", "actual_cpi", "forecast_cpi"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected metadata columns: {sorted(missing)}")

    df["release_date"] = (
        df["release_date"].astype(str).str.strip().str.replace(".", "", regex=False)
    )
    df["release_time"] = df["release_time"].astype(str).str.strip()
    dt_str = df["release_date"] + " " + df["release_time"]
    df["release_time_et"] = pd.to_datetime(dt_str, format="mixed").dt.tz_localize(EASTERN_TZ)
    df["release_date_key"] = df["release_time_et"].dt.strftime("%Y-%m-%d")

    df["actual_cpi"] = pd.to_numeric(df["actual_cpi"])
    df["forecast_cpi"] = pd.to_numeric(df["forecast_cpi"])
    df["surprise"] = df["actual_cpi"] - df["forecast_cpi"]
    return df.sort_values("release_time_et").reset_index(drop=True)


def categorize_surprise(surprise: float, threshold: float) -> str:
    if surprise > threshold:
        return "hot"
    if surprise < -threshold:
        return "cold"
    return "inline"


def load_cpi_metadata(events_path: Path = RAW_EVENTS_PATH, surprise_threshold: float = 0.1) -> pd.DataFrame:
    df = pd.read_csv(events_path)
    df = normalize_events_table(df)
    df["surprise_category"] = df["surprise"].apply(lambda s: categorize_surprise(float(s), surprise_threshold))
    return df


def parse_release_date_from_trade_file(trades_file: Path) -> str:
    token = trades_file.stem.replace("BTCUSDT-trades-", "")
    ts = pd.Timestamp(token).tz_localize(EASTERN_TZ).replace(
        hour=RELEASE_HOUR_ET, minute=RELEASE_MINUTE_ET, second=0, microsecond=0
    )
    return ts.strftime("%Y-%m-%d")


def load_trade_file(trades_file: Path) -> pd.DataFrame:
    df = pd.read_csv(trades_file)
    df["time_utc"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    if "time_et" in df.columns:
        parsed_et = pd.to_datetime(df["time_et"], errors="coerce", utc=True)
        df["time_et_dt"] = parsed_et.dt.tz_convert(EASTERN_TZ)
    else:
        df["time_et_dt"] = df["time_utc"].dt.tz_convert(EASTERN_TZ)
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
    return df.sort_values("time_utc").reset_index(drop=True)


def load_trade_days(
    trades_dir: Path = RAW_CLEANED_TRADES_DIR,
    events_path: Path = RAW_EVENTS_PATH,
    surprise_threshold: float = 0.1,
) -> pd.DataFrame:
    metadata = load_cpi_metadata(events_path=events_path, surprise_threshold=surprise_threshold)
    metadata_by_date = metadata.set_index("release_date_key")

    trade_files = sorted(trades_dir.glob("BTCUSDT-trades-*.csv"))
    if not trade_files:
        raise FileNotFoundError(f"No cleaned trade files found in {trades_dir}")

    rows: list[dict[str, object]] = []
    for trade_file in trade_files:
        release_date_key = parse_release_date_from_trade_file(trade_file)
        if release_date_key not in metadata_by_date.index:
            continue
        meta_row = metadata_by_date.loc[release_date_key]
        rows.append(
            {
                "trade_file": trade_file,
                "release_date_key": release_date_key,
                "reference_month": str(meta_row["reference_month"]),
                "release_time_et": meta_row["release_time_et"],
                "actual_cpi": float(meta_row["actual_cpi"]),
                "forecast_cpi": float(meta_row["forecast_cpi"]),
                "surprise": float(meta_row["surprise"]),
                "surprise_category": str(meta_row["surprise_category"]),
            }
        )

    if not rows:
        raise ValueError("No trade files matched CPI metadata release dates.")

    return pd.DataFrame(rows).sort_values("release_time_et").reset_index(drop=True)

