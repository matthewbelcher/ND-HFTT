from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_SIGNALS_PATH = REPO_ROOT / "data/raw/events/FOMC_signals_raw_2025.csv"
OUTPUT_PATH = REPO_ROOT / "data/processed/cpi_signals.csv"


def parse_release_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    df["release_date"] = (
        df["release_date"]
        .astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
    )

    df["release_time"] = df["release_time"].astype(str).str.strip()

    dt_str = df["release_date"] + " " + df["release_time"]

    df["release_time_et"] = pd.to_datetime(dt_str, format="mixed")
    df["release_time_et"] = df["release_time_et"].dt.tz_localize("America/New_York")
    df["release_time_utc"] = df["release_time_et"].dt.tz_convert("UTC")

    return df


def compute_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["actual_cpi"] = pd.to_numeric(df["actual_cpi"])
    df["forecast_cpi"] = pd.to_numeric(df["forecast_cpi"])

    df["surprise"] = df["actual_cpi"] - df["forecast_cpi"]
    df["signal_strength"] = df["surprise"].abs()

    def map_signal(surprise: float) -> int:
        if surprise < 0:
            return 1
        if surprise > 0:
            return -1
        return 0

    df["signal"] = df["surprise"].apply(map_signal)

    return df


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_SIGNALS_PATH)
    df = parse_release_timestamp(df)
    df = compute_signal(df)

    out = df.sort_values("release_time_utc")

    out.to_csv(OUTPUT_PATH, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()