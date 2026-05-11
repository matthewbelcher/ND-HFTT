import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

BINANCE_DIR = REPO_ROOT / "data/raw/binance"
EVENTS_PATH = REPO_ROOT / "data/raw/events/FOMC_events_2025.csv"
PROCESSED_DIR = REPO_ROOT / "data/processed"
PLOTS_DIR = REPO_ROOT / "results/plots"

PRE_MINUTES = 60
POST_MINUTES = 180


def load_market_data(binance_dir: Path) -> pd.DataFrame:
    """Load and combine all BTCUSDT 1m CSV files from a directory."""
    csv_files = sorted(binance_dir.glob("BTCUSDT-1m-*.csv")) + sorted(binance_dir.glob("BTCUSDT-1m-*.zip"))

    if not csv_files:
        raise FileNotFoundError(f"No BTCUSDT CSV files found in {binance_dir}")

    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df["source_file"] = file.name
        dfs.append(df)

    market_df = pd.concat(dfs, ignore_index=True)
    market_df = market_df.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)

    return market_df


def check_market_data(df: pd.DataFrame) -> None:
    """Print basic diagnostics for the combined market data."""
    print(df.head())
    print(df.dtypes)
    print(df.shape)
    print(df["open_time"].min(), df["open_time"].max())
    print(df["open_time"].diff().value_counts().head())


def load_events(events_path: Path) -> pd.DataFrame:
    """Load CPI events and parse ET/UTC timestamps."""
    events_df = pd.read_csv(events_path)

    events_df["Release Date"] = events_df["Release Date"].str.replace(".", "", regex=False)

    events_df["event_time_et"] = pd.to_datetime(
        events_df["Release Date"] + " " + events_df["Release Time"],
        format="mixed"
    )

    events_df["event_time_et"] = events_df["event_time_et"].dt.tz_localize("America/New_York")
    events_df["event_time_utc"] = events_df["event_time_et"].dt.tz_convert("UTC")

    return events_df


def extract_event_window(
    market_df: pd.DataFrame,
    event_time: pd.Timestamp,
    event_name: str,
    reference_month: str,
    pre_minutes: int = 60,
    post_minutes: int = 180
) -> pd.DataFrame:
    """Extract a window of market data around a specified event."""
    window = market_df[
        (market_df["open_time"] >= event_time - pd.Timedelta(minutes=pre_minutes)) &
        (market_df["open_time"] <= event_time + pd.Timedelta(minutes=post_minutes))
    ].copy()

    if window.empty:
        return pd.DataFrame()

    window["event_name"] = event_name
    window["reference_month"] = reference_month
    window["event_time_utc"] = event_time
    window["minutes_from_event"] = (
        (window["open_time"] - event_time).dt.total_seconds() / 60
    ).astype(int)

    event_rows = window.loc[window["minutes_from_event"] == 0, "close"]
    if event_rows.empty:
        return pd.DataFrame()

    event_price = event_rows.iloc[0]
    window["event_price"] = event_price
    window["return_from_event"] = (window["close"] / event_price) - 1

    return window


def build_event_study_dataset(
    market_df: pd.DataFrame,
    events_df: pd.DataFrame,
    pre_minutes: int = 60,
    post_minutes: int = 180
) -> pd.DataFrame:
    """Loop over all events and build one combined event-study dataframe."""
    all_windows = []

    market_start = market_df["open_time"].min()
    market_end = market_df["open_time"].max()

    for _, row in events_df.iterrows():
        event_time = row["event_time_utc"]

        # Skip events outside available market data range
        if event_time < market_start or event_time > market_end:
            continue

        event_name = "FOMC"
        reference_month = row["Reference Month"]

        window = extract_event_window(
            market_df=market_df,
            event_time=event_time,
            event_name=event_name,
            reference_month=reference_month,
            pre_minutes=pre_minutes,
            post_minutes=post_minutes
        )

        if not window.empty:
            all_windows.append(window)

    if not all_windows:
        return pd.DataFrame()

    event_study_df = pd.concat(all_windows, ignore_index=True)
    return event_study_df


def plot_single_event(window: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Plot one event window."""
    title = f"BTCUSDT Return Around {window['event_name'].iloc[0]} Release ({window['event_time_utc'].iloc[0]})"

    plt.figure(figsize=(10, 5))
    plt.plot(window["minutes_from_event"], window["return_from_event"])
    plt.axvline(0, linestyle="--")
    plt.title(title)
    plt.xlabel("Minutes From Event")
    plt.ylabel("Return From Event")
    plt.tight_layout()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = str(window["event_time_utc"].iloc[0]).replace(' ', '_')
        plt.savefig(output_dir / f"cpi_{safe_name}.png")

    plt.show()


def plot_all_events(event_study_df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Plot each event window separately."""
    """DO NOT USE, PLOTS BASED ON REFERENCE MONTH"""
    if event_study_df.empty:
        print("No event windows to plot.")
        return

    grouped = event_study_df.groupby(["reference_month", "event_time_utc"])

    for (_, _), window in grouped:
        plot_single_event(window, output_dir=output_dir)


def plot_overlaid_events(event_study_df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Plot all event windows on one chart for comparison."""
    if event_study_df.empty:
        print("No event windows to plot.")
        return

    plt.figure(figsize=(10, 6))

    for reference_month, window in event_study_df.groupby("reference_month"):
        plt.plot(window["minutes_from_event"], window["return_from_event"], label=reference_month)

    plt.axvline(0, linestyle="--")
    plt.title("BTCUSDT Return Around CPI Releases")
    plt.xlabel("Minutes From Event")
    plt.ylabel("Return From Event")
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "cpi_all_events_overlay.png")

    plt.show()


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    market_df = load_market_data(BINANCE_DIR)
    check_market_data(market_df)

    print()
    print("------------------------------------------------------------")
    print()

    events_df = load_events(EVENTS_PATH)
    print(events_df.head())

    print()
    print("------------------------------------------------------------")
    print()

    event_study_df = build_event_study_dataset(
        market_df=market_df,
        events_df=events_df,
        pre_minutes=PRE_MINUTES,
        post_minutes=POST_MINUTES,
    )

    if event_study_df.empty:
        print("No matching event windows found.")
        return

    print(event_study_df.head())
    print(event_study_df.shape)

    event_study_df.to_csv(PROCESSED_DIR / "cpi_event_study.csv", index=False)
    event_study_df.to_parquet(PROCESSED_DIR / "cpi_event_study.parquet", index=False)

    plot_all_events(event_study_df, output_dir=PLOTS_DIR)


if __name__ == "__main__":
    main()