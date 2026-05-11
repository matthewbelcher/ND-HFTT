from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BINANCE_DIR = REPO_ROOT / "data/raw/binance"
EVENTS_PATH = REPO_ROOT / "data/raw/events/CPI_ticker_events.csv"
PLOTS_DIR = REPO_ROOT / "data/processed/plots"

WINDOW_MS = 600_000  # 10 minutes
MILLISECOND_STEPS = np.arange(10, 1001, 10)
CHUNK_SIZE = 2_000_000


@dataclass(frozen=True)
class ReleaseConfig:
    reference_month: str
    release_time_utc: str


# Add new CPI releases here without changing analysis logic.
TARGET_RELEASES: list[ReleaseConfig] = [
    ReleaseConfig(reference_month="January 2024", release_time_utc="2024-02-13 13:30:00+00:00"),
    ReleaseConfig(reference_month="February 2024", release_time_utc="2024-03-12 13:30:00+00:00"),
]


def load_target_release_events(events_path: Path, release_configs: list[ReleaseConfig]) -> pd.DataFrame:
    """Load CPI events and return only configured release rows with UTC timestamps."""
    events_df = pd.read_csv(events_path)
    target_df = pd.DataFrame({"Reference Month": [release.reference_month for release in release_configs]})
    filtered = events_df.merge(target_df, on="Reference Month", how="inner")
    if filtered.empty:
        raise ValueError("Configured CPI reference months did not match CPI_ticker_events.csv.")

    release_time_map = {
        release.reference_month: pd.Timestamp(release.release_time_utc, tz="UTC")
        for release in release_configs
    }
    filtered["event_time_utc"] = filtered["Reference Month"].map(release_time_map)

    return filtered[["Reference Month", "event_time_utc"]].sort_values(
        "event_time_utc"
    ).reset_index(drop=True)


def load_book_ticker_data(binance_dir: Path, min_event_ms: int, max_event_ms: int) -> pd.DataFrame:
    """Load and combine all book ticker CSV rows in the inclusive event_time range."""
    csv_files = sorted(binance_dir.glob("BTCUSDT-bookTicker*"))
    if not csv_files:
        raise FileNotFoundError(f"No BTCUSDT book ticker files found in {binance_dir}")

    usecols = ["best_bid_price", "best_ask_price", "event_time"]
    chunks: list[pd.DataFrame] = []

    for csv_file in csv_files:
        chunk_iter = pd.read_csv(
            csv_file,
            usecols=usecols,
            dtype={"best_bid_price": "float64", "best_ask_price": "float64", "event_time": "int64"},
            chunksize=CHUNK_SIZE,
        )

        for chunk in chunk_iter:
            filtered = chunk[
                (chunk["event_time"] >= min_event_ms) & (chunk["event_time"] <= max_event_ms)
            ]
            if not filtered.empty:
                chunks.append(filtered)

    if not chunks:
        raise ValueError("No ticker rows found for configured CPI event windows.")

    ticker_df = pd.concat(chunks, ignore_index=True)
    ticker_df["mid_price"] = (ticker_df["best_bid_price"] + ticker_df["best_ask_price"]) / 2.0
    ticker_df["event_time_utc"] = pd.to_datetime(ticker_df["event_time"], unit="ms", utc=True)
    ticker_df = ticker_df.sort_values("event_time").reset_index(drop=True)
    return ticker_df[["event_time", "event_time_utc", "mid_price"]]


def compute_ms_change_series(
    event_df: pd.DataFrame,
    release_time_ms: int,
    pre_release_price: float,
    ms_steps: np.ndarray,
) -> pd.DataFrame:
    """Compute mid-price changes at each requested millisecond offset using nearest tick."""
    sorted_event_df = event_df.sort_values("event_time")[["event_time", "mid_price"]].copy()
    sorted_event_df["target_ms"] = sorted_event_df["event_time"] - release_time_ms

    target_df = pd.DataFrame({"elapsed_ms": ms_steps.astype("int64")})

    merged = pd.merge_asof(
        target_df,
        sorted_event_df.rename(columns={"target_ms": "elapsed_ms"}),
        on="elapsed_ms",
        direction="nearest",
    )

    merged["price_change"] = merged["mid_price"] - pre_release_price
    merged["pct_change"] = (merged["price_change"] / pre_release_price) * 100.0
    return merged


def summarize_release(
    reference_month: str,
    release_time_utc: pd.Timestamp,
    pre_release_price: float,
    ms_changes: pd.DataFrame,
) -> None:
    """Print requested summary metrics for one release."""
    key_points = [10, 100, 1000]
    summary_row: dict[str, float | str] = {
        "release_date": release_time_utc.date().isoformat(),
        "release_time_utc": release_time_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "reference_month": reference_month,
        "pre_release_price": pre_release_price,
    }

    for ms in key_points:
        row = ms_changes.loc[ms_changes["elapsed_ms"] == ms]
        if row.empty:
            raise ValueError(f"Missing millisecond change row for +{ms} ms.")
        row_data = row.iloc[0]
        summary_row[f"price_+{ms}ms"] = float(row_data["mid_price"])
        summary_row[f"pct_change_+{ms}ms"] = float(row_data["pct_change"])

    summary_df = pd.DataFrame([summary_row])
    print()
    print(f"=== {reference_month} CPI ({release_time_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC) ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:,.8f}"))


def plot_release_analysis(
    reference_month: str,
    release_time_utc: pd.Timestamp,
    event_df: pd.DataFrame,
    ms_changes: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create and save a 2-panel figure for one CPI release."""
    output_dir.mkdir(parents=True, exist_ok=True)

    elapsed_seconds = (event_df["event_time"] - int(release_time_utc.timestamp() * 1000)) / 1000.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    fig.suptitle(
        f"BTCUSD Book Ticker Reaction - {reference_month} CPI\nRelease: {release_time_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )

    axes[0].plot(elapsed_seconds, event_df["mid_price"], color="tab:blue", linewidth=1)
    axes[0].set_title("A) Mid-Price During First 10 Minutes After Release")
    axes[0].set_xlabel("Elapsed Time From Release (seconds)")
    axes[0].set_ylabel("Mid-Price (USD)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(ms_changes["elapsed_ms"], ms_changes["price_change"], color="tab:red", linewidth=1.2)
    axes[1].set_title("B) Millisecond-Resolution Mid-Price Change")
    axes[1].set_xlabel("Elapsed Time From Release (milliseconds)")
    axes[1].set_ylabel("Price Change From t=0 (USD)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    safe_name = reference_month.lower().replace(" ", "_")
    output_path = output_dir / f"btcusd_cpi_ticker_{safe_name}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    release_events = load_target_release_events(EVENTS_PATH, TARGET_RELEASES)

    release_times_ms = release_events["event_time_utc"].apply(lambda ts: int(ts.timestamp() * 1000))
    min_event_ms = int(release_times_ms.min() - 1_000)  # include one second pre-release for baseline lookup
    max_event_ms = int(release_times_ms.max() + WINDOW_MS)

    ticker_df = load_book_ticker_data(BINANCE_DIR, min_event_ms=min_event_ms, max_event_ms=max_event_ms)

    for _, event in release_events.iterrows():
        reference_month = event["Reference Month"]
        release_time_utc: pd.Timestamp = event["event_time_utc"]
        release_time_ms = int(release_time_utc.timestamp() * 1000)

        pre_release_rows = ticker_df[ticker_df["event_time"] <= release_time_ms]
        if pre_release_rows.empty:
            raise ValueError(f"No tick at or before release for {reference_month} ({release_time_utc}).")
        pre_release_row = pre_release_rows.iloc[-1]
        pre_release_price = float(pre_release_row["mid_price"])

        window_end_ms = release_time_ms + WINDOW_MS
        event_window = ticker_df[
            (ticker_df["event_time"] >= release_time_ms) & (ticker_df["event_time"] <= window_end_ms)
        ].copy()
        if event_window.empty:
            raise ValueError(f"No post-release ticks found in 10-minute window for {reference_month}.")

        ms_changes = compute_ms_change_series(
            event_df=event_window,
            release_time_ms=release_time_ms,
            pre_release_price=pre_release_price,
            ms_steps=MILLISECOND_STEPS,
        )

        plot_release_analysis(
            reference_month=reference_month,
            release_time_utc=release_time_utc,
            event_df=event_window,
            ms_changes=ms_changes,
            output_dir=PLOTS_DIR,
        )
        summarize_release(
            reference_month=reference_month,
            release_time_utc=release_time_utc,
            pre_release_price=pre_release_price,
            ms_changes=ms_changes,
        )


if __name__ == "__main__":
    main()
