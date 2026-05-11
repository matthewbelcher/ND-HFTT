from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BINANCE_DIR = REPO_ROOT / "data/raw/binance"
EVENTS_PATH = REPO_ROOT / "data/raw/events/CPI_ticker_events.csv"
PLOTS_DIR = REPO_ROOT / "data/processed/plots/signals"

SHORT_WINDOW_MS = 2_000
LONG_WINDOW_MS = 600_000
CHUNK_SIZE = 2_000_000
AGG_WINDOWS_MS = [100, 200, 500, 1000]

# Add future releases by appending a dict to this list.
RELEASES: list[dict[str, str]] = [
    {"label": "Jan 2024 CPI", "t0_utc": "2024-02-13 13:30:00"},
    {"label": "Feb 2024 CPI", "t0_utc": "2024-03-12 13:30:00"},
]


@dataclass(frozen=True)
class ReleaseEvent:
    label: str
    t0_utc: pd.Timestamp
    t0_ms: int


def to_release_events(releases: list[dict[str, str]]) -> list[ReleaseEvent]:
    parsed: list[ReleaseEvent] = []
    for release in releases:
        t0 = pd.Timestamp(release["t0_utc"], tz="UTC")
        parsed.append(ReleaseEvent(label=release["label"], t0_utc=t0, t0_ms=int(t0.timestamp() * 1000)))
    return sorted(parsed, key=lambda x: x.t0_ms)


def parse_filename_date_token(token: str, is_end: bool) -> pd.Timestamp:
    """Parse YYYY-MM or YYYY-MM-DD token to UTC timestamp."""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
        base = pd.Timestamp(token, tz="UTC")
        if is_end:
            return base + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
        return base

    if re.fullmatch(r"\d{4}-\d{2}", token):
        year, month = token.split("-")
        year_i = int(year)
        month_i = int(month)
        if is_end:
            month_end_day = calendar.monthrange(year_i, month_i)[1]
            return pd.Timestamp(year=year_i, month=month_i, day=month_end_day, tz="UTC") + pd.Timedelta(
                days=1
            ) - pd.Timedelta(milliseconds=1)
        return pd.Timestamp(year=year_i, month=month_i, day=1, tz="UTC")

    raise ValueError(f"Unsupported date token in filename: {token}")


def parse_book_ticker_file_range(csv_path: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """
    Parse date range from filename.
    Expected pattern example: BTCUSDT-bookTicker-2024-01-2024-02-13-window.csv
    """
    match = re.search(r"BTCUSDT-bookTicker-(\d{4}-\d{2}(?:-\d{2})?)-(\d{4}-\d{2}(?:-\d{2})?)", csv_path.name)
    if not match:
        return None

    start_token, end_token = match.group(1), match.group(2)
    start_ts = parse_filename_date_token(start_token, is_end=False)
    end_ts = parse_filename_date_token(end_token, is_end=True)
    return start_ts, end_ts


def select_overlapping_files(
    binance_dir: Path, analysis_start_ms: int, analysis_end_ms: int
) -> list[Path]:
    analysis_start = pd.to_datetime(analysis_start_ms, unit="ms", utc=True)
    analysis_end = pd.to_datetime(analysis_end_ms, unit="ms", utc=True)

    # Only read CSV files; ZIP archives are intentionally ignored.
    all_files = sorted(binance_dir.glob("BTCUSDT-bookTicker*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No book ticker CSV files found in {binance_dir}")

    selected: list[Path] = []
    fallback: list[Path] = []
    for csv_file in all_files:
        file_range = parse_book_ticker_file_range(csv_file)
        if file_range is None:
            fallback.append(csv_file)
            continue

        file_start, file_end = file_range
        overlaps = (file_start <= analysis_end) and (file_end >= analysis_start)
        if overlaps:
            selected.append(csv_file)

    # If any file cannot be parsed, include it to avoid false exclusion.
    selected.extend(fallback)
    if not selected:
        raise ValueError("No overlapping book ticker files found for requested windows.")
    return selected


def load_events_table(events_path: Path) -> pd.DataFrame:
    events_df = pd.read_csv(events_path)
    events_df["event_time_et"] = pd.to_datetime(
        events_df["Release Date"] + " " + events_df["Release Time"], format="%Y-%m-%d %H:%M:%S"
    ).dt.tz_localize("America/New_York")
    events_df["event_time_utc"] = events_df["event_time_et"].dt.tz_convert("UTC")
    return events_df


def load_book_ticker_data(selected_files: list[Path], min_ms: int, max_ms: int) -> pd.DataFrame:
    usecols = [
        "update_id",
        "best_bid_price",
        "best_bid_qty",
        "best_ask_price",
        "best_ask_qty",
        "transaction_time",
        "event_time",
    ]
    dtypes = {
        "update_id": "int64",
        "best_bid_price": "float64",
        "best_bid_qty": "float64",
        "best_ask_price": "float64",
        "best_ask_qty": "float64",
        "transaction_time": "int64",
        "event_time": "int64",
    }

    chunks: list[pd.DataFrame] = []
    for csv_file in selected_files:
        for chunk in pd.read_csv(csv_file, usecols=usecols, dtype=dtypes, chunksize=CHUNK_SIZE):
            filtered = chunk[(chunk["event_time"] >= min_ms) & (chunk["event_time"] <= max_ms)]
            if not filtered.empty:
                chunks.append(filtered)

    if not chunks:
        raise ValueError("No ticker rows found for requested event windows.")

    ticker_df = pd.concat(chunks, ignore_index=True)
    ticker_df["mid_price"] = (ticker_df["best_bid_price"] + ticker_df["best_ask_price"]) / 2.0
    ticker_df["imbalance"] = (ticker_df["best_bid_qty"] - ticker_df["best_ask_qty"]) / (
        ticker_df["best_bid_qty"] + ticker_df["best_ask_qty"]
    )
    ticker_df["spread"] = ticker_df["best_ask_price"] - ticker_df["best_bid_price"]
    ticker_df["event_time_utc"] = pd.to_datetime(ticker_df["event_time"], unit="ms", utc=True)
    return ticker_df.sort_values("event_time").reset_index(drop=True)


def nearest_values(series_df: pd.DataFrame, target_ms: np.ndarray, value_cols: list[str]) -> pd.DataFrame:
    left = pd.DataFrame({"target_ms": target_ms.astype("int64")}).sort_values("target_ms").reset_index(drop=True)
    right = series_df[["event_time"] + value_cols].sort_values("event_time").rename(
        columns={"event_time": "target_ms"}
    )
    return pd.merge_asof(left, right, on="target_ms", direction="nearest")


def compute_event_features(event_data: pd.DataFrame, t0_ms: int) -> dict[str, float | int]:
    nearest = nearest_values(event_data, np.array([t0_ms]), ["mid_price"])
    mid_t0 = float(nearest.iloc[0]["mid_price"])

    post_rows = event_data[event_data["event_time"] >= t0_ms]
    changed = post_rows[post_rows["mid_price"] != mid_t0]
    if changed.empty:
        first_tick_ms = -1
        first_tick_direction = -1
    else:
        first_row = changed.iloc[0]
        first_tick_ms = int(first_row["event_time"] - t0_ms)
        first_tick_direction = 1 if float(first_row["mid_price"]) > mid_t0 else -1

    windows = {w: event_data[(event_data["event_time"] >= t0_ms) & (event_data["event_time"] <= t0_ms + w)] for w in AGG_WINDOWS_MS}
    mean_imbalance = {
        w: (float(df["imbalance"].mean()) if not df.empty else np.nan)
        for w, df in windows.items()
    }

    velocity_targets = np.array([t0_ms, t0_ms + 100, t0_ms + 300, t0_ms + 1000], dtype="int64")
    velocity_prices = nearest_values(event_data, velocity_targets, ["mid_price"])
    mid_0, mid_100, mid_300, mid_1000 = [float(v) for v in velocity_prices["mid_price"].to_numpy()]
    velocity_0_100 = (mid_100 - mid_0) / 100.0
    velocity_100_300 = (mid_300 - mid_100) / 200.0
    velocity_300_1000 = (mid_1000 - mid_300) / 700.0
    acceleration = velocity_100_300 - velocity_0_100

    label_targets = np.array([t0_ms + 60_000, t0_ms + 300_000, t0_ms + 600_000], dtype="int64")
    label_prices = nearest_values(event_data, label_targets, ["mid_price"])
    label_1min = 1 if float(label_prices.iloc[0]["mid_price"]) > mid_0 else -1
    label_5min = 1 if float(label_prices.iloc[1]["mid_price"]) > mid_0 else -1
    label_10min = 1 if float(label_prices.iloc[2]["mid_price"]) > mid_0 else -1

    features: dict[str, float | int] = {
        "pre_release_price": mid_0,
        "first_tick_ms": first_tick_ms,
        "first_tick_direction": first_tick_direction,
        "mean_imbalance_0_100ms": mean_imbalance[100],
        "mean_imbalance_0_200ms": mean_imbalance[200],
        "mean_imbalance_0_500ms": mean_imbalance[500],
        "mean_imbalance_0_1000ms": mean_imbalance[1000],
        "velocity_0_100ms": velocity_0_100,
        "velocity_100_300ms": velocity_100_300,
        "velocity_300_1000ms": velocity_300_1000,
        "acceleration": acceleration,
        "label_1min": label_1min,
        "label_5min": label_5min,
        "label_10min": label_10min,
    }
    return features


def plot_event(
    release: ReleaseEvent,
    event_2s: pd.DataFrame,
    features: dict[str, float | int],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    elapsed_ms = event_2s["event_time"] - release.t0_ms

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True, gridspec_kw={"height_ratios": [2, 2, 2, 1.4]})
    fig.suptitle(f"{release.label} - Release {release.t0_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    axes[0].plot(elapsed_ms, event_2s["mid_price"], color="tab:blue", linewidth=1.0)
    axes[0].axvline(0, linestyle="--", color="black", linewidth=1)
    axes[0].set_ylabel("Mid Price (USD)")
    axes[0].set_title("Mid-Price (0-2000ms)")
    axes[0].grid(alpha=0.3)

    pos = event_2s["imbalance"].where(event_2s["imbalance"] >= 0)
    neg = event_2s["imbalance"].where(event_2s["imbalance"] < 0)
    axes[1].plot(elapsed_ms, pos, color="green", linewidth=1.0)
    axes[1].plot(elapsed_ms, neg, color="red", linewidth=1.0)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].axvline(0, linestyle="--", color="black", linewidth=1)
    axes[1].set_ylabel("Imbalance")
    axes[1].set_title("Order Book Imbalance")
    axes[1].grid(alpha=0.3)

    axes[2].plot(elapsed_ms, event_2s["spread"], color="tab:purple", linewidth=1.0)
    axes[2].axvline(0, linestyle="--", color="black", linewidth=1)
    axes[2].set_ylabel("Spread (USD)")
    axes[2].set_title("Bid-Ask Spread")
    axes[2].grid(alpha=0.3)

    summary_lines = [
        f"first_tick_direction: {int(features['first_tick_direction'])}",
        f"first_tick_ms: {int(features['first_tick_ms'])}",
        f"velocity[0-100ms]: {float(features['velocity_0_100ms']):.6f}",
        f"acceleration: {float(features['acceleration']):.6f}",
        f"mean imbalance[0-200ms]: {float(features['mean_imbalance_0_200ms']):.6f}",
        (
            "labels (+1m/+5m/+10m): "
            f"{int(features['label_1min'])} / {int(features['label_5min'])} / {int(features['label_10min'])}"
        ),
    ]
    axes[3].axis("off")
    axes[3].text(
        0.01,
        0.95,
        "\n".join(summary_lines),
        transform=axes[3].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "whitesmoke", "alpha": 0.95, "edgecolor": "gray"},
    )

    axes[2].set_xlabel("Elapsed Time From Release (ms)")
    axes[0].set_xlim(0, SHORT_WINDOW_MS)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    safe_label = re.sub(r"[^a-z0-9]+", "_", release.label.lower()).strip("_")
    fig.savefig(output_dir / f"signal_detection_{safe_label}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    releases = to_release_events(RELEASES)

    # Load CPI events for consistency checks and future extensibility.
    _ = load_events_table(EVENTS_PATH)

    min_ms = min(release.t0_ms for release in releases) - 1_000
    max_ms = max(release.t0_ms for release in releases) + LONG_WINDOW_MS

    selected_files = select_overlapping_files(BINANCE_DIR, min_ms, max_ms)
    ticker_df = load_book_ticker_data(selected_files, min_ms=min_ms, max_ms=max_ms)

    summary_rows: list[dict[str, float | int | str]] = []
    for release in releases:
        event_long = ticker_df[
            (ticker_df["event_time"] >= release.t0_ms) & (ticker_df["event_time"] <= release.t0_ms + LONG_WINDOW_MS)
        ].copy()
        if event_long.empty:
            raise ValueError(f"No rows in [t0, t0+10min] for {release.label}.")

        event_2s = event_long[event_long["event_time"] <= release.t0_ms + SHORT_WINDOW_MS].copy()
        if event_2s.empty:
            raise ValueError(f"No rows in [t0, t0+2000ms] for {release.label}.")

        features = compute_event_features(event_long, release.t0_ms)
        plot_event(release, event_2s, features, PLOTS_DIR)

        summary_rows.append(
            {
                "release_label": release.label,
                "t0_utc": release.t0_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "pre_release_price": features["pre_release_price"],
                "first_tick_ms": features["first_tick_ms"],
                "first_tick_direction": features["first_tick_direction"],
                "mean_imbalance_0_200ms": features["mean_imbalance_0_200ms"],
                "velocity_0_100ms": features["velocity_0_100ms"],
                "acceleration": features["acceleration"],
                "label_1min": features["label_1min"],
                "label_5min": features["label_5min"],
                "label_10min": features["label_10min"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:,.8f}"))


if __name__ == "__main__":
    main()
