from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
TRADES_DIR = REPO_ROOT / "data/raw/trades"
CLEANED_TRADES_DIR = REPO_ROOT / "data/raw/cleaned trades"

CHUNK_SIZE = 2_000_000
WINDOW_START_HOUR_ET = 12  # 6:30 AM ET
WINDOW_END_HOUR_ET = 16  # 10:30 AM ET
WINDOW_MINUTE_ET = 0


def parse_release_date(trades_file: Path) -> pd.Timestamp:
    """Read YYYY-MM-DD from BTCUSDT-trades-YYYY-MM-DD.csv."""
    date_str = trades_file.stem.rsplit("-", 3)[-3:]
    return pd.Timestamp("-".join(date_str), tz="America/New_York")


def filter_chunk_to_event_window(chunk: pd.DataFrame, release_date_et: pd.Timestamp) -> pd.DataFrame:
    """Keep only trades within [06:30, 10:30] ET on release day."""
    trade_time_et = pd.to_datetime(chunk["time"], unit="ms", utc=True).dt.tz_convert("America/New_York")

    window_start_et = release_date_et.replace(
        hour=WINDOW_START_HOUR_ET, minute=WINDOW_MINUTE_ET, second=0, microsecond=0
    )
    window_end_et = release_date_et.replace(
        hour=WINDOW_END_HOUR_ET, minute=WINDOW_MINUTE_ET, second=0, microsecond=0
    )

    mask = (trade_time_et >= window_start_et) & (trade_time_et <= window_end_et)
    filtered = chunk.loc[mask].copy()
    if filtered.empty:
        return filtered

    # Keep original schema and append ET timestamp as requested.
    filtered["time_et"] = trade_time_et.loc[mask].dt.strftime("%Y-%m-%d %H:%M:%S.%f%z")
    return filtered


def clean_trade_files() -> None:
    trade_files = sorted(TRADES_DIR.glob("BTCUSDT-trades-*.csv"))
    if not trade_files:
        raise FileNotFoundError(f"No trade files found in {TRADES_DIR}")

    CLEANED_TRADES_DIR.mkdir(parents=True, exist_ok=True)

    total_rows_written = 0

    for trade_file in trade_files:
        release_date_et = parse_release_date(trade_file)
        rows_written_for_file = 0
        output_path = CLEANED_TRADES_DIR / trade_file.name
        first_write = True

        if output_path.exists():
            output_path.unlink()

        chunk_iter = pd.read_csv(
            trade_file,
            dtype={
                "id": "int64",
                "price": "float64",
                "qty": "float64",
                "quote_qty": "float64",
                "time": "int64",
                "is_buyer_maker": "bool",
            },
            chunksize=CHUNK_SIZE,
        )

        for chunk in chunk_iter:
            filtered = filter_chunk_to_event_window(chunk, release_date_et)
            if filtered.empty:
                continue

            filtered.to_csv(output_path, mode="a", header=first_write, index=False)
            first_write = False

            rows_written = len(filtered)
            rows_written_for_file += rows_written
            total_rows_written += rows_written

        print(f"{trade_file.name}: kept {rows_written_for_file:,} rows -> {output_path}")

    print(f"\nWrote {total_rows_written:,} total filtered rows across cleaned files in {CLEANED_TRADES_DIR}")


def main() -> None:
    clean_trade_files()


if __name__ == "__main__":
    main()
