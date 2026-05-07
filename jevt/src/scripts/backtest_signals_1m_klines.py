from pathlib import Path
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EVENT_STUDY_PATH = REPO_ROOT / "data/processed/cpi_event_study.parquet"
SIGNALS_PATH = REPO_ROOT / "data/processed/cpi_signals.csv"
OUTPUT_PATH = REPO_ROOT / "data/processed/cpi_backtest_results.csv"

# where delay = 0 means entering on first minute bar after the release
DELAYS = [0, 1, 2, 5, 10, 30]
HOLDS = [1, 5, 10, 30, 60]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    event_study = pd.read_parquet(EVENT_STUDY_PATH)
    signals = pd.read_csv(SIGNALS_PATH)

    signals["release_time_utc"] = pd.to_datetime(signals["release_time_utc"], utc=True)
    event_study["event_time_utc"] = pd.to_datetime(event_study["event_time_utc"], utc=True)

    return event_study, signals


def get_price_at_minute(window: pd.DataFrame, minute: int) -> float | None:
    rows = window.loc[window["minutes_from_event"] == minute, "close"]
    if rows.empty:
        return None
    return float(rows.iloc[0])


def simulate_trade(window: pd.DataFrame, signal: int, delay: int, hold: int) -> dict | None:
    if signal == 0:
        return None

    entry_price = get_price_at_minute(window, delay)
    exit_price = get_price_at_minute(window, delay + hold)

    if entry_price is None or exit_price is None:
        return None

    raw_return = (exit_price / entry_price) - 1.0

    if signal == 1:
        trade_return = raw_return
        direction = "long"
    else:
        trade_return = -raw_return
        direction = "short"

    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "raw_return": raw_return,
        "trade_return": trade_return,
        "direction": direction,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    event_study, signals = load_inputs()

    event_keys = event_study[["event_time_utc"]].drop_duplicates()
    merged_events = signals.merge(
        event_keys,
        left_on="release_time_utc",
        right_on="event_time_utc",
        how="inner",
    )

    all_results = []

    for _, event_row in merged_events.iterrows():
        event_time_utc = event_row["event_time_utc"]
        signal = int(event_row["signal"])
        surprise = float(event_row["surprise"])
        reference_month = str(event_row["reference_month"])

        window = event_study[
            (event_study["event_time_utc"] == event_time_utc)
        ].copy()

        expected_trades = 0
        executed_trades = 0
        for delay in DELAYS:
            for hold in HOLDS:
                expected_trades += 1
                result = simulate_trade(window, signal=signal, delay=delay, hold=hold)
                if result is None:
                    continue
                executed_trades += 1

                all_results.append({
                    "reference_month": reference_month,
                    "event_time_utc": event_time_utc,
                    "signal": signal,
                    "surprise": surprise,
                    "delay_minutes": delay,
                    "hold_minutes": hold,
                    **result,
                })
        print(f"\nTrade Coverage: {executed_trades} / {expected_trades} "
      f"({executed_trades / expected_trades:.2%})")

    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        print("No backtest results generated.")
        return

    results_df.to_csv(OUTPUT_PATH, index=False)
    print(results_df.head().to_string(index=False))

    summary = (
        results_df
        .groupby(["delay_minutes", "hold_minutes"], as_index=False)
        .agg(
            n_trades=("trade_return", "count"),
            avg_return=("trade_return", "mean"),
            median_return=("trade_return", "median"),
            win_rate=("trade_return", lambda s: (s > 0).mean()),
        )
        .sort_values(["hold_minutes", "delay_minutes"])
    )

    print("\nBacktest Summary:\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()