from __future__ import annotations

from analysis import create_exploratory_plots
from config import FEATURES_OUTPUT_PATH, PLOTS_DIR, PROCESSED_DIR, SIGNALS_OUTPUT_PATH, VALIDATION_OUTPUT_PATH
from features import build_feature_table
from signals import apply_signals
from validation import run_validations


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_df = build_feature_table()
    feature_df.to_csv(FEATURES_OUTPUT_PATH, index=False)

    signal_df = apply_signals(feature_df)
    signal_df.to_csv(SIGNALS_OUTPUT_PATH, index=False)

    validation_df = run_validations(signal_df)
    validation_df.to_csv(VALIDATION_OUTPUT_PATH, index=False)

    create_exploratory_plots(signal_df, PLOTS_DIR)

    print(f"Wrote features to {FEATURES_OUTPUT_PATH}")
    print(f"Wrote signal results to {SIGNALS_OUTPUT_PATH}")
    print(f"Wrote validation summary to {VALIDATION_OUTPUT_PATH}")
    print(f"Wrote exploratory plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()

