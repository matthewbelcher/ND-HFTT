#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from features import build_features, load_raw
from model import (
    BaseModel,
    TrainResult,
    make_split,
    plot_comparison,
    plot_feature_importance,
    refit_on_train_and_val,
    train_all,
)
from backtest import (
    backtest_model,
    plot_bankroll,
    print_summary as print_backtest_summary,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and backtest BTC Up/Down prediction models.",
    )
    parser.add_argument(
        "--data",
        required=True,
        metavar="FILE",
        help="Path to the raw Coinbase BTC trade CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory to write charts, results, and saved models. "
        "If omitted, charts display interactively and models are not saved.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.60,
        metavar="FLOAT",
        help="Training fraction (default: 0.60).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.20,
        metavar="FLOAT",
        help="Validation fraction (default: 0.20). " "Set to 0 to skip subset search.",
    )
    parser.add_argument(
        "--skip-svm",
        action="store_true",
        help="Skip the SVM model (slowest - useful for quick iteration).",
    )
    parser.add_argument(
        "--load-from",
        default=None,
        metavar="DIR",
        help="Load previously-saved models from DIR instead of training "
        "from scratch. Each subdirectory must contain model_meta.json.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["flat", "long_only"],
        help="Backtest strategies to run on each model " "(default: flat long_only).",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Starting bankroll for backtest (default: $1000).",
    )
    parser.add_argument(
        "--bet-fraction",
        type=float,
        default=0.10,
        help="Fraction of bankroll wagered per bet (default: 0.10).",
    )
    return parser.parse_args()


def _header(title: str) -> None:
    print(f"{'─' * 60}\n  {title}\n{'─' * 60}")


def _print_train_summary(results: list[TrainResult]) -> pd.DataFrame:
    """Compact training-side metrics table (validation accuracy)."""
    rows = [
        {
            "Model": r.name,
            "Val accuracy": f"{r.accuracy:.4f}",
            "Log Loss": f"{r.log_loss:.4f}" if r.log_loss is not None else "-",
            "# Features": r.n_features,
            "Top feature": r.features[0] if r.features else "-",
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    sep = "=" * 72
    print(f"\n{sep}\n  TRAINING SUMMARY\n{sep}")
    print(df.to_string(index=False))
    print(sep + "\n")
    return df


def _load_models(load_dir: Path) -> list[BaseModel]:
    """Resurrect every saved model under `load_dir` into a BaseModel object."""
    if not load_dir.is_dir():
        raise FileNotFoundError(f"--load-from directory not found: {load_dir}")

    models: list[BaseModel] = []
    for sub in sorted(load_dir.iterdir()):
        if (sub / BaseModel._META_NAME).exists():
            models.append(BaseModel.load_from(sub))

    if not models:
        raise RuntimeError(
            f"No saved models found under {load_dir} - "
            f"expected one subdirectory per model with model_meta.json."
        )
    return models


def _save_train_charts(
    out_dir: Path | None,
    results: list[TrainResult],
) -> None:
    """Comparison + feature-importance charts."""
    comp_path = str(out_dir / "model_comparison.png") if out_dir else None
    plot_comparison(results, save_path=comp_path)

    for r in results:
        if r.importances is None:
            continue
        slug = r.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = str(out_dir / f"feature_importance_{slug}.png") if out_dir else None
        plot_feature_importance(r, save_path=path)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 1–2  Data + features
    _header("STEP 1 - LOAD DATA")
    raw = load_raw(args.data)

    _header("STEP 2 - FEATURE ENGINEERING")
    data = build_features(raw)

    save_models_dir = (out_dir / "models") if out_dir else None

    # 3–5  Train (or load)
    if args.load_from:
        _header(f"LOAD MODELS FROM {args.load_from}")
        models = _load_models(Path(args.load_from))
    else:
        _header("STEP 3 - TRAIN  (chronological split + subset search)")
        models, results = train_all(
            data,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            skip_svm=args.skip_svm,
            save_dir=save_models_dir,
        )

        _header("STEP 4 - RESULTS")
        _print_train_summary(results)
        _save_train_charts(out_dir, results)

        if args.val_frac > 0.0:
            _header("STEP 5 - REFIT ON TRAIN + VAL")
            for model in models:
                refit_on_train_and_val(
                    model,
                    data,
                    train_frac=args.train_frac,
                    val_frac=args.val_frac,
                )
                if save_models_dir is not None:
                    # Overwrite the train-only checkpoint so the saved
                    # artefact is the production-ready model.
                    model.save(save_models_dir / type(model).__name__)

    # 6–7  Test split + backtest
    _header("STEP 6 - HELD-OUT TEST EVALUATION")
    split = make_split(data, train_frac=args.train_frac, val_frac=args.val_frac)
    if not split.has_test:
        print("No test slice available (train_frac + val_frac ~ 1.0). Done.")
        return

    backtests = []
    for model in models:
        for strategy in args.strategies:
            r = backtest_model(
                model,
                split.X_test,
                split.y_test,
                strategy=strategy,
                bankroll=args.bankroll,
                bet_fraction=args.bet_fraction,
            )
            backtests.append(r)
            print(r.summary())

            if out_dir is not None:
                slug = f"{type(model).__name__}_{strategy}".lower()
                plot_bankroll(r, save_path=str(out_dir / f"bankroll_{slug}.png"))

    summary_df = print_backtest_summary(backtests)

    # 8  Persist results table
    if out_dir is not None:
        json_path = out_dir / "backtest_summary.json"
        summary_df.to_json(json_path, orient="records", indent=4)
        print(f"Backtest JSON saved -> {json_path}")

    print("\nDone")


if __name__ == "__main__":
    main()
