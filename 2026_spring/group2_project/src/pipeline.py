#!/usr/bin/env python3
"""
# one-off prediction using a saved model
python src/pipeline.py --data-dir Data/BitcoinData --models-dir runs/models

# train fresh on the active CSV, save, predict
python src/pipeline.py --data-dir Data/BitcoinData --models-dir runs/models --train

# poll forever, predict every 5 minutes as new buckets close
python src/pipeline.py --data-dir Data/BitcoinData --models-dir runs/models --watch
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from features import (
    ALL_FEATURES,
    TARGET_COL,
    build_features,
    load_raw,
)
from model import (
    BaseModel,
    XGBoostModel,
    make_split,
    refit_on_train_and_val,
)
from backtest import backtest_model, print_summary as print_backtest_summary

DEFAULT_CSV_GLOB = "btcusd_trades_*.csv"
DEFAULT_POLL_SECONDS = 300  # one 5-minute bucket


# CSV discovery


def find_latest_csv(
    data_dir: str | Path,
    pattern: str = DEFAULT_CSV_GLOB,
) -> Path:
    """
    Pick the most recently-modified CSV in `data_dir` matching `pattern`.

    The Coinbase collector writes one timestamped file per session; this
    helper finds the active one without hard-coding its name.
    """
    data_dir = Path(data_dir)
    candidates = sorted(
        data_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern!r} in {data_dir}")
    return candidates[0]


# Prediction record


@dataclass
class Prediction:
    """One model's call on a single 5-minute bucket."""

    bucket_time: pd.Timestamp
    direction: int  # 0 = down, 1 = up
    probability: float  # P(up)
    model_name: str

    def __str__(self) -> str:
        arrow = "UP" if self.direction == 1 else "DOWN"
        return (
            f"[{self.bucket_time}]  {self.model_name:25s}  "
            f"-> {arrow}  (P(up)={self.probability:.4f})"
        )


# Pipeline


class LivePipeline:
    """
    Orchestrates the live feature -> predict loop for a single model family.

    A pipeline owns:
        - the directory holding live data CSVs
        - the model class (used when training fresh) and/or model directory
          (used when loading)
        - a single fitted BaseModel instance, lazily produced

    Typical usage:

        pipe = LivePipeline(
            data_dir   = "Data/BitcoinData",
            model_dir  = "runs/models/XGBoostModel",
        )
        pipe.load()            # bring the saved model into memory
        print(pipe.predict_latest())

        # or, train fresh on the most recent CSV
        pipe = LivePipeline(
            data_dir   = "Data/BitcoinData",
            model_cls  = XGBoostModel,
            model_dir  = "runs/models/XGBoostModel",
        )
        pipe.train_and_save()
        print(pipe.predict_latest())
    """

    def __init__(
        self,
        data_dir: str | Path,
        model_cls: type[BaseModel] = XGBoostModel,
        model_dir: str | Path | None = None,
        csv_pattern: str = DEFAULT_CSV_GLOB,
        train_frac: float = 0.60,
        val_frac: float = 0.20,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.csv_pattern = csv_pattern
        self.model_cls = model_cls
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.model: BaseModel | None = None

    # data loading

    def latest_csv(self) -> Path:
        """Most recently-modified CSV in the data directory."""
        return find_latest_csv(self.data_dir, self.csv_pattern)

    def build_dataset(self, csv_path: str | Path | None = None) -> pd.DataFrame:
        """
        Load the active CSV and engineer the full feature DataFrame.

        Parameters
        ----------
        csv_path : optional override. Defaults to the latest CSV under
                   self.data_dir matching self.csv_pattern.
        """
        path = Path(csv_path) if csv_path is not None else self.latest_csv()
        print(f"Pipeline data: {path}")
        return build_features(load_raw(str(path)))

    # model lifecycle

    def load(self) -> BaseModel:
        """Load `self.model_dir` into memory. Sets and returns self.model."""
        if self.model_dir is None:
            raise RuntimeError("LivePipeline.load() requires `model_dir`.")
        self.model = BaseModel.load_from(self.model_dir)
        return self.model

    def train_and_save(
        self,
        csv_path: str | Path | None = None,
    ) -> BaseModel:
        """
        Train a fresh model of `self.model_cls` end-to-end:
            build features -> 60/20/20 split -> subset search on val
            -> refit on train+val -> save (if model_dir was supplied).

        Returns the fitted model and stores it on self.
        """
        data = self.build_dataset(csv_path)
        split = make_split(data, train_frac=self.train_frac, val_frac=self.val_frac)

        model = self.model_cls()
        print(f"Training {model.name}...")
        result = model.train(
            split.X_train,
            split.y_train,
            X_val=split.X_val,
            y_val=split.y_val,
        )
        print(
            f"  Val accuracy : {result.accuracy:.4f}  "
            f"using {result.n_features} features"
        )
        print(f"  Features     : {result.features}")

        if self.val_frac > 0.0:
            refit_on_train_and_val(
                model,
                data,
                train_frac=self.train_frac,
                val_frac=self.val_frac,
            )

        if self.model_dir is not None:
            model.save(self.model_dir)

        self.model = model
        return model

    # inference

    def predict_latest(
        self,
        csv_path: str | Path | None = None,
    ) -> Prediction:
        """
        Predict the direction of the latest fully-formed 5-minute bucket.

        Returns a `Prediction` carrying timestamp, class label, and P(up).
        Note: build_features() drops the trailing window because its
        `next_up_down` label is unknown - so the "latest fully-formed
        bucket" here means the last row whose features are stable.
        """
        if self.model is None:
            raise RuntimeError(
                "No model loaded. Call .load() or .train_and_save() first."
            )

        data = self.build_dataset(csv_path)
        if data.empty:
            raise RuntimeError("Feature DataFrame is empty - not enough history yet.")

        latest = data.iloc[[-1]]
        direction = int(self.model.predict(latest[ALL_FEATURES])[0])
        probability = float(self.model.predict_proba(latest[ALL_FEATURES])[0])

        return Prediction(
            bucket_time=pd.Timestamp(latest["time"].iloc[0]),
            direction=direction,
            probability=probability,
            model_name=self.model.name,
        )

    def backtest_tail(
        self,
        csv_path: str | Path | None = None,
        strategies: list[str] | None = None,
        bankroll: float = 1000.0,
        bet_fraction: float = 0.10,
    ) -> list:
        """
        Quick sanity check: run a backtest on the test slice of the active
        CSV using the currently-loaded model. Useful for "is this saved
        model still earning on recent data?".
        """
        if self.model is None:
            raise RuntimeError("No model loaded.")
        if strategies is None:
            strategies = ["flat", "long_only"]

        data = self.build_dataset(csv_path)
        split = make_split(data, train_frac=self.train_frac, val_frac=self.val_frac)
        if not split.has_test:
            raise RuntimeError("No test slice available for backtest_tail.")

        results = [
            backtest_model(
                self.model,
                split.X_test,
                split.y_test,
                strategy=s,
                bankroll=bankroll,
                bet_fraction=bet_fraction,
            )
            for s in strategies
        ]
        print_backtest_summary(results)
        return results

    # watch loop

    def watch(
        self,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        max_iterations: int | None = None,
    ) -> None:
        """
        Poll the data directory and emit a fresh prediction whenever a new
        5-minute bucket closes.

        Parameters
        ----------
        poll_seconds   : how often to re-evaluate. Defaults to 300 (5 min).
        max_iterations : stop after N predictions. None = forever.
        """
        if self.model is None:
            raise RuntimeError("No model loaded.")

        seen: pd.Timestamp | None = None
        iterations = 0

        while True:
            try:
                pred = self.predict_latest()
            except (FileNotFoundError, RuntimeError) as exc:
                print(f"  watch: {exc}")
            else:
                if pred.bucket_time != seen:
                    print(pred)
                    seen = pred.bucket_time
                    iterations += 1
                    if max_iterations is not None and iterations >= max_iterations:
                        return

            time.sleep(poll_seconds)


# Class lookup for CLI


def _resolve_model_class(name: str) -> type[BaseModel]:
    """Map --model-cls strings (case-insensitive aliases) to a class."""
    aliases = {
        "lr": "LogisticRegressionModel",
        "logistic": "LogisticRegressionModel",
        "logisticregression": "LogisticRegressionModel",
        "xgb": "XGBoostModel",
        "xgboost": "XGBoostModel",
        "rf": "RandomForestModel",
        "randomforest": "RandomForestModel",
        "svm": "SVMModel",
    }
    canonical = aliases.get(name.lower(), name)

    # Lazy import to avoid duplicating the registry list.
    from model import MODEL_REGISTRY  # noqa: PLC0415  (runtime lookup)

    if canonical not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model class {name!r}. "
            f"Options: {list(MODEL_REGISTRY)} or aliases {list(aliases)}."
        )
    return MODEL_REGISTRY[canonical]


# CLI


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live BTC Up/Down prediction pipeline.",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        metavar="DIR",
        help="Directory holding live Coinbase trade CSVs.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        metavar="DIR",
        help="Where the model is saved / will be saved.",
    )
    parser.add_argument(
        "--model-cls",
        default="xgboost",
        help="Model class to train (when --train is set). "
        "Aliases: lr, xgb, rf, svm. Default: xgb.",
    )
    parser.add_argument(
        "--csv-pattern",
        default=DEFAULT_CSV_GLOB,
        help=f"Glob for trade CSVs (default: {DEFAULT_CSV_GLOB}).",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a fresh model on the latest CSV before predicting.",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run a quick backtest on the test slice of the active CSV.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Poll forever and emit a prediction every poll_seconds.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=DEFAULT_POLL_SECONDS,
        help="Polling interval when --watch is set (default: 300).",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.60,
        help="Training fraction (default: 0.60).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.20,
        help="Validation fraction (default: 0.20).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_cls = _resolve_model_class(args.model_cls)

    pipe = LivePipeline(
        data_dir=args.data_dir,
        model_cls=model_cls,
        model_dir=args.model_dir,
        csv_pattern=args.csv_pattern,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    if args.train:
        pipe.train_and_save()
    elif args.model_dir:
        pipe.load()
    else:
        raise SystemExit(
            "Specify --train (to fit a fresh model) or --model-dir "
            "(to load a saved one)."
        )

    if args.backtest:
        pipe.backtest_tail()

    if args.watch:
        print(
            f"Watching {pipe.data_dir} every {args.poll_seconds:.0f}s "
            f"(Ctrl-C to stop)..."
        )
        pipe.watch(poll_seconds=args.poll_seconds)
    else:
        print(pipe.predict_latest())


if __name__ == "__main__":
    main()
