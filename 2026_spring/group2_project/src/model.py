#!/usr/bin/env python3

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

from features import ALL_FEATURES, TARGET_COL

# Splitting


@dataclass
class Split:
    """Container for a chronological train/val/test split.

    Any of `X_val`/`y_val` may be None when a 2-way split is used.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame | None = None
    y_val: pd.Series | None = None
    X_test: pd.DataFrame | None = None
    y_test: pd.Series | None = None

    @property
    def has_validation(self) -> bool:
        return self.X_val is not None and self.y_val is not None

    @property
    def has_test(self) -> bool:
        return self.X_test is not None and self.y_test is not None


def make_split(
    data: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> Split:
    """
    Chronological train / val / test split (matches the notebook's 60/20/20).

    No shuffling - the first `train_frac` rows are training, the next
    `val_frac` rows are validation, and everything else is the held-out
    test set.  Pass `val_frac=0.0` to get a 2-way (train/test) split.

    Parameters
    ----------
    data       : feature DataFrame (output of build_features()).
    train_frac : fraction of windows used for training.
    val_frac   : fraction of windows used for validation.

    Returns
    -------
    Split
        Dataclass holding the three (X, y) pairs.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be in [0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must leave room for a test set")

    X = data[ALL_FEATURES]
    y = data[TARGET_COL]
    n = len(X)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    if val_frac == 0.0:
        return Split(
            X_train=X.iloc[:train_end],
            y_train=y.iloc[:train_end],
            X_test=X.iloc[train_end:],
            y_test=y.iloc[train_end:],
        )
    return Split(
        X_train=X.iloc[:train_end],
        y_train=y.iloc[:train_end],
        X_val=X.iloc[train_end:val_end],
        y_val=y.iloc[train_end:val_end],
        X_test=X.iloc[val_end:],
        y_test=y.iloc[val_end:],
    )


# Base model


@dataclass
class TrainResult:
    """Plain return type for `.train()` so callers can rely on attribute access."""

    name: str
    accuracy: float
    log_loss: float | None
    n_features: int
    features: list[str]
    importances: pd.Series | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        out = {
            "model": self.name,
            "accuracy": self.accuracy,
            "log_loss": self.log_loss,
            "n_features": self.n_features,
            "features": self.features,
        }
        if self.importances is not None:
            out["importances"] = self.importances
        return out


class BaseModel(abc.ABC):
    """
    Abstract base for every model in the project.

    Subclasses must override:
        name             - display string, e.g. "Logistic Regression".
        needs_scaler     - whether features should be StandardScaler-scaled.
        _build_estimator - return a *fresh* unfitted estimator.
        _rank_features   - given training data, return ALL_FEATURES ordered
                           by importance (most important first).

    Subclasses may override:
        _select_subset   - feature-subset search over a validation set.
                           Default behaviour: try k = 1..len(ALL_FEATURES)
                           with the top-k from _rank_features() and keep the
                           k with the highest validation accuracy.
    """

    name: str = "Base"
    needs_scaler: bool = False

    def __init__(self, **hyperparameters: Any) -> None:
        self.hyperparameters: dict[str, Any] = dict(hyperparameters)
        self.estimator: Any = None
        self.scaler: StandardScaler | None = None
        self.features: list[str] = []
        self.last_importances: pd.Series | None = None

    # hooks subclasses must implement

    @abc.abstractmethod
    def _build_estimator(self) -> Any:
        """Return a fresh, unfitted estimator with self.hyperparameters."""

    @abc.abstractmethod
    def _rank_features(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_train_scaled: np.ndarray | None,
    ) -> list[str]:
        """Order ALL_FEATURES by importance (most important first)."""

    # -shared training pipeline

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> TrainResult:
        """
        Fit the model and (optionally) pick the best feature subset on val.

        If a validation set is supplied, this runs feature-subset search
        and keeps the subset that maximises validation accuracy. If no
        validation set is supplied, it refits the previously-selected
        feature subset (or all features, if none was previously chosen)
        on the supplied training data - useful for "load model, train more".

        Parameters
        ----------
        X_train, y_train : training data.
        X_val,   y_val   : optional validation data for subset search.

        Returns
        -------
        TrainResult
        """
        if self.needs_scaler:
            self.scaler = StandardScaler().fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        else:
            X_train_scaled = None
            X_val_scaled = None

        # subset search (only if val provided)
        if X_val is not None and y_val is not None:
            self.last_importances = None
            ranked = self._rank_features(X_train, y_train, X_train_scaled)

            best_acc = -1.0
            best_features: list[str] = []
            best_estimator: Any = None
            best_probs: np.ndarray | None = None

            for k in range(1, len(ranked) + 1):
                subset = ranked[:k]
                est = self._build_estimator()
                X_tr_k, X_te_k = self._slice(
                    X_train, X_val, X_train_scaled, X_val_scaled, subset
                )
                est.fit(X_tr_k, y_train)
                acc = accuracy_score(y_val, est.predict(X_te_k))
                if acc > best_acc:
                    best_acc = acc
                    best_features = list(subset)
                    best_estimator = est
                    best_probs = est.predict_proba(X_te_k)

            self.estimator = best_estimator
            self.features = best_features
            ll = float(log_loss(y_val, best_probs)) if best_probs is not None else None

            return TrainResult(
                name=self.name,
                accuracy=best_acc,
                log_loss=ll,
                n_features=len(best_features),
                features=best_features,
                importances=self._importance_series(),
            )

        # no val: refit on supplied training data
        if not self.features:
            self.features = list(ALL_FEATURES)

        self.estimator = self._build_estimator()
        X_tr = self._project(X_train, X_train_scaled, self.features)
        self.estimator.fit(X_tr, y_train)
        preds = self.estimator.predict(X_tr)
        probs = self.estimator.predict_proba(X_tr)
        acc = accuracy_score(y_train, preds)
        ll = float(log_loss(y_train, probs))

        return TrainResult(
            name=self.name,
            accuracy=acc,
            log_loss=ll,
            n_features=len(self.features),
            features=list(self.features),
            importances=self._importance_series(),
        )

    # prediction

    def _ensure_fitted(self) -> None:
        if self.estimator is None:
            raise RuntimeError(
                f"{self.name} has not been trained yet - call .train() or "
                f".load_from() first."
            )
        if not self.features:
            raise RuntimeError(f"{self.name} has no feature subset selected.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return class predictions (0 or 1) for `X`."""
        self._ensure_fitted()
        X_proj = self._project_for_inference(X)
        return self.estimator.predict(X_proj)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(up) for each row of `X`."""
        self._ensure_fitted()
        X_proj = self._project_for_inference(X)
        return self.estimator.predict_proba(X_proj)[:, 1]

    # internal helpers

    def _slice(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_train_scaled: np.ndarray | None,
        X_val_scaled: np.ndarray | None,
        subset: list[str],
    ) -> tuple[Any, Any]:
        """Project a (train, val) pair onto the feature subset."""
        return (
            self._project(X_train, X_train_scaled, subset),
            self._project(X_val, X_val_scaled, subset),
        )

    @staticmethod
    def _project(
        X: pd.DataFrame,
        X_scaled: np.ndarray | None,
        subset: list[str],
    ) -> Any:
        """Return X projected onto `subset` - scaled if a scaled array exists."""
        if X_scaled is None:
            return X[subset]
        idx = [ALL_FEATURES.index(f) for f in subset]
        return X_scaled[:, idx]

    def _project_for_inference(self, X: pd.DataFrame) -> Any:
        """Apply the stored scaler (if any) and project onto self.features."""
        if self.scaler is not None:
            scaled = self.scaler.transform(X[ALL_FEATURES])
            idx = [ALL_FEATURES.index(f) for f in self.features]
            return scaled[:, idx]
        return X[self.features]

    def _importance_series(self) -> pd.Series | None:
        """Hook for tree-based models that produced an importance ranking."""
        return self.last_importances

    # persistence

    _META_NAME = "model_meta.json"
    _SCALER_NAME = "scaler.joblib"
    _ESTIMATOR_NAME = "estimator.joblib"

    def save(self, directory: str | Path) -> Path:
        """
        Persist the trained model to `directory`. Creates the directory if
        needed. Files written:
            model_meta.json   - class name, hyperparameters, features
            estimator.joblib  - the fitted estimator
            scaler.joblib     - only if needs_scaler is True

        Returns the directory path for chaining.
        """
        self._ensure_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        meta = {
            "class": type(self).__name__,
            "name": self.name,
            "hyperparameters": self.hyperparameters,
            "features": list(self.features),
        }
        (directory / self._META_NAME).write_text(json.dumps(meta, indent=2))
        joblib.dump(self.estimator, directory / self._ESTIMATOR_NAME)
        if self.scaler is not None:
            joblib.dump(self.scaler, directory / self._SCALER_NAME)

        print(f"Saved {self.name} -> {directory}")
        return directory

    @classmethod
    def load_from(cls, directory: str | Path) -> "BaseModel":
        """
        Reconstruct a model from a directory written by `save()`. Picks the
        right subclass based on the meta file, so it's safe to call on the
        BaseModel itself (e.g. `BaseModel.load_from(path)`).
        """
        directory = Path(directory)
        meta = json.loads((directory / cls._META_NAME).read_text())

        target_cls = MODEL_REGISTRY.get(meta["class"], cls)
        instance = target_cls(**meta.get("hyperparameters", {}))
        instance.features = list(meta.get("features", []))
        instance.estimator = joblib.load(directory / cls._ESTIMATOR_NAME)
        scaler_path = directory / cls._SCALER_NAME
        if scaler_path.exists():
            instance.scaler = joblib.load(scaler_path)

        print(f"Loaded {instance.name} <- {directory}")
        return instance


# Concrete models


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression baseline with SelectKBest (ANOVA F-score).

    Default C=0.05 - stronger regularisation than sklearn's default to keep
    coefficients small on this small financial dataset.
    """

    name = "Logistic Regression"
    needs_scaler = True

    def __init__(self, C: float = 0.05, max_iter: int = 1000) -> None:
        super().__init__(C=C, max_iter=max_iter)

    def _build_estimator(self) -> LogisticRegression:
        return LogisticRegression(**self.hyperparameters)

    def _rank_features(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_train_scaled: np.ndarray | None,
    ) -> list[str]:
        # f_classif scores are computed on the *full* feature matrix; the
        # subset search then picks the top-k by score for each k.
        scores = SelectKBest(f_classif, k="all").fit(X_train_scaled, y_train).scores_
        ranked = pd.Series(scores, index=ALL_FEATURES).sort_values(ascending=False)
        return ranked.index.tolist()


class XGBoostModel(BaseModel):
    """
    XGBoost classifier with feature-importance-based subset search.

    Hyperparameters come from the notebook:
        n_estimators=100, learning_rate=0.03, max_depth=3, subsample=0.8.
    """

    name = "XGBoost"
    needs_scaler = False

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.03,
        max_depth: int = 3,
        subsample: float = 0.8,
        eval_metric: str = "logloss",
        verbosity: int = 0,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            eval_metric=eval_metric,
            verbosity=verbosity,
        )

    def _build_estimator(self) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(**self.hyperparameters)

    def _rank_features(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_train_scaled: np.ndarray | None,
    ) -> list[str]:
        base = self._build_estimator()
        base.fit(X_train[ALL_FEATURES], y_train)
        importances = pd.Series(
            base.feature_importances_, index=ALL_FEATURES
        ).sort_values(ascending=False)
        self.last_importances = importances
        return importances.index.tolist()


class RandomForestModel(BaseModel):
    """
    Random Forest classifier with feature-importance-based subset search.
    Same iterative selection strategy as XGBoost.
    """

    name = "Random Forest"
    needs_scaler = False

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

    def _build_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(**self.hyperparameters)

    def _rank_features(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_train_scaled: np.ndarray | None,
    ) -> list[str]:
        base = self._build_estimator()
        base.fit(X_train[ALL_FEATURES], y_train)
        importances = pd.Series(
            base.feature_importances_, index=ALL_FEATURES
        ).sort_values(ascending=False)
        self.last_importances = importances
        return importances.index.tolist()


class SVMModel(BaseModel):
    """
    RBF-kernel SVM with Recursive Feature Elimination (linear surrogate).

    SVM is the slowest of the four models; pass `skip_svm=True` to
    `train_all()` for fast iteration.
    """

    name = "SVM (RBF)"
    needs_scaler = True

    def __init__(self, kernel: str = "rbf", probability: bool = True) -> None:
        super().__init__(kernel=kernel, probability=probability)

    def _build_estimator(self) -> SVC:
        return SVC(**self.hyperparameters)

    def _rank_features(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_train_scaled: np.ndarray | None,
    ) -> list[str]:
        # RFE with a linear SVM surrogate ranks all features at once. Lower
        # ranking = more important; rank 1 = selected at every k.
        rfe = RFE(
            estimator=SVC(kernel="linear"),
            n_features_to_select=1,
        )
        rfe.fit(X_train_scaled, y_train)
        ranked_idx = np.argsort(rfe.ranking_)  # ascending: rank 1 first
        return [ALL_FEATURES[i] for i in ranked_idx]


# Registry - used by BaseModel.load_from() to resurrect the right subclass.
MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "LogisticRegressionModel": LogisticRegressionModel,
    "XGBoostModel": XGBoostModel,
    "RandomForestModel": RandomForestModel,
    "SVMModel": SVMModel,
}


# Multi-model orchestration


def default_model_zoo(skip_svm: bool = False) -> list[BaseModel]:
    """All four model families, freshly instantiated with notebook defaults."""
    zoo: list[BaseModel] = [
        LogisticRegressionModel(),
        XGBoostModel(),
        RandomForestModel(),
    ]
    if not skip_svm:
        zoo.append(SVMModel())
    return zoo


def train_all(
    data: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    skip_svm: bool = False,
    save_dir: str | Path | None = None,
) -> tuple[list[BaseModel], list[TrainResult]]:
    """
    Train every model in the default zoo on a chronological split.

    Parameters
    ----------
    data       : output of build_features() - feature matrix + target.
    train_frac : training fraction (default 0.60, matches the notebook).
    val_frac   : validation fraction (default 0.20). Subset search runs on
                 this slice; pass val_frac=0.0 to skip subset search.
    skip_svm   : if True, the SVM (slowest) is omitted.
    save_dir   : if given, every trained model is saved under
                 save_dir / <class_name> after training.

    Returns
    -------
    models  : list of fitted BaseModel instances (in zoo order).
    results : list of TrainResult records (parallel to `models`).
    """
    split = make_split(data, train_frac=train_frac, val_frac=val_frac)
    print(f"Train windows : {len(split.y_train):,}")
    if split.has_validation:
        print(f"Val   windows : {len(split.y_val):,}")
        print(f"Val up rate   : {split.y_val.mean():.1%}")
    if split.has_test:
        print(f"Test  windows : {len(split.y_test):,}")
        print(f"Test up rate  : {split.y_test.mean():.1%}")
    print()

    models = default_model_zoo(skip_svm=skip_svm)
    results: list[TrainResult] = []

    for model in models:
        print(f"Training {model.name}...")
        result = model.train(
            split.X_train,
            split.y_train,
            X_val=split.X_val,
            y_val=split.y_val,
        )
        results.append(result)
        print(
            f"  Best accuracy : {result.accuracy:.4f}  "
            f"using {result.n_features} features"
        )
        print(f"  Features      : {result.features}\n")

        if save_dir is not None:
            model.save(Path(save_dir) / type(model).__name__)

    return models, results


def refit_on_train_and_val(
    model: BaseModel,
    data: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> BaseModel:
    """
    Refit `model` on the union of its train + val slices using the feature
    subset already chosen on val. Returns the same model for chaining.

    This matches the notebook's pattern of running feature selection on val,
    then retraining on train+val before evaluating on test.
    """
    if not model.features:
        raise RuntimeError(
            f"{model.name} has no feature subset selected - "
            f"call .train() with a validation set first."
        )

    split = make_split(data, train_frac=train_frac, val_frac=val_frac)
    if not split.has_validation:
        raise ValueError("refit_on_train_and_val requires val_frac > 0")

    X_full = pd.concat([split.X_train, split.X_val])
    y_full = pd.concat([split.y_train, split.y_val])

    # Reuse the existing subset; train() with no val set refits on all rows.
    model.train(X_full, y_full)
    return model


# Plots


def plot_comparison(
    results: list[TrainResult] | list[dict],
    save_path: str | Path | None = None,
) -> None:
    """
    Bar chart comparing held-out accuracy across all trained models.

    A red dashed line marks the 50 % random baseline. Bars at or above
    50 % are blue; bars below are coral.
    """
    rows = [r if isinstance(r, dict) else r.to_dict() for r in results]
    names = [r["model"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    colors = ["#185FA5" if a >= 0.5 else "#993C1D" for a in accs]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, accs, color=colors, alpha=0.85, width=0.5)

    ax.axhline(
        0.5,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="Random baseline (50 %)",
    )

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{acc:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_ylim(min(0.45, min(accs) - 0.02), max(accs) + 0.04)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_title(
        "Best accuracy per model after feature selection\n"
        "(walk-forward validation, chronological split)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    _save_or_show(fig, save_path)


def plot_feature_importance(
    result: TrainResult | dict,
    save_path: str | Path | None = None,
) -> None:
    """Horizontal bar chart of feature importances (XGB / RF only)."""
    payload = result if isinstance(result, dict) else result.to_dict()
    if "importances" not in payload or payload["importances"] is None:
        print(f"No importances available for {payload['model']} - skipping.")
        return

    imp = payload["importances"].sort_values(ascending=True)
    colors = [
        "#185FA5" if feat in payload["features"] else "#B4B2A9" for feat in imp.index
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    imp.plot(kind="barh", ax=ax, color=colors, alpha=0.85)
    ax.set_xlabel("Feature importance score", fontsize=11)
    ax.set_title(
        f"{payload['model']} - feature importance\n"
        f"(blue = in best subset, gray = excluded)",
        fontsize=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    _save_or_show(fig, save_path)


def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)
