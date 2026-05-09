"""Load the offline-trained augmented 10s GBM (ensemble_with_btc_mom.py).

Augmented vs baseline:
  - Baseline (ensemble_gbm_10s_live.joblib):    16 features, AUC 0.643
  - Augmented (ensemble_gbm_10s_btcmom.joblib): 19 features, AUC 0.674
  - BTC momentum (btc_mom_5s/10s/accel) account for 43% of feature importance.
  - BTC filter (|btc_mom_10s| > 0.0001, agrees with pred): AUC → 0.705, acc 0.659.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from kalshi_live.features_live import FEATS

# Points to the augmented model trained in ensemble_with_btc_mom.py.
# Fall back to the original model if the augmented one is missing.
DEFAULT_MODEL_RELPATH = Path("output") / "ensemble_gbm_10s_btcmom.joblib"
FALLBACK_MODEL_RELPATH = Path("output") / "ensemble_gbm_10s_live.joblib"


def load_10s_model(
    data_dir: Path,
    model_path: Path | None = None,
) -> GradientBoostingClassifier:
    if model_path is None:
        augmented = data_dir / DEFAULT_MODEL_RELPATH
        model_path = augmented if augmented.exists() else data_dir / FALLBACK_MODEL_RELPATH

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing trained model at {model_path}. "
            "Run `python ensemble_with_btc_mom.py` to train the augmented model, "
            "or `python ensemble_profitability.py` for the original baseline."
        )

    bundle: Any = joblib.load(model_path)

    # Support both save formats:
    #   dict format: {"model": gbm, "feature_names": [...]}  (preferred)
    #   raw format:  GradientBoostingClassifier directly
    if isinstance(bundle, dict):
        model = bundle["model"]
        saved_names: List[str] | None = bundle.get("feature_names")
    else:
        model = bundle
        saved_names = None

    if saved_names is not None and list(saved_names) != FEATS:
        raise ValueError(
            f"Saved model feature_names do not match features_live.FEATS.\n"
            f"  Saved:    {saved_names}\n"
            f"  Expected: {FEATS}\n"
            "Re-run ensemble_with_btc_mom.py after any feature changes."
        )
    if not isinstance(model, GradientBoostingClassifier):
        raise TypeError(f"Expected GradientBoostingClassifier, got {type(model)}")

    print(f"[model] Loaded: {model_path.name}  "
          f"({model.n_estimators} trees, {model.n_features_in_} features)")
    return model


def predict_proba_up(model: GradientBoostingClassifier, X: np.ndarray) -> float:
    return float(model.predict_proba(X)[0, 1])
