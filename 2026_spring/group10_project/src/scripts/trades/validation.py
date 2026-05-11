from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import LOOCV_MIN_TRAIN_SIZE, PERMUTATION_ITERATIONS, RANDOM_SEED, WALK_FORWARD_INITIAL_TRAIN


@dataclass(frozen=True)
class ValidationMetric:
    signal_name: str
    metric_name: str
    value: float
    split: str


def _safe_mean(values: list[float]) -> float:
    arr = np.array(values, dtype="float64")
    if arr.size == 0:
        return float("nan")
    if np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return float(1 - (ss_res / ss_tot))


def _fit_linear_predict(train: pd.DataFrame, test: pd.DataFrame, target_col: str) -> np.ndarray:
    x_train = train[["abs_surprise", "abs_ofi_60s"]].to_numpy(dtype="float64")
    x_test = test[["abs_surprise", "abs_ofi_60s"]].to_numpy(dtype="float64")
    y_train = train[target_col].to_numpy(dtype="float64")
    x_train_aug = np.column_stack([x_train, np.ones(len(x_train))])
    x_test_aug = np.column_stack([x_test, np.ones(len(x_test))])
    coeffs, *_ = np.linalg.lstsq(x_train_aug, y_train, rcond=None)
    return x_test_aug @ coeffs


def _signal_c_direction_from_train(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    mapping: dict[str, float] = {}
    for regime, regime_df in train.groupby("signal_c_regime"):
        mean_ret = float(regime_df["ret_120m"].mean())
        mapping[regime] = 1.0 if mean_ret >= 0 else -1.0
    default_dir = 1.0 if float(train["ret_120m"].mean()) >= 0 else -1.0
    return np.array([mapping.get(regime, default_dir) for regime in test["signal_c_regime"]], dtype="float64")


def walk_forward_signal_a(df: pd.DataFrame, initial_train: int = WALK_FORWARD_INITIAL_TRAIN) -> list[ValidationMetric]:
    ordered = df.sort_values("release_time_et").reset_index(drop=True)
    accuracies: list[float] = []
    pnls: list[float] = []
    for split_idx in range(initial_train, len(ordered)):
        test = ordered.iloc[split_idx]
        acc = test["signal_a_correct_120m"]
        pnl = test["signal_a_pnl_120m"]
        accuracies.append(float(acc) if pd.notna(acc) else np.nan)
        pnls.append(float(pnl) if pd.notna(pnl) else np.nan)
    return [
        ValidationMetric("signal_a", "accuracy_120m", _safe_mean(accuracies), "walk_forward"),
        ValidationMetric("signal_a", "pnl_120m", _safe_mean(pnls), "walk_forward"),
    ]


def loocv_signal_a(df: pd.DataFrame) -> list[ValidationMetric]:
    ordered = df.sort_values("release_time_et").reset_index(drop=True)
    if len(ordered) < LOOCV_MIN_TRAIN_SIZE:
        return []
    accuracies = ordered["signal_a_correct_120m"].astype("float64").tolist()
    pnls = ordered["signal_a_pnl_120m"].astype("float64").tolist()
    return [
        ValidationMetric("signal_a", "accuracy_120m", _safe_mean(accuracies), "loocv"),
        ValidationMetric("signal_a", "pnl_120m", _safe_mean(pnls), "loocv"),
    ]


def evaluate_signal_b_oos(
    df: pd.DataFrame, target_col: str, metric_suffix: str, initial_train: int = WALK_FORWARD_INITIAL_TRAIN
) -> list[ValidationMetric]:
    ordered = df.sort_values("release_time_et").reset_index(drop=True)
    work = ordered[["release_time_et", "abs_surprise", "abs_ofi_60s", target_col]].dropna().reset_index(drop=True)
    if len(work) < LOOCV_MIN_TRAIN_SIZE:
        return []

    # LOOCV predictions
    loocv_true: list[float] = []
    loocv_pred: list[float] = []
    for i in range(len(work)):
        train = work.drop(index=i)
        test = work.iloc[[i]]
        pred = _fit_linear_predict(train, test, target_col=target_col)
        loocv_pred.append(float(pred[0]))
        loocv_true.append(float(test.iloc[0][target_col]))

    loocv_true_arr = np.array(loocv_true, dtype="float64")
    loocv_pred_arr = np.array(loocv_pred, dtype="float64")

    # Walk-forward predictions
    wf_true: list[float] = []
    wf_pred: list[float] = []
    for split_idx in range(initial_train, len(work)):
        train = work.iloc[:split_idx]
        test = work.iloc[[split_idx]]
        pred = _fit_linear_predict(train, test, target_col=target_col)
        wf_pred.append(float(pred[0]))
        wf_true.append(float(test.iloc[0][target_col]))

    wf_true_arr = np.array(wf_true, dtype="float64")
    wf_pred_arr = np.array(wf_pred, dtype="float64")

    return [
        ValidationMetric("signal_b", f"oos_r2_{metric_suffix}", _safe_r2(loocv_true_arr, loocv_pred_arr), "loocv"),
        ValidationMetric("signal_b", f"oos_corr_{metric_suffix}", _safe_corr(loocv_true_arr, loocv_pred_arr), "loocv"),
        ValidationMetric("signal_b", f"oos_r2_{metric_suffix}", _safe_r2(wf_true_arr, wf_pred_arr), "walk_forward"),
        ValidationMetric("signal_b", f"oos_corr_{metric_suffix}", _safe_corr(wf_true_arr, wf_pred_arr), "walk_forward"),
    ]


def compute_signal_b_fits(df: pd.DataFrame) -> list[ValidationMetric]:
    work = df[["abs_surprise", "abs_ofi_60s", "signal_b_target_vol"]].dropna().copy()
    if work.empty:
        return []
    corr = float(work["abs_surprise"].corr(work["signal_b_target_vol"]))

    x1 = work["abs_surprise"].to_numpy()
    y = work["signal_b_target_vol"].to_numpy()
    r2_base = float(np.corrcoef(x1, y)[0, 1] ** 2) if len(work) > 1 else np.nan

    x2 = np.column_stack([work["abs_surprise"].to_numpy(), work["abs_ofi_60s"].to_numpy(), np.ones(len(work))])
    coeffs, *_ = np.linalg.lstsq(x2, y, rcond=None)
    y_hat = x2 @ coeffs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_aug = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else np.nan

    return [
        ValidationMetric("signal_b", "corr_abs_surprise_vs_vol", corr, "in_sample"),
        ValidationMetric("signal_b", "r2_abs_surprise_only", r2_base, "in_sample"),
        ValidationMetric("signal_b", "r2_surprise_plus_ofi", r2_aug, "in_sample"),
    ]


def evaluate_signal_c(df: pd.DataFrame) -> list[ValidationMetric]:
    work = df[["signal_c_regime", "signal_c_pnl_120m"]].dropna()
    if work.empty:
        return []
    metrics: list[ValidationMetric] = []
    for regime, g in work.groupby("signal_c_regime"):
        metrics.append(
            ValidationMetric("signal_c", f"avg_pnl_120m_{regime}", float(g["signal_c_pnl_120m"].mean()), "in_sample")
        )
    return metrics


def evaluate_signal_c_oos(df: pd.DataFrame, initial_train: int = WALK_FORWARD_INITIAL_TRAIN) -> list[ValidationMetric]:
    ordered = df.sort_values("release_time_et").reset_index(drop=True)
    work = ordered[["release_time_et", "signal_c_regime", "ret_120m"]].dropna().reset_index(drop=True)
    if len(work) < LOOCV_MIN_TRAIN_SIZE:
        return []

    # LOOCV predictions
    loocv_pnl: list[float] = []
    loocv_acc: list[float] = []
    for i in range(len(work)):
        train = work.drop(index=i)
        test = work.iloc[[i]]
        pred_dir = _signal_c_direction_from_train(train, test)[0]
        true_ret = float(test.iloc[0]["ret_120m"])
        loocv_pnl.append(float(pred_dir * true_ret))
        loocv_acc.append(float(np.sign(true_ret) == pred_dir))

    # Walk-forward predictions
    wf_pnl: list[float] = []
    wf_acc: list[float] = []
    for split_idx in range(initial_train, len(work)):
        train = work.iloc[:split_idx]
        test = work.iloc[[split_idx]]
        pred_dir = _signal_c_direction_from_train(train, test)[0]
        true_ret = float(test.iloc[0]["ret_120m"])
        wf_pnl.append(float(pred_dir * true_ret))
        wf_acc.append(float(np.sign(true_ret) == pred_dir))

    return [
        ValidationMetric("signal_c", "oos_pnl_120m", _safe_mean(loocv_pnl), "loocv"),
        ValidationMetric("signal_c", "oos_accuracy_120m", _safe_mean(loocv_acc), "loocv"),
        ValidationMetric("signal_c", "oos_pnl_120m", _safe_mean(wf_pnl), "walk_forward"),
        ValidationMetric("signal_c", "oos_accuracy_120m", _safe_mean(wf_acc), "walk_forward"),
    ]


def permutation_test_signal_a(
    df: pd.DataFrame, iterations: int = PERMUTATION_ITERATIONS, random_seed: int = RANDOM_SEED
) -> list[ValidationMetric]:
    rng = np.random.default_rng(random_seed)
    work = df.copy()
    real_metric = float(np.nanmean(work["signal_a_pnl_120m"].to_numpy(dtype="float64")))
    permuted: list[float] = []
    for _ in range(iterations):
        shuffled = work.copy()
        shuffled["surprise"] = rng.permutation(shuffled["surprise"].to_numpy())
        shuffled["signal_a_direction_perm"] = np.sign(shuffled["ofi_0_60s"]).where(
            np.sign(shuffled["ofi_0_60s"]) == np.sign(shuffled["surprise"]), 0
        )
        pnl_perm = shuffled["signal_a_direction_perm"] * shuffled["ret_120m"]
        permuted.append(float(np.nanmean(pnl_perm.to_numpy(dtype="float64"))))
    permuted_arr = np.array(permuted)
    p_value = float((np.sum(permuted_arr >= real_metric) + 1) / (len(permuted_arr) + 1))
    return [
        ValidationMetric("signal_a", "perm_pvalue_pnl_120m", p_value, "permutation"),
        ValidationMetric("signal_a", "real_pnl_120m", real_metric, "permutation"),
    ]


def permutation_test_signal_b(
    df: pd.DataFrame, target_col: str, metric_suffix: str, iterations: int = PERMUTATION_ITERATIONS, random_seed: int = RANDOM_SEED
) -> list[ValidationMetric]:
    rng = np.random.default_rng(random_seed)
    ordered = df.sort_values("release_time_et").reset_index(drop=True)
    work = ordered[["abs_surprise", "abs_ofi_60s", target_col]].dropna().reset_index(drop=True)
    if len(work) < LOOCV_MIN_TRAIN_SIZE:
        return []

    # Real LOOCV R2
    real_true: list[float] = []
    real_pred: list[float] = []
    for i in range(len(work)):
        train = work.drop(index=i)
        test = work.iloc[[i]]
        pred = _fit_linear_predict(train, test, target_col=target_col)
        real_pred.append(float(pred[0]))
        real_true.append(float(test.iloc[0][target_col]))
    real_r2 = _safe_r2(np.array(real_true, dtype="float64"), np.array(real_pred, dtype="float64"))

    perm_values: list[float] = []
    for _ in range(iterations):
        shuffled = work.copy()
        shuffled[target_col] = rng.permutation(shuffled[target_col].to_numpy())
        perm_true: list[float] = []
        perm_pred: list[float] = []
        for i in range(len(shuffled)):
            train = shuffled.drop(index=i)
            test = shuffled.iloc[[i]]
            pred = _fit_linear_predict(train, test, target_col=target_col)
            perm_pred.append(float(pred[0]))
            perm_true.append(float(test.iloc[0][target_col]))
        perm_values.append(_safe_r2(np.array(perm_true, dtype="float64"), np.array(perm_pred, dtype="float64")))

    perm_arr = np.array(perm_values, dtype="float64")
    p_value = float((np.sum(perm_arr >= real_r2) + 1) / (len(perm_arr) + 1))
    return [
        ValidationMetric("signal_b", f"perm_pvalue_oos_r2_{metric_suffix}", p_value, "permutation"),
        ValidationMetric("signal_b", f"real_oos_r2_{metric_suffix}", real_r2, "permutation"),
    ]


def permutation_test_signal_c(
    df: pd.DataFrame, iterations: int = PERMUTATION_ITERATIONS, random_seed: int = RANDOM_SEED
) -> list[ValidationMetric]:
    rng = np.random.default_rng(random_seed)
    ordered = df.sort_values("release_time_et").reset_index(drop=True)
    work = ordered[["signal_c_regime", "ret_120m"]].dropna().reset_index(drop=True)
    if len(work) < LOOCV_MIN_TRAIN_SIZE:
        return []

    # Real LOOCV metrics
    real_pnl: list[float] = []
    real_acc: list[float] = []
    for i in range(len(work)):
        train = work.drop(index=i)
        test = work.iloc[[i]]
        pred_dir = _signal_c_direction_from_train(train, test)[0]
        true_ret = float(test.iloc[0]["ret_120m"])
        real_pnl.append(float(pred_dir * true_ret))
        real_acc.append(float(np.sign(true_ret) == pred_dir))
    real_pnl_mean = _safe_mean(real_pnl)
    real_acc_mean = _safe_mean(real_acc)

    perm_pnl_vals: list[float] = []
    perm_acc_vals: list[float] = []
    for _ in range(iterations):
        shuffled = work.copy()
        shuffled["ret_120m"] = rng.permutation(shuffled["ret_120m"].to_numpy())
        perm_pnl: list[float] = []
        perm_acc: list[float] = []
        for i in range(len(shuffled)):
            train = shuffled.drop(index=i)
            test = shuffled.iloc[[i]]
            pred_dir = _signal_c_direction_from_train(train, test)[0]
            true_ret = float(test.iloc[0]["ret_120m"])
            perm_pnl.append(float(pred_dir * true_ret))
            perm_acc.append(float(np.sign(true_ret) == pred_dir))
        perm_pnl_vals.append(_safe_mean(perm_pnl))
        perm_acc_vals.append(_safe_mean(perm_acc))

    perm_pnl_arr = np.array(perm_pnl_vals, dtype="float64")
    perm_acc_arr = np.array(perm_acc_vals, dtype="float64")
    p_value_pnl = float((np.sum(perm_pnl_arr >= real_pnl_mean) + 1) / (len(perm_pnl_arr) + 1))
    p_value_acc = float((np.sum(perm_acc_arr >= real_acc_mean) + 1) / (len(perm_acc_arr) + 1))

    return [
        ValidationMetric("signal_c", "perm_pvalue_oos_pnl_120m", p_value_pnl, "permutation"),
        ValidationMetric("signal_c", "real_oos_pnl_120m", real_pnl_mean, "permutation"),
        ValidationMetric("signal_c", "perm_pvalue_oos_accuracy_120m", p_value_acc, "permutation"),
        ValidationMetric("signal_c", "real_oos_accuracy_120m", real_acc_mean, "permutation"),
    ]


def run_validations(df: pd.DataFrame) -> pd.DataFrame:
    metrics: list[ValidationMetric] = []
    metrics.extend(walk_forward_signal_a(df))
    metrics.extend(loocv_signal_a(df))
    metrics.extend(compute_signal_b_fits(df))
    metrics.extend(evaluate_signal_b_oos(df, target_col="range_0_120m", metric_suffix="range_120m"))
    metrics.extend(evaluate_signal_b_oos(df, target_col="realized_vol_5m_mean", metric_suffix="realized_vol_5m_mean"))
    metrics.extend(evaluate_signal_c(df))
    metrics.extend(evaluate_signal_c_oos(df))
    metrics.extend(permutation_test_signal_a(df))
    metrics.extend(
        permutation_test_signal_b(df, target_col="range_0_120m", metric_suffix="range_120m")
    )
    metrics.extend(
        permutation_test_signal_b(df, target_col="realized_vol_5m_mean", metric_suffix="realized_vol_5m_mean")
    )
    metrics.extend(permutation_test_signal_c(df))
    return pd.DataFrame([m.__dict__ for m in metrics])

