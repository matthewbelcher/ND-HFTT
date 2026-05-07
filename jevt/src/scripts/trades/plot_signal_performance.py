from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATION_PATH = REPO_ROOT / "data/processed/cpi_trade_validation_summary.csv"
OUTPUT_DIR = REPO_ROOT / "results/plots/cpi_trade_research"


def _save(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close(fig)


def plot_permutation_pvalues(df: pd.DataFrame) -> None:
    pvals = df[df["metric_name"].str.startswith("perm_pvalue_")].copy()
    pvals = pvals.sort_values("value", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=pvals, x="metric_name", y="value", hue="signal_name", dodge=False, ax=ax)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1.2, label="p=0.05 threshold")
    ax.set_title("Permutation P-Values by Signal Metric")
    ax.set_xlabel("Metric")
    ax.set_ylabel("p-value")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="upper right")
    _save(fig, "signal_permutation_pvalues.png")


def plot_oos_metrics(df: pd.DataFrame) -> None:
    oos = df[
        df["metric_name"].str.startswith("oos_")
        & df["split"].isin(["loocv", "walk_forward"])
    ].copy()
    oos["metric_split"] = oos["metric_name"] + " (" + oos["split"] + ")"

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=oos, x="metric_split", y="value", hue="signal_name", dodge=False, ax=ax)
    ax.axhline(0.0, color="black", linestyle="-", linewidth=1)
    ax.set_title("Out-of-Sample Metrics (LOOCV / Walk-Forward)")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="best")
    _save(fig, "signal_oos_metrics.png")


def plot_pass_fail_matrix(df: pd.DataFrame) -> None:
    signals = ["signal_a", "signal_b", "signal_c"]

    oos_pass = {
        "signal_a": False,
        "signal_b": False,
        "signal_c": False,
    }
    perm_pass = {
        "signal_a": False,
        "signal_b": False,
        "signal_c": False,
    }

    # OOS pass rule: any LOOCV OOS metric strictly positive or accuracy > 0.5.
    for signal in signals:
        sub = df[(df["signal_name"] == signal) & (df["split"].isin(["loocv", "walk_forward"]))]
        for _, row in sub.iterrows():
            metric = str(row["metric_name"])
            value = row["value"]
            if pd.isna(value):
                continue
            if "accuracy" in metric and float(value) > 0.5:
                oos_pass[signal] = True
            elif "oos_" in metric and float(value) > 0:
                oos_pass[signal] = True
            elif metric in {"pnl_120m", "accuracy_120m"} and float(value) > 0:
                oos_pass[signal] = True

    # Permutation pass rule: all permutation p-values for that signal must be < 0.05.
    for signal in signals:
        sub = df[
            (df["signal_name"] == signal)
            & (df["split"] == "permutation")
            & (df["metric_name"].str.startswith("perm_pvalue_"))
        ]
        if not sub.empty and (sub["value"] < 0.05).all():
            perm_pass[signal] = True

    matrix = pd.DataFrame(
        {
            "signal": signals,
            "oos_predictive_power": [int(oos_pass[s]) for s in signals],
            "permutation_significant": [int(perm_pass[s]) for s in signals],
        }
    )
    matrix["final_valid"] = (
        (matrix["oos_predictive_power"] == 1) & (matrix["permutation_significant"] == 1)
    ).astype(int)

    plot_df = matrix.set_index("signal")[["oos_predictive_power", "permutation_significant", "final_valid"]]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.heatmap(plot_df, annot=True, cmap="Reds", cbar=False, vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Signal Test Pass/Fail Matrix (1=Pass, 0=Fail)")
    ax.set_xlabel("Criterion")
    ax.set_ylabel("Signal")
    _save(fig, "signal_test_pass_fail_matrix.png")


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(VALIDATION_PATH)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    plot_permutation_pvalues(df)
    plot_oos_metrics(df)
    plot_pass_fail_matrix(df)

    print(f"Wrote performance plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

