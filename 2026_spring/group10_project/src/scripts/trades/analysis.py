from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def _save_plot(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def create_exploratory_plots(df: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="surprise", y="ret_5m", ax=ax)
    ax.set_title("Surprise vs 5-minute return")
    _save_plot(fig, output_dir / "surprise_vs_ret_5m.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="surprise", y="ret_120m", ax=ax)
    ax.set_title("Surprise vs 120-minute return")
    _save_plot(fig, output_dir / "surprise_vs_ret_120m.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="ofi_0_60s", y="ret_120m", ax=ax)
    ax.set_title("OFI[0,60s] vs 120-minute return")
    _save_plot(fig, output_dir / "ofi60s_vs_ret_120m.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x="surprise_category", y="ret_120m", ax=ax)
    ax.set_title("120-minute return by surprise category")
    _save_plot(fig, output_dir / "ret_120m_by_surprise_category.png")

    numeric_df = df.select_dtypes(include=["number"])
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0.0, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    _save_plot(fig, output_dir / "feature_correlation_matrix.png")

