#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from model import BaseModel

# Strategies


def _flat_bet(prob_up: float, bankroll: float, bet_fraction: float) -> float:
    """Always bet a flat fraction of bankroll. Probability is ignored."""
    del prob_up  # explicitly unused - strategies share a signature
    return bankroll * bet_fraction


def _long_only_bet(prob_up: float, bankroll: float, bet_fraction: float) -> float:
    """Only bet when P(up) > 0.5; size scaled by P(up)."""
    if prob_up <= 0.5:
        return 0.0
    return bankroll * bet_fraction * prob_up


def _confidence_bet(
    prob_up: float,
    bankroll: float,
    bet_fraction: float,
    skip_zone: tuple[float, float] = (0.35, 0.65),
) -> float:
    """
    Skip the indecisive zone, then size by distance from 0.5 (x 2 so a
    fully-confident prediction bets the full `bet_fraction`).
    """
    low, high = skip_zone
    if low <= prob_up <= high:
        return 0.0
    return bankroll * bet_fraction * abs(prob_up - 0.5) * 2.0


STRATEGIES = {
    "flat": _flat_bet,
    "long_only": _long_only_bet,
    "confidence": _confidence_bet,
}


# Result type


@dataclass
class BacktestResult:
    """Self-describing result of a single bankroll simulation."""

    name: str
    strategy: str
    bankroll_start: float
    bankroll_end: float
    history: list[float]
    trades: pd.DataFrame  # actual / predicted / prob / bet / pnl
    accuracy: float
    correct: int
    n: int
    p_value: float

    @property
    def net_pnl(self) -> float:
        return self.bankroll_end - self.bankroll_start

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def summary(self) -> str:
        return (
            f"{self.name:25s}  "
            f"strategy={self.strategy:10s}  "
            f"acc={self.accuracy:.4f} ({self.correct}/{self.n})  "
            f"end=${self.bankroll_end:>10,.2f}  "
            f"pnl=${self.net_pnl:>+10,.2f}  "
            f"p={self.p_value:.4f}"
        )


# Simulation


def simulate_bankroll(
    actuals: pd.Series | np.ndarray | Iterable[int],
    predictions: pd.Series | np.ndarray | Iterable[int],
    probabilities: pd.Series | np.ndarray | Iterable[float] | None = None,
    *,
    name: str = "model",
    strategy: str = "long_only",
    bankroll: float = 1000.0,
    bet_fraction: float = 0.10,
    skip_zone: tuple[float, float] = (0.35, 0.65),
) -> BacktestResult:
    """
    Replay a sequence of predictions through a betting strategy.

    Parameters
    ----------
    actuals       : ground-truth labels (0 = down, 1 = up).
    predictions   : the model's class predictions.
    probabilities : the model's P(up). Required for long_only and
                    confidence strategies.
    name          : label for printing / plotting.
    strategy      : one of STRATEGIES - "flat", "long_only", "confidence".
    bankroll      : starting bankroll.
    bet_fraction  : fraction of bankroll wagered on a full-confidence bet.
    skip_zone     : passed through to the confidence strategy.

    Returns
    -------
    BacktestResult
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Options: {list(STRATEGIES)}")

    actuals = pd.Series(actuals).reset_index(drop=True)
    predictions = pd.Series(predictions).reset_index(drop=True)
    if probabilities is None:
        if strategy != "flat":
            raise ValueError(f"strategy='{strategy}' requires `probabilities`.")
        probabilities = pd.Series([np.nan] * len(actuals))
    probabilities = pd.Series(probabilities).reset_index(drop=True)

    if not (len(actuals) == len(predictions) == len(probabilities)):
        raise ValueError("actuals, predictions, probabilities must align.")

    bet_fn = STRATEGIES[strategy]
    history: list[float] = [bankroll]
    trades_log: list[dict] = []

    for actual, pred, prob in zip(actuals, predictions, probabilities):
        if strategy == "confidence":
            bet = bet_fn(prob, bankroll, bet_fraction, skip_zone=skip_zone)
        else:
            bet = bet_fn(prob, bankroll, bet_fraction)

        if bet == 0.0:
            pnl = 0.0
        elif pred == actual:
            pnl = bet
        else:
            pnl = -bet
        bankroll += pnl
        history.append(bankroll)

        trades_log.append(
            {
                "actual": int(actual),
                "predicted": int(pred),
                "prob": float(prob),
                "bet": float(bet),
                "pnl": float(pnl),
                "bankroll": float(bankroll),
            }
        )

    n = len(actuals)
    correct = int((predictions.values == actuals.values).sum())
    accuracy = correct / n if n else 0.0
    p_value = binomtest(correct, n, p=0.5, alternative="greater").pvalue if n else 1.0

    return BacktestResult(
        name=name,
        strategy=strategy,
        bankroll_start=history[0],
        bankroll_end=bankroll,
        history=history,
        trades=pd.DataFrame(trades_log),
        accuracy=accuracy,
        correct=correct,
        n=n,
        p_value=p_value,
    )


def backtest_model(
    model: BaseModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    strategy: str = "long_only",
    bankroll: float = 1000.0,
    bet_fraction: float = 0.10,
    skip_zone: tuple[float, float] = (0.35, 0.65),
) -> BacktestResult:
    """
    Convenience: take a fitted model and run `simulate_bankroll` on its
    predictions for the supplied held-out set.
    """
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    return simulate_bankroll(
        actuals=y_test,
        predictions=preds,
        probabilities=probs,
        name=model.name,
        strategy=strategy,
        bankroll=bankroll,
        bet_fraction=bet_fraction,
        skip_zone=skip_zone,
    )


# Plots


def plot_bankroll(
    result: BacktestResult,
    save_path: str | Path | None = None,
) -> None:
    """Equity curve for a single backtest run."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(result.history, color="#185FA5", linewidth=1.4)
    ax.axhline(
        result.bankroll_start,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label=f"Starting bankroll  (${result.bankroll_start:,.0f})",
    )
    ax.set_xlabel("Trade #", fontsize=11)
    ax.set_ylabel("Bankroll ($)", fontsize=11)
    ax.set_title(
        f"{result.name} - {result.strategy} strategy\n"
        f"final ${result.bankroll_end:,.2f}  "
        f"(acc {result.accuracy:.4f}, p={result.p_value:.4f})",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)


def print_summary(results: list[BacktestResult]) -> pd.DataFrame:
    """Print a one-line summary per result and return them as a DataFrame."""
    rows = [
        {
            "Model": r.name,
            "Strategy": r.strategy,
            "Accuracy": f"{r.accuracy:.4f}",
            "Correct": f"{r.correct}/{r.n}",
            "Final $": f"{r.bankroll_end:,.2f}",
            "Net PnL": f"{r.net_pnl:+,.2f}",
            "p-value": f"{r.p_value:.4f}",
            "Sig.": "yes" if r.significant else "no",
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    sep = "=" * 90
    print(f"\n{sep}\n  BACKTEST SUMMARY\n{sep}")
    print(df.to_string(index=False))
    print(sep + "\n")
    return df
