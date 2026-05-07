from pathlib import Path
PLOTS_DIR = Path("../../results/plots")

## Return vs. Delay
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = REPO_ROOT / "data/processed/cpi_backtest_results.csv"


df = pd.read_csv(RESULTS_PATH)

summary = (
    df.groupby(["delay_minutes", "hold_minutes"])
    .agg(avg_return=("trade_return", "mean"))
    .reset_index()
)

plt.figure(figsize=(8, 5))

for hold in sorted(summary["hold_minutes"].unique()):
    subset = summary[summary["hold_minutes"] == hold]
    plt.plot(subset["delay_minutes"], subset["avg_return"], label=f"Hold={hold}m")

plt.xlabel("Delay (minutes)")
plt.ylabel("Average Return")
plt.title("Return vs Execution Delay")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / f"klines_backtest_return_vs_delay.png")
plt.show()

## Heatmap
import seaborn as sns

pivot = summary.pivot(index="hold_minutes", columns="delay_minutes", values="avg_return")

plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn", center=0)

plt.title("Return Heatmap (Hold vs Delay)")
plt.xlabel("Delay (minutes)")
plt.ylabel("Hold (minutes)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / f"klines_backtest_heatmap.png")
plt.show()

## Win summary
win_summary = (
    df.groupby(["delay_minutes", "hold_minutes"])
    .agg(win_rate=("trade_return", lambda s: (s > 0).mean()))
    .reset_index()
)

plt.figure(figsize=(8, 5))

for hold in sorted(win_summary["hold_minutes"].unique()):
    subset = win_summary[win_summary["hold_minutes"] == hold]
    plt.plot(subset["delay_minutes"], subset["win_rate"], label=f"Hold={hold}m")

plt.xlabel("Delay (minutes)")
plt.ylabel("Win Rate")
plt.title("Win Rate vs Delay")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / f"klines_backtest_win_summary.png")
plt.show()

## Distribution of returns
plt.figure(figsize=(8, 5))
plt.hist(df["trade_return"], bins=30)
plt.title("Distribution of Trade Returns")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(PLOTS_DIR / f"klines_backtest_distr_returns.png")
plt.show()