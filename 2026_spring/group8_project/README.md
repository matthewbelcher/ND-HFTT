# Reinforcement Learning for Adaptive Optimal Execution

**Team 8:** Timothy Gunn, Malik Mashigo
**Course:** CSE 40438 — High-Frequency Trading Technologies (Spring 2026)
**Instructor:** Prof. Matthew Belcher, University of Notre Dame

## Project Overview

This project trains a Proximal Policy Optimization (PPO) agent to execute a parent order against a historical NDFEX order-book replay and compares it to TWAP and VWAP baselines on Implementation Shortfall.

The custom Gymnasium environment exposes a 7-dimensional state (inventory urgency, time urgency, spread, L1 / L3 book imbalance, normalized bid / ask liquidity) and five discrete aggression levels ranging from "do nothing" to passive bid-side post, passive ask-side post, aggressive take, and immediate market order. Each action has an associated fill probability and fraction-of-remaining inventory. The reward function penalizes per-step Implementation Shortfall and applies a terminal penalty for unexecuted inventory so the agent learns to balance execution cost against completion risk.

Training uses Stable-Baselines3 PPO with 8 parallel environments, evaluated against TWAP and VWAP across 200-episode test runs.

## Repository Layout

- `Final_Report.pdf` — Final report
- `hft_semester_project.py` — Self-contained training pipeline (exported from a Colab notebook):
  - `OrderExecutionEnv` — Custom Gymnasium environment over historical NDFEX book replay
  - TWAP and VWAP baselines simulated under the same fill model
  - `RewardLoggerCallback` and PPO training loop with `EvalCallback`
  - Evaluation, action-distribution, and reward-curve plotters

## Build / Run

The script was developed in Google Colab; expected to run with:

```bash
pip install gymnasium stable-baselines3 pandas numpy matplotlib scipy
python hft_semester_project.py
```

Input data (`rl_book_data.csv`, `rl_trade_data.csv`) must be available at the path referenced inside the script. Trained models, reward curves, and IS-distribution histograms are written to the configured save path.

## Authors

Timothy Gunn, Malik Mashigo
