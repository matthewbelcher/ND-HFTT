# CPI and FOMC News Reaction in BTC-USDT Futures

This project was completed for a High Frequency Trading course and studies how BTC-USDT futures react to scheduled macroeconomic news releases. The main focus was Consumer Price Index (CPI) releases, with a later extension to Federal Open Market Committee (FOMC) events.

The central research question was:

> Do scheduled macroeconomic releases create a tradable signal in BTC, and how does that signal change as execution latency increases?

## Project Overview

The project began with a simple event-study framework using 1-minute Binance BTC-USDT kline data. We aligned BTC price data around scheduled CPI release times and measured normalized returns before and after each event.

We then tested a naive macro signal:

- CPI surprise = actual CPI - forecast CPI
- Negative surprise → long BTC
- Positive surprise → short BTC
- Neutral surprise → no trade

Using this signal, we ran delay/hold backtests to study how performance changed as simulated execution latency increased.

Finally, we moved to a more granular trade-level analysis. The goal was to determine whether higher-frequency features, such as early returns, order-flow imbalance, and volatility measures, could produce a stronger signal than the simple CPI surprise rule.

## Project Phases

### 1. One-Minute Event Study

The first phase used 1-minute BTC-USDT klines because they were reliable, easy to work with, and easy to align around scheduled release times. This gave us a broad view of how BTC behaved around CPI announcements.

We created event windows around each release and plotted normalized BTC returns before and after the event.

### 2. Delay/Hold Backtesting

The second phase converted the CPI surprise into a simple trading strategy. For each event, the strategy entered after a specified delay and exited after a fixed holding period.

Tested delays:

- 0 minutes
- 1 minute
- 2 minutes
- 5 minutes
- 10 minutes
- 30 minutes

Tested holding periods:

- 1 minute
- 5 minutes
- 10 minutes
- 30 minutes
- 60 minutes

This helped us evaluate whether any signal survived realistic latency. The CPI signal was generally fragile, inconsistent across events, and sensitive to execution delay.

### 3. Trade-Level Feature Research

After reviewing the 1-minute results, we pivoted to a more granular analysis. The 1-minute bars were useful for initial exploration, but they were likely too coarse for a true HFT-style question.

In the trade-level phase, we engineered features around CPI release windows, including:

- Early returns
- Order-flow imbalance
- Volatility/range measures
- Pre-release drift
- Post-release continuation and reversal behavior

We then tested signal families using validation methods such as leave-one-out cross validation, walk-forward testing, and permutation analysis.

## Key Findings

The main conclusion was that scheduled macro releases do create volatility in BTC-USDT, but volatility does not necessarily imply a tradable directional edge.

For CPI releases:

- The naive surprise-based signal was small and inconsistent.
- Performance was highly sensitive to delay and holding period.
- The best-looking regions were not strong enough to conclude that the strategy was robust.
- Much of the relevant market reaction appeared to happen faster than 1-minute bars could capture.

For FOMC events:

- We later extended parts of the pipeline to FOMC releases.
- Some FOMC delay/hold results appeared stronger than the CPI results.
- However, the sample size was small, so these results should be treated as suggestive rather than conclusive.

## Repository Notes

Some of the original CPI CSV files and data extraction scripts were later modified to point to FOMC event data as part of the project extension. As a result, certain filenames may still reference CPI even though the current script paths or processed outputs point to FOMC events.

However, the original CPI event-study plots and backtest plots still survive in the `plots/` directory. These plots reflect the CPI-focused portion of the project and were used in the final presentation.

## Data Sources

The project used data from:

- Binance BTC-USDT klines and trade data
- Bureau of Labor Statistics CPI release timing information
- Historical actual vs. forecast CPI data from investing.com
- FOMC event data for the later project extension

## Lessons Learned

This project was a useful introduction to signal research in a high-frequency context. We began with a simple macroeconomic intuition, tested it using lower-frequency data, identified weaknesses in the initial approach, and then moved toward more granular feature development.

Although the final CPI signal was not robustly profitable, the process helped demonstrate several important ideas:

- Event alignment matters.
- Latency can eliminate apparent trading opportunities.
- Lower-frequency data can hide the true reaction window.
- Backtests can look promising with small samples, so validation is critical.
- Signal research often requires several design pivots before reaching a reliable conclusion.

## Authors

Jack Rellinger, Ty Friedman, Vinny Galassi, and Ethan Koran
