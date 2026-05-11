Group: JEVT
Members: Vinny Galassi, Ty Friedman, Ethan Koran, Jack Rellinger

# Project Description

Our project studied whether scheduled macroeconomic news releases create a tradable signal in the BTC-USDT market, with a particular focus on CPI releases and a brief extension to FOMC announcements. The motivating question was not only whether Bitcoin reacts to macro news, but also how quickly that reaction is incorporated into price. Because this was a high-frequency trading course, latency became central to the project: even if a signal exists at the moment of release, it may disappear by the time a slower trader can observe, process, and execute on it.

We approached the project in three phases.

### Phase 1: Event study using 1-minute BTC-USDT klines
The first phase used Binance 1-minute BTC-USDT candlestick data around CPI release times. This gave us a clean and manageable way to align price data around scheduled events. CPI releases occur at known times, 8:30 AM ET, about halfway through each month, which made it possible to center each event window around the release and measure normalized returns before and after the announcement. The goal of this initial phase was to create a way to measure how BTC prices react to CPI releases and evaluate whether a trading strategy could profit from that reaction.

The core signal was based on CPI surprise:
CPI surprise = actual CPI - forecast CPI

The economic intuition was simple. If inflation came in lower than expected, that should be risk-on and bullish for Bitcoin, so the signal would go long BTC. If inflation came in higher than expected, that should be risk-off and bearish for Bitcoin, so the signal would go short BTC. A zero surprise generated no trade.

The 1-minute event plots showed that BTC often moved noticeably around release times, but the direction and timing were not consistent across events. The return paths show some events with sharp post-release moves, while others reversed or drifted after the first reaction. This suggested that CPI releases were associated with volatility, but not necessarily with a stable directional edge.

### Phase 2: Naïve backtesting and latency sensitivity
The second phase turned the CPI surprise idea into a simple backtest. For each event, we generated a signal, entered after a specified delay, held for a fixed period, and measured the return. We tested delays of 0, 1, 2, 5, 10, and 30 minutes, and holding periods of 1, 5, 10, 30, and 60 minutes.

This let us directly test the latency question: if CPI contains a signal, does profitability decay as execution becomes slower?

The backtest plots showed that the naïve CPI signal was fragile. The return-versus-delay and win-rate-versus-delay plots show that performance varied significantly across hold periods, and the heatmap showed that there was no uniformly profitable delay/hold region. The conclusion was that the immediate 1-minute signal was small, sensitive to delay, and inconsistent across events. The best-looking region appeared to be around a 5–10 minute delay with a 5–10 minute holding period, with average returns around 0.0007, but this was not strong enough to conclude that the strategy was reliably profitable.

#### Later Extension of Phase 2: FOMC
In addition to our CPI analysis, we wanted to also stretch this initial idea to a brief examination of FOMC releases, as FED decisions may have a stronger impact on BTC-USDT. We replaced parts of some of our klines scripts to instead analyze windows coinciding with FOMC releases. The FOMC analysis does appear to produce some stronger-looking delay/hold combinations than the CPI analysis, but the result should be treated carefully. The current processed backtest file labeled cpi_backtest_results.csv is actually based on FOMC events, and only three FOMC events generated nonzero trades in that output. The strongest average-return cells were positive, including a 5-minute delay / 5-minute hold region and a 30-minute delay / 60-minute hold region, but because these are based on such a small number of events, I would describe this as “suggestive” rather than conclusive. The brief FOMC extension appeared more promising in the 1-minute backtest plots, but it needs more data before being treated as a robust signal.

### Phase 3: Granular trade-level signal research
The third phase moved from 1-minute bars to trade-by-trade analysis. This was the main design pivot of the project. The 1-minute data was useful for getting started, but it was too coarse to answer the true HFT question. If most of the price discovery happens in the first milliseconds or seconds after the release, then a 1-minute candle hides the most important part of the market reaction.

In the granular phase, we built features around CPI release windows using trade-level data. Instead of relying only on the CPI surprise itself, we tried to measure market behavior around the release: order-flow imbalance, early returns, volatility, price movement thresholds, pre-release drift, and post-release continuation or reversal. We then tested whether these features could generate stronger signals than the original CPI surprise rule.

The trade-level research used more realistic validation methods, including leave-one-out cross validation and permutation testing. This was important because the dataset was small and the risk of overfitting was high. The results were not strong: the signal families did not pass both out-of-sample performance and permutation significance. 

This supported the broader conclusion of the project: BTC may react sharply to CPI releases, but the exploitable directional signal is either extremely short-lived, already competed away by faster participants, or too inconsistent to capture with the features we tested. HFT bots appear to flood the market within the first 100 ms, most direction-setting happens very quickly, and CPI does not appear to have a reliable longer-term effect on Bitcoin price.

# Challenges Overcome
The largest challenge we overcame was deciding how to do signal research in the first place. At the beginning, the project seemed like a straightforward event study: collect CPI releases, calculate the surprise, align BTC prices around the event, and test whether the market moved in the expected direction. However, once we started looking at the results, we realized that signal research is less about finding one obvious relationship and more about building a careful process for testing many possible explanations without fooling ourselves. We had to decide what counted as a signal, what timeframe mattered, how to handle latency, and how to separate real predictive value from noise.

# Design Pivot
Our biggest design pivot came from moving from a zoomed-out analysis to a more granular one. We started with 1-minute klines because the data was robust, easy to clean, and easy to align across events. That was the right place to begin because it gave us a broad understanding of how BTC behaved around CPI releases. But after inspecting individual event plots and running the delay/hold backtests, we saw that 1-minute data was probably too coarse. The market reaction often happened immediately around the release, and by the time a full minute had passed, much of the signal may already have been incorporated into price. This pushed us toward trade-level feature engineering, where we could look at order flow, early returns, volatility, and continuation/reversal behavior over much shorter horizons.

# Learning Experiences
One of the main learning experiences was that even a “failed” trading signal can still teach a lot about how real signal development works. Parts of our approach were probably misguided at times, especially when we expected a simple macro surprise variable to translate cleanly into profitable BTC trades. But the overall research process became more realistic as the project developed. We began with a simple, interpretable hypothesis, tested it on clean lower-frequency data, identified its weaknesses, and then moved toward higher-frequency features and stricter validation. That progression mirrors how signal research might work in a real trading role: start with a broad economic intuition, use simple analysis to understand the shape of the opportunity, and then move into more granular HFT-style feature development only if the early evidence justifies it.

# Conclusion
The final takeaway is that CPI and FOMC releases do create moments of volatility in BTC-USDT, but volatility alone is not the same as a tradable edge. The CPI signal was not robust, and while the FOMC extension looked more promising in some preliminary plots, the sample was too small to draw a firm conclusion. The project ultimately showed us that latency, data frequency, and validation design are just as important as the original trading idea.