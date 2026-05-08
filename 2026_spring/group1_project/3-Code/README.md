
## Strategy1_BasketArbitrage/

Implements the Strategy #1 thesis: if the prices of a basket of mutually
exclusive legs don't sum properly, buy all of the legs for a guaranteed payout.

Core math and analyzers:

- `basket.py` — fee model and basket evaluation (profitability math with fees and slippage).
- `normalize.py` — normalizes Kalshi orderbook responses into a common form.
- `analyze_event.py` — minimal event-level arb analyzer (the tool used for the F1 example).
- `analyze_event_arb.py` — full event arbitrage flow with order sizing.
- `analyze_universe.py` — sweeps many events at once.
- `market_finder.py` — finds candidate events/markets (close-soon filtering).
- `place_limit_orders.py` — implements the hybrid limit-order strategy

- `client.py`, `kalshi_common.py` — Kalshi REST/auth helpers used by the analyzers.

Tests:

- `tests/test_basket.py` — basket math.
- `tests/test_normalize.py` — orderbook normalization.

## Scanner/

The automated scanner looking for abitrage opportunities.

- `full_set_arb.py` — main full-set arbitrage scanner.
- `scan_closing_arb.py` — scans markets that are about to close.
- `fetch_event.py`, `fetch_market.py` — Kalshi data fetchers used by the scanner.
- `OlympicsSearch.py` — authenticated Olympics market discovery helper.
- `olympics_open_markets.jsonl` — saved Olympics market discovery output.
- `kalshi_common.py` — Kalshi auth helpers used by `OlympicsSearch.py`.
- `series_fee_cache.json` — cached series fee rates.
- `arb_found.txt`, `amount.txt` — example scanner outputs.
- `KXWOFREESKI-XTAER26MEDAL_event.json`, `KXWOFREESKI-XTAER26MEDAL-CHN_market.json`  
— sample event the scanner skipped because it has 3 winners

## Strategy2_CryptoRangeVsThreshold/

Implements Strategy #2: covering all outcome possibilities by combining YES/NO
buys across the Kalshi BTC Range (`KXBTC`) and Threshold (`KXBTCD`) markets.

- `kxbtc_parity_math.py` — bucket/threshold parity math and trade recipes.
- `kxbtc_parity_parse.py` — classifies markets as Range vs. Threshold.
- `analyze_kxbtc_parity.py` — analyzer that checks for parity arbitrage.
- `kxbtc_parity_maker_bot.py` — maker-style bot built on top of the parity analyzer.
- `basket.py`, `client.py`, `normalize.py`, `kalshi_common.py`, `kalshi_fixed_point.py`
  — shared dependencies, aligned with the latest `KalshiProjectJack` helper code.
- `tests/test_kxbtc_parity.py` — parity math tests.

## Strategy3A_BitcoinPDFModel/

Implements Strategy #3A: model the probability that the 15-minute BTC contract
resolves YES using a Probability Distribution Function with the current BTC
price as the mean and a standard deviation derived from the log returns of the
previous 200 minutes (adjusted for Kalshi's moving-average settlement).

- `15min_scanner.py` — continuously checks the Kalshi 15-minute BTC market and
compares live quotes against the PDF model, with position sizing and a simple
risk gate that blocks entering when already in a position.
- `scanner.py` — earlier/standalone version of the pricing model scanner.
- `test_order.py` — Kalshi auth + order placement helpers used by the scanner.

## Strategy3B_BitcoinSpikeHFT/

Implements Strategy #3B: exploit the fact that Kalshi's contract price tracks a
moving average of BTC, so a sharp Coinbase spike leads the Kalshi move. 

- `PriceHFT.py` — Coinbase/Binance live feed subscriber and spike-driven  
entry/exit logic.

## MarketDataTools/

General Kalshi market data utilities that were previously in `KalshiProjectJack`.

- `track.py` — websocket BBO/trade logger for one or more markets.
- `kalshi_bbo_one_market.py` — small one-market BBO websocket watcher.
- `analyze.py` — analyzer for BBO/trade CSV logs.
- `kalshi_common.py` — shared auth and orderbook helper code for these tools.

Local runtime output is kept under `logs/` and ignored by git.

## Strategy4_InsiderAnalysis/

Kalshi culture/insider-market analysis package, including its original
`pyproject.toml`, CLI package, tests, and example config/ticker files.

## Strategy5_SpotifyCharts/

Spotify chart collection and artist stream analysis scripts plus the historical
CSV inputs/outputs used by those scripts.

