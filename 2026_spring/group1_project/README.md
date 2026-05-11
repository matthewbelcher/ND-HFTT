# Kalshi Prediction Market Inefficiency Investigation

**Team 1:** Colby Whitehouse, Jack Decker, Will Sullivan
**Course:** CSE 40438 — High-Frequency Trading Technologies (Spring 2026)
**Instructor:** Prof. Matthew Belcher, University of Notre Dame

## Project Overview

This project investigates pricing inefficiencies on Kalshi, a CFTC-regulated prediction market. The team developed and evaluated five distinct strategy families spanning fixed-outcome basket arbitrage, BTC range-vs-threshold parity, short-duration crypto pricing, culture/event-flow signal analysis, and Spotify-chart-driven signal research.

## Repository Layout

- `1-Proposal/` — Original project proposal (PDF)
- `2-Presentation/` — Final presentation slides (PDF)
- `3-Code/` — Strategy code, organized by approach:
  - `Strategy1_BasketArbitrage/` — Fixed-outcome basket arbitrage with depth-aware fee modeling
  - `Strategy1_Scanner/` — Automated arbitrage scanner across Kalshi events
  - `Strategy2_CryptoRangeVsThreshold/` — KXBTC range vs threshold parity analyzer and maker bot
  - `Strategy3A_BitcoinPDFModel/` — 15-minute BTC contract pricing via Gaussian PDF
  - `Strategy3B_BitcoinSpikeHFT/` — Coinbase/Binance spike-driven BTC trading prototype
  - `Strategy4_InsiderAnalysis/` — Installable Python package for Kalshi culture/event flow analysis
  - `Strategy5_SpotifyCharts/` — Spotify chart collection and artist-stream features
  - `MarketDataTools/` — Kalshi websocket BBO/trade logger and analyzer
- `4-Report/Trading Report.pdf` — Final report documenting methodology and results

## Build / Run

Per-strategy details and execution instructions are in `3-Code/README.md`. Strategy 4 is an installable package (see its `pyproject.toml`).

## Authors

Colby Whitehouse, Jack Decker, Will Sullivan
