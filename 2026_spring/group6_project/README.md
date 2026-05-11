# Ghostbook — NDFEX-Compatible Replay Engine and Matching Engine

**Team 6:** Jack O'Connor, Brian Rabideau
**Course:** CSE 40438 — High-Frequency Trading Technologies (Spring 2026)
**Instructor:** Prof. Matthew Belcher, University of Notre Dame

## Project Overview

Ghostbook is a drop-in exchange simulator that speaks the same wire protocol as the NDFEX matching engine, so a trading system can run **identically** against:

- the live NDFEX exchange,
- a Ghostbook `ReplayEngine` driven by a captured NDFEX market-data feed, or
- a fresh `MatchingEngine` instance from a clean book.

The trading client only swaps a `ConnectionSpec` — no recompilation required. This enables deterministic, bit-reproducible strategy iteration with wire semantics matching production.

The project also defines a clean native Ghostbook protocol (FIX/ITCH-inspired: framed messages with sequence numbers, CRC32, session lifecycle, Logon/HeartBeat/SequenceReset) alongside the NDFEX-compatibility layer.

## Repository Layout

- `cpp/`
  - `CMakeLists.txt`, `Makefile` — Build entry points
  - `include/ghostbook/` — Public headers
    - `gateway.h`, `protocol.h` — Native Ghostbook protocol
    - `ndfex/protocol.h` — NDFEX wire-protocol structs (packed, magic `GOIRISH!`)
  - `src/`
    - `gateway/` — TCP gateway (native) + NDFEX UDP/TCP gateway
    - `matching_engine/` — RFC-style matching engine with Day / IOC / FOK / PostOnly TIFs
    - `replay_engine/` — Captured-feed replay engine sharing the matching-engine interface
  - `tests/` — Matching-engine, gateway, UDP/TCP integration, replay-gateway tests, and a microbenchmark
    - `scenarios/` — Deterministic JSON scenario fixtures for queue-position and latency regressions
- `submodules/team6-trading-system/` — Team 6's trading system (used to demonstrate Ghostbook integration; submodule)
- `Makefile` — Top-level convenience wrapper

## Build

```bash
cd 2026_spring/group6_project/cpp && make BUILD_TYPE=Release
```

## Tests

```bash
cd 2026_spring/group6_project/cpp && make test
```

## Benchmark

```bash
cd 2026_spring/group6_project/cpp && make bench
```

## Authors

Jack O'Connor, Brian Rabideau
