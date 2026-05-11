# hft-tcpstack — Userspace TCP Stack for HFT

**Team 9:** Kevin Clemen
**Course:** CSE 40438 — High-Frequency Trading Technologies (Spring 2026)
**Instructor:** Prof. Matthew Belcher, University of Notre Dame

## Project Overview

`hft-tcpstack` is a userspace TCP/IP stack targeting ultra-low-latency trading workloads. The stack bypasses the Linux kernel via two pluggable I/O backends — a raw-socket backend for portability and a DPDK backend for kernel-bypass on supported NICs — and implements a full RFC 793 connection state machine in userspace.

Implemented mechanisms include the three-way handshake, all 10 TCP states (CLOSED, LISTEN, SYN_SENT, SYN_RECEIVED, ESTABLISHED, FIN_WAIT_1/2, TIME_WAIT, CLOSE_WAIT, LAST_ACK), full teardown on both active and passive close paths, RFC 1982 serial-number arithmetic for sequence wraparound, a retransmission timer with retry limits, sliding-window flow control, IPv4 / TCP checksum, and a trading-style message framer layered on top.

A microbenchmark (`bench/benchmark.cpp`) measures in-process round-trip latency using `rdtsc` with TSC calibration, reporting p50 / p99 / p99.9 / mean / min / max. A separate harness (`bench/bench_kernel_tcp.cpp`) provides a comparison baseline against the Linux kernel TCP stack.

The project was developed across five phases (see `PROPOSAL.md` for the original deliverables list and the git history for phase-by-phase commits).

## Repository Layout

- `PROPOSAL.md` — Original project proposal and deliverables list
- `README.md` — This file
- `Makefile` — Top-level build (`make`, `make test`, `make bench`, `make BACKEND_DPDK=1 ...`)
- `include/` — Public headers (`io/`, `msg/`, `net/`, `tcp/`)
- `src/` — Implementations
  - `io/raw_socket.cpp`, `io/dpdk_port.cpp` — I/O backends
  - `net/ethernet.cpp`, `net/ipv4.cpp`, `net/tcp.cpp` — L2/L3/L4 header handling
  - `tcp/connection.cpp`, `tcp/retransmit.cpp` — Connection state machine
  - `msg/message.cpp` — Trading-message framer
- `test/` — Unit tests (state machine, message framer, checksum, retransmit, raw socket, DPDK port)
- `bench/` — In-process and kernel-comparison latency benchmarks
- `scripts/` — Shell helpers for testing (`capture_syn.sh`, etc.)

## Build

```bash
cd 2026_spring/group9_project
make                          # Default: raw-socket backend only
make BACKEND_DPDK=1 all       # Include the DPDK backend (requires DPDK headers/libs)
```

## Tests

```bash
make test
```

Each test file produces a standalone binary in `bin/`. The DPDK backend test is excluded unless `BACKEND_DPDK=1`.

## Benchmark

```bash
make bench                          # In-process round-trip latency
make BACKEND_DPDK=1 bench           # Same + DPDK raw I/O latency
make bench-kernel                   # Kernel TCP comparison baseline (if available)
```

Latency results are written to `data/benchmark_results.txt`.

## Authors

Kevin Clemen
