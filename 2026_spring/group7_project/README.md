# FPGA HFT Trading Pipeline

**Team 7:** Michael Agsam, Andrew Linares, Evan Bartek
**Course:** CSE 40438 — High-Frequency Trading Technologies (Spring 2026)
**Instructor:** Prof. Matthew Belcher, University of Notre Dame

## Project Overview

This project implements a fully hardware-based HFT trading pipeline on a DE2-115 Altera (Cyclone IV) FPGA using the Quartus synthesis toolchain. The pipeline ingests Ethernet frames via RGMII through a Marvell 88E1111 PHY, parses NDFEX `trade_summary` market-data messages in hardware (magic `GOIRISH!`, message type, signed 32-bit price), and emits a hardcoded NDFEX `NEW_ORDER` BUY UDP packet over Ethernet whenever the incoming price clears a threshold. End-to-end decision logic stays in RTL — no CPU in the hot path.

The implementation builds on the open-source LispEngineer Ethernet Repeater (used because Intel's advanced Ethernet IP cores require a paid license), extended with a custom packet-parsing finite state machine, a reserved TX buffer for the pre-built outbound packet, and a hand-assembled IPv4/UDP/NDFEX header stack with a pre-computed IP header checksum.

## Repository Layout

- `HFT Group 7 Final Project Paper.pdf` — Final 20-page report
- `README.md` — This file
- `EthernetRepeaterInstructions.md` — Architecture overview of the Ethernet repeater + step-by-step custom-packet injection guide
- `EthernetRepeat-customOrder/` — Git submodule containing the Quartus project (Verilog/SystemVerilog source). Repository: `https://github.com/ndhft/hft_group_7_FPGA`
- `send_market_update.py` — Python utility that constructs and sends an NDFEX `trade_summary` UDP packet (matching the packed C++ struct) to the FPGA for testing
- `fpga_ethernet_output.pcap` — Wireshark capture demonstrating the FPGA emitting an 84-byte BUY packet on a price > $10 trigger and correctly suppressing output on price ≤ $10
- `accelerator-client.py` — Auxiliary helper

## Build / Run

The FPGA project is built and flashed via the Intel Quartus environment. See the report's "Demonstration Setup" section for the full flow (Quartus install, project load, synthesis, Active Serial programming, manual ARP entry on the host, then `python send_market_update.py --price 15`).

```bash
# Clone with submodule
git clone --recurse-submodules <repo>

# Send a test trade-summary packet to the FPGA
python 2026_spring/group7_project/send_market_update.py
```

## Authors

Michael Agsam, Andrew Linares, Evan Bartek
