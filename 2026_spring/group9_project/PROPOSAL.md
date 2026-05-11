# hft-tcpstack

# Project Proposal

## Userspace TCP Stackm

---

## Project Overview

The objective of this project is to design and implement a high-performance userspace TCP stack optimized for
ultra-low-latency financial trading systems. Traditional Linux TCP networking introduces latency due to system calls,
kernel-user context switching, interrupt handling, and general-purpose congestion control. In high-frequency trading (
HFT), where microseconds directly impact profitability, this overhead becomes a critical limitation.

This project will reimplement core TCP functionality in userspace using a kernel-bypass framework such as **AF_XDP** or
**DPDK**. Rather than relying on standard socket APIs (`send()` and `recv()`), the system will manually construct and
process Ethernet, IPv4, and TCP headers, manage connection state, and implement retransmission and flow control logic.

The final product will be a C/C++ library capable of establishing a TCP connection and exchanging structured
trading-style protocol messages without using the Linux kernel’s TCP stack.

---

## Technical Background

TCP operates at Layer 4 of the OSI model and provides reliable, ordered, congestion-controlled communication. The Linux
implementation is robust and general-purpose but not optimized for extreme low-latency environments.

To reduce overhead, modern trading systems use kernel-bypass networking techniques that allow direct access to NIC
buffers. This project focuses on recreating core TCP mechanisms in userspace, including:

* Three-way handshake (SYN, SYN-ACK, ACK)
* Sequence number tracking
* Acknowledgment processing
* Retransmission logic
* Sliding-window flow control
* Connection teardown


---


1. **Userspace TCP Library**

    * Establishes TCP connections without standard socket APIs
    * Handles sequence numbers and acknowledgments
    * Implements retransmission timers
    * Supports structured trading-style message exchange

2. **Performance Evaluation**

    * Latency comparison against standard Linux TCP and Onload
    * Round-trip timing benchmarks
    * Analysis of performance improvements

3. **Final Report**

    * Architecture and design decisions
    * Implementation challenges
    * Debugging strategies
    * Performance results


---

