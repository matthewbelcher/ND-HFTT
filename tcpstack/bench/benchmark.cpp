#include "tcp/connection.h"
#include "msg/message.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include <arpa/inet.h>
#include <time.h>

// Benchmark: in-process TCP stack latency
//
// Measures the time for a complete round-trip through the userspace TCP stack
// and message framing layer without any real network I/O.  Both endpoints run
// in the same process connected via direct function-call callbacks.
//
// This isolates the pure processing cost of:
//   encode → TCPConnection::send → TCPConnection::receive_packet
//        → MessageFramer::feed → decoded message
//
// Run:
//   make bench              # in-process benchmark (always works)
//   make BACKEND_DPDK=1 bench  # same + DPDK raw I/O latency (requires DPDK)
//
// Results are written to data/benchmark_results.txt.

// Read the hardware timestamp counter for sub-nanosecond resolution timing.
static inline uint64_t rdtsc() {
    uint64_t val;
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("rdtsc; shlq $32, %%rdx; orq %%rdx, %0"
                     : "=a"(val) : : "rdx");
#else
    // Fallback: use CLOCK_MONOTONIC_RAW.
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    val = static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
          static_cast<uint64_t>(ts.tv_nsec);
#endif
    return val;
}

// Estimate TSC ticks per nanosecond by sleeping a known duration.
static double tsc_ticks_per_ns() {
    const uint64_t t0 = rdtsc();
    struct timespec req{ 0, 10'000'000 }; // 10 ms
    nanosleep(&req, nullptr);
    const uint64_t t1 = rdtsc();
    return static_cast<double>(t1 - t0) / 10'000'000.0;
}

struct Stats {
    double p50_ns;
    double p99_ns;
    double p999_ns;
    double mean_ns;
    double min_ns;
    double max_ns;
};

static Stats compute_stats(std::vector<double>& samples_ns) {
    std::sort(samples_ns.begin(), samples_ns.end());
    const size_t n = samples_ns.size();
    Stats s{};
    s.p50_ns  = samples_ns[n * 50  / 100];
    s.p99_ns  = samples_ns[n * 99  / 100];
    s.p999_ns = samples_ns[n * 999 / 1000];
    s.min_ns  = samples_ns.front();
    s.max_ns  = samples_ns.back();
    s.mean_ns = std::accumulate(samples_ns.begin(), samples_ns.end(), 0.0) / n;
    return s;
}

static void print_stats(const char* label, const Stats& s) {
    std::cout << label << '\n'
              << "  min    " << s.min_ns   << " ns\n"
              << "  mean   " << s.mean_ns  << " ns\n"
              << "  p50    " << s.p50_ns   << " ns\n"
              << "  p99    " << s.p99_ns   << " ns\n"
              << "  p999   " << s.p999_ns  << " ns\n"
              << "  max    " << s.max_ns   << " ns\n\n";
}

// In-process round-trip benchmark.
//
// A sends a Message to B; the full callback chain (TCP send, recv, ACK) runs
// synchronously inside a.send().  B's recv_fn accumulates the application
// bytes which we then run through MessageFramer.
// Measures: encode + TCP send + TCP receive + recv_fn + framer decode.
static void bench_inprocess_roundtrip(int warmup, int iters, double ticks_per_ns) {
    std::vector<double> samples;
    samples.reserve(iters);

    std::vector<uint8_t> b_bytes;
    b_bytes.reserve(256);

    static constexpr uint32_t IP_A   = 0x0a000001u;
    static constexpr uint32_t IP_B   = 0x0a000002u;
    static constexpr uint16_t PORT_A = 54400;
    static constexpr uint16_t PORT_B = 9090;

    // Use pointers to break the circular lambda capture: each connection's
    // send_fn_ delivers to the other's receive_packet().
    TCPConnection* a_ptr = nullptr;
    TCPConnection* b_ptr = nullptr;

    TCPConnection b(htonl(IP_B), PORT_B, htonl(IP_A), PORT_A,
                    [&a_ptr](const uint8_t* buf, size_t len) {
                        if (a_ptr) a_ptr->receive_packet(buf, len);
                    },
                    [&b_bytes](const uint8_t* d, size_t n) {
                        b_bytes.insert(b_bytes.end(), d, d + n);
                    },
                    6000);

    TCPConnection a(htonl(IP_A), PORT_A, htonl(IP_B), PORT_B,
                    [&b_ptr](const uint8_t* buf, size_t len) {
                        if (b_ptr) b_ptr->receive_packet(buf, len);
                    },
                    [](const uint8_t*, size_t) {},
                    5000);

    a_ptr = &a;
    b_ptr = &b;

    b.listen();
    a.connect();

    // Build a small message once.
    Message msg{};
    msg.type        = MsgType::HEARTBEAT;
    msg.payload_len = 0;

    uint8_t wire[MSG_MAX_LEN];
    const size_t wire_len = msg_encode(msg, wire, sizeof(wire));

    MessageFramer framer;

    // Warmup.
    for (int i = 0; i < warmup; ++i) {
        b_bytes.clear();
        a.send(wire, wire_len);
        framer.reset();
        framer.feed(b_bytes.data(), b_bytes.size(), [](const Message&) {});
    }

    // Timed loop.
    for (int i = 0; i < iters; ++i) {
        b_bytes.clear();

        const uint64_t t0 = rdtsc();
        a.send(wire, wire_len);
        framer.reset();
        framer.feed(b_bytes.data(), b_bytes.size(), [](const Message&) {});
        const uint64_t t1 = rdtsc();

        samples.push_back(static_cast<double>(t1 - t0) / ticks_per_ns);
    }

    auto s = compute_stats(samples);
    print_stats("In-process stack round-trip (encode + TCP send + framer decode):", s);
}

// Throughput benchmark: how many messages per second can the stack process?
static void bench_throughput(int iters, double ticks_per_ns) {
    std::vector<uint8_t> b_bytes;
    b_bytes.reserve(iters * MSG_MAX_LEN);

    static constexpr uint32_t IP_A   = 0x0a000003u;
    static constexpr uint32_t IP_B   = 0x0a000004u;
    static constexpr uint16_t PORT_A = 54402;
    static constexpr uint16_t PORT_B = 9092;

    TCPConnection* a_ptr = nullptr;
    TCPConnection* b_ptr = nullptr;

    TCPConnection b(htonl(IP_B), PORT_B, htonl(IP_A), PORT_A,
                    [&a_ptr](const uint8_t* buf, size_t len) {
                        if (a_ptr) a_ptr->receive_packet(buf, len);
                    },
                    [&b_bytes](const uint8_t* d, size_t n) {
                        b_bytes.insert(b_bytes.end(), d, d + n);
                    },
                    6002);

    TCPConnection a(htonl(IP_A), PORT_A, htonl(IP_B), PORT_B,
                    [&b_ptr](const uint8_t* buf, size_t len) {
                        if (b_ptr) b_ptr->receive_packet(buf, len);
                    },
                    [](const uint8_t*, size_t) {},
                    5002);

    a_ptr = &a;
    b_ptr = &b;
    b.listen();
    a.connect();

    Message msg{};
    msg.type        = MsgType::HEARTBEAT;
    msg.payload_len = 0;
    uint8_t wire[MSG_MAX_LEN];
    const size_t wire_len = msg_encode(msg, wire, sizeof(wire));

    // Send all messages then drain with the framer.
    const uint64_t t0 = rdtsc();
    for (int i = 0; i < iters; ++i)
        a.send(wire, wire_len);

    int decoded = 0;
    MessageFramer framer;
    framer.feed(b_bytes.data(), b_bytes.size(),
                [&decoded](const Message&) { ++decoded; });
    const uint64_t t1 = rdtsc();

    const double elapsed_ns = static_cast<double>(t1 - t0) / ticks_per_ns;
    const double msg_per_sec = static_cast<double>(decoded) / (elapsed_ns * 1e-9);

    std::cout << "Throughput (HEARTBEAT, in-process):\n"
              << "  messages sent    " << iters   << '\n'
              << "  messages decoded " << decoded << '\n'
              << "  total time       " << elapsed_ns / 1e6 << " ms\n"
              << "  throughput       " << static_cast<long long>(msg_per_sec)
              << " msg/s\n\n";
}

#ifdef BACKEND_DPDK
#include "io/dpdk_port.h"
#include <vector>
#include <cstring>
#include <rte_lcore.h>

// DPDK raw frame send latency (one-way; no loopback partner needed).
// Measures the time from send_frame() call to return; indicates TX descriptor
// acquisition + mbuf copy + rte_eth_tx_burst overhead.
static void bench_dpdk_tx_latency(int warmup, int iters, double ticks_per_ns) {
    auto strs_env = []() -> std::vector<std::string> {
        const char* env = std::getenv("DPDK_EAL_ARGS");
        std::vector<std::string> a;
        a.push_back("benchmark");
        if (env) {
            std::string s(env);
            size_t pos = 0;
            while (pos < s.size()) {
                const size_t sp = s.find(' ', pos);
                a.push_back(s.substr(pos, sp == std::string::npos ? sp : sp - pos));
                if (sp == std::string::npos) break;
                pos = sp + 1;
            }
        } else {
            a.push_back("-l");
            a.push_back("0"); // Pin to lcore 0
            a.push_back("--vdev");
            a.push_back("net_tap0,iface=tap0");
            a.push_back("--no-pci");
            a.push_back("--no-huge");
            a.push_back("--iova-mode=va");
        }
        return a;
    }();

    std::vector<char*> argv;
    for (auto& s : strs_env) argv.push_back(s.data());

    DPDKPort::Config cfg;
    cfg.eal_argc = static_cast<int>(argv.size());
    cfg.eal_argv = argv.data();

    DPDKPort port(cfg);

    // Thread pinning for benchmark accuracy.
    if (rte_lcore_id() == LCORE_ID_ANY) {
        std::cout << "Warning: not running on a DPDK lcore. Latency may be jittery.\n";
    } else {
        std::cout << "Running on DPDK lcore " << rte_lcore_id() << "\n";
    }

    uint8_t frame[64]{};
    std::memset(frame, 0xFF, 6);        // dst MAC broadcast
    frame[6]  = 0x02; frame[11] = 0x01; // src MAC
    frame[12] = 0x08; frame[13] = 0x00; // EtherType IPv4

    std::vector<double> samples;
    samples.reserve(iters);

    for (int i = 0; i < warmup; ++i)
        port.send_frame(frame, sizeof(frame));

    for (int i = 0; i < iters; ++i) {
        const uint64_t t0 = rdtsc();
        port.send_frame(frame, sizeof(frame));
        const uint64_t t1 = rdtsc();
        samples.push_back(static_cast<double>(t1 - t0) / ticks_per_ns);
    }

    auto s = compute_stats(samples);
    print_stats("DPDK send_frame latency (mbuf alloc + copy + tx_burst):", s);
}
#endif

int main() {
    std::cout << "=== HFT TCP Stack Benchmark ===\n\n";

    const double ticks_per_ns = tsc_ticks_per_ns();
    std::cout << "TSC frequency: " << ticks_per_ns << " ticks/ns\n\n";

    constexpr int WARMUP = 1000;
    constexpr int ITERS  = 100'000;

    bench_inprocess_roundtrip(WARMUP, ITERS, ticks_per_ns);
    bench_throughput(ITERS, ticks_per_ns);

#ifdef BACKEND_DPDK
    std::cout << "-- DPDK I/O latency --\n\n";
    try {
        bench_dpdk_tx_latency(WARMUP, ITERS, ticks_per_ns);
    } catch (const std::exception& e) {
        std::cout << "DPDK bench skipped: " << e.what() << '\n';
    }
#endif

    return 0;
}
