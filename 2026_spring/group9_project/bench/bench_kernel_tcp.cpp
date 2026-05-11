// bench_kernel_tcp.cpp
//
// Baseline comparison: standard kernel TCP over loopback.
//
// Measures the same two metrics as bench/benchmark.cpp but via real kernel
// socket calls so the numbers can be placed next to the userspace stack.
//
// Methodology
// -----------
//   Round-trip latency
//     A server thread listens on 127.0.0.1:19090.  The client thread
//     connects, then in a tight loop does:
//       write(3 bytes) -> server recv -> server write(3 bytes) -> client recv
//     We timestamp the write() call and the return of recv() to get a full
//     kernel-mediated round-trip.  This crosses 4 syscall boundaries and
//     goes through the kernel TCP state machine, socket buffers, and the
//     loopback driver.
//
//   Throughput
//     Client sends 100 000 messages without waiting for individual ACKs,
//     server drains as fast as possible.  Measures raw kernel TCP send rate.
//
// Run:
//   make bench-kernel      (added to Makefile)
//   ./bin/bench_kernel_tcp

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

// Same rdtsc helper as the userspace benchmark for a fair apples-to-apples
// time measurement.
static inline uint64_t rdtsc() {
    uint64_t val;
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("rdtsc; shlq $32, %%rdx; orq %%rdx, %0"
                     : "=a"(val) : : "rdx");
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    val = static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
          static_cast<uint64_t>(ts.tv_nsec);
#endif
    return val;
}

static double tsc_ticks_per_ns() {
    const uint64_t t0 = rdtsc();
    struct timespec req{ 0, 10'000'000 };
    nanosleep(&req, nullptr);
    const uint64_t t1 = rdtsc();
    return static_cast<double>(t1 - t0) / 10'000'000.0;
}

struct Stats {
    double min_ns, mean_ns, p50_ns, p99_ns, p999_ns, max_ns;
};

static Stats compute_stats(std::vector<double>& s) {
    std::sort(s.begin(), s.end());
    const size_t n = s.size();
    Stats r{};
    r.min_ns  = s.front();
    r.max_ns  = s.back();
    r.mean_ns = std::accumulate(s.begin(), s.end(), 0.0) / n;
    r.p50_ns  = s[n * 50  / 100];
    r.p99_ns  = s[n * 99  / 100];
    r.p999_ns = s[n * 999 / 1000];
    return r;
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

// Disable Nagle's algorithm on a socket so small messages are sent
// immediately rather than batched.  Without this, a 3-byte write would sit
// in the Nagle buffer waiting for more data, artificially inflating latency.
static void set_nodelay(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

// Make the socket non-blocking only for the listen accept path so we can
// let the server thread exit cleanly.  Actual data I/O stays blocking.
static int make_server_socket(uint16_t port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = htons(port);
    bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    listen(fd, 1);
    return fd;
}

static int make_client_socket(uint16_t port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = htons(port);
    connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    return fd;
}

// Recv exactly n bytes or return false on EOF / error.
static bool recv_exact(int fd, uint8_t* buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = recv(fd, buf + got, n - got, 0);
        if (r <= 0) return false;
        got += static_cast<size_t>(r);
    }
    return true;
}

// -------------------------------------------------------------------------
// Round-trip latency benchmark
// -------------------------------------------------------------------------

static void bench_rtt(int warmup, int iters, double ticks_per_ns) {
    constexpr uint16_t PORT = 19090;
    std::atomic<bool> server_ready{false};

    // Server thread: ping-pong forever; exits when connection closes.
    std::thread server([&]() {
        int listener = make_server_socket(PORT);
        server_ready.store(true, std::memory_order_release);
        int conn = accept(listener, nullptr, nullptr);
        set_nodelay(conn);
        uint8_t buf[3];
        while (recv_exact(conn, buf, 3)) {
            send(conn, buf, 3, MSG_NOSIGNAL);
        }
        close(conn);
        close(listener);
    });

    // Wait for the server to be ready.
    while (!server_ready.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    // Small extra delay so listen() is definitely in place.
    struct timespec t{ 0, 1'000'000 };
    nanosleep(&t, nullptr);

    int client = make_client_socket(PORT);
    set_nodelay(client);

    const uint8_t msg[3] = { 0x01, 0x00, 0x00 }; // HEARTBEAT frame
    uint8_t reply[3];
    std::vector<double> samples;
    samples.reserve(iters);

    // Warmup.
    for (int i = 0; i < warmup; ++i) {
        send(client, msg, 3, 0);
        recv_exact(client, reply, 3);
    }

    // Timed loop.
    for (int i = 0; i < iters; ++i) {
        const uint64_t t0 = rdtsc();
        send(client, msg, 3, 0);
        recv_exact(client, reply, 3);
        const uint64_t t1 = rdtsc();
        samples.push_back(static_cast<double>(t1 - t0) / ticks_per_ns);
    }

    close(client);
    server.join();

    auto s = compute_stats(samples);
    print_stats("Kernel TCP round-trip over loopback (write + recv, 3-byte ping-pong):", s);
}

// -------------------------------------------------------------------------
// Throughput benchmark
// -------------------------------------------------------------------------

static void bench_throughput(int iters, double ticks_per_ns) {
    constexpr uint16_t PORT = 19091;
    std::atomic<bool> server_ready{false};
    std::atomic<int>  decoded{0};

    // Server thread: counts messages received.
    std::thread server([&]() {
        int listener = make_server_socket(PORT);
        server_ready.store(true, std::memory_order_release);
        int conn = accept(listener, nullptr, nullptr);
        set_nodelay(conn);
        uint8_t buf[3];
        while (recv_exact(conn, buf, 3)) {
            decoded.fetch_add(1, std::memory_order_relaxed);
        }
        close(conn);
        close(listener);
    });

    while (!server_ready.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    struct timespec t{ 0, 1'000'000 };
    nanosleep(&t, nullptr);

    int client = make_client_socket(PORT);
    set_nodelay(client);

    const uint8_t msg[3] = { 0x01, 0x00, 0x00 };

    const uint64_t t0 = rdtsc();
    for (int i = 0; i < iters; ++i) {
        send(client, msg, 3, 0);
    }
    close(client); // EOF tells server to stop
    server.join();
    const uint64_t t1 = rdtsc();

    const double elapsed_ns  = static_cast<double>(t1 - t0) / ticks_per_ns;
    const double msg_per_sec = static_cast<double>(decoded.load()) / (elapsed_ns * 1e-9);

    std::cout << "Kernel TCP throughput (HEARTBEAT, loopback):\n"
              << "  messages sent    " << iters          << '\n'
              << "  messages decoded " << decoded.load() << '\n'
              << "  total time       " << elapsed_ns / 1e6 << " ms\n"
              << "  throughput       " << static_cast<long long>(msg_per_sec)
              << " msg/s\n\n";
}

int main() {
    std::cout << "=== Kernel TCP Baseline Benchmark ===\n\n";

    const double ticks_per_ns = tsc_ticks_per_ns();
    std::cout << "TSC frequency: " << ticks_per_ns << " ticks/ns\n\n";

    constexpr int WARMUP = 1000;
    constexpr int ITERS  = 100'000;

    bench_rtt(WARMUP, ITERS, ticks_per_ns);
    bench_throughput(ITERS, ticks_per_ns);

    return 0;
}
