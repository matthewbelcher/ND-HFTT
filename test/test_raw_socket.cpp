#include "io/raw_socket.h"

#include <iostream>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <chrono>
#include <utility>

#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <linux/if_packet.h>
#include <net/ethernet.h>

// Mini test runner
static int g_pass = 0;
static int g_fail = 0;

static void pass(const char* name) {
    std::cout << "  PASS  " << name << '\n';
    ++g_pass;
}

static void fail(const char* name, const char* reason) {
    std::cout << "  FAIL  " << name << " — " << reason << '\n';
    ++g_fail;
}

// Returns true if the process holds CAP_NET_RAW (can open AF_PACKET sockets).
static bool has_cap_net_raw() {
    int fd = ::socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (fd < 0) return false;
    ::close(fd);
    return true;
}

// Phase 1 tests

// A non-existent interface name must cause a std::runtime_error at
// construction time (ioctl SIOCGIFINDEX will fail even with CAP_NET_RAW).
static void test_bad_iface() {
    const char* name = "RawSocket: throws on bad interface name";
    try {
        RawSocket s("nonexistent_iface_xyz99");
        fail(name, "expected std::runtime_error, none thrown");
    } catch (const std::runtime_error&) {
        pass(name);
    }
}

// Construction on loopback must succeed and leave the socket in a valid state.
static void test_open_lo() {
    const char* name = "RawSocket: opens loopback interface";
    try {
        RawSocket s("lo");
        pass(name);
    } catch (const std::exception& e) {
        fail(name, e.what());
    }
}

// fd() must be a valid (non-negative) file descriptor after construction.
static void test_fd_valid() {
    const char* name = "RawSocket: fd() >= 0 after open";
    try {
        RawSocket s("lo");
        if (s.fd() >= 0)
            pass(name);
        else
            fail(name, "fd() returned negative value");
    } catch (const std::exception& e) {
        fail(name, e.what());
    }
}

// if_index() must be positive — the kernel assigns ifindex starting at 1.
static void test_if_index_positive() {
    const char* name = "RawSocket: if_index() > 0 after open";
    try {
        RawSocket s("lo");
        if (s.if_index() > 0)
            pass(name);
        else
            fail(name, "if_index() returned non-positive value");
    } catch (const std::exception& e) {
        fail(name, e.what());
    }
}

// After move construction the new object must own the fd and the source must
// be invalidated (fd == -1).
static void test_move_ctor() {
    const char* name = "RawSocket: move constructor transfers ownership";
    try {
        RawSocket a("lo");
        int original_fd = a.fd();

        RawSocket b(std::move(a));

        if (a.fd() != -1) {
            fail(name, "source fd not invalidated after move");
            return;
        }
        if (b.fd() != original_fd) {
            fail(name, "destination fd does not match original");
            return;
        }
        pass(name);
    } catch (const std::exception& e) {
        fail(name, e.what());
    }
}

// Move assignment must transfer fd and invalidate the source.
static void test_move_assign() {
    const char* name = "RawSocket: move assignment transfers ownership";
    try {
        RawSocket a("lo");
        RawSocket b("lo");
        int a_fd = a.fd();

        b = std::move(a);

        if (a.fd() != -1) {
            fail(name, "source fd not invalidated after move-assign");
            return;
        }
        if (b.fd() != a_fd) {
            fail(name, "destination fd does not match source's original fd");
            return;
        }
        pass(name);
    } catch (const std::exception& e) {
        fail(name, e.what());
    }
}

// Full TX→RX loopback roundtrip on 'lo': the sender socket transmits a
// crafted Ethernet frame, the receiver socket reads it back and verifies
// the payload matches.
static void test_loopback_roundtrip() {
    const char* name = "RawSocket: loopback send/recv roundtrip on 'lo'";

    // IEEE Local Experimental EtherType — won't be confused with real traffic.
    static constexpr uint16_t TEST_ET  = 0x88B5;
    static constexpr size_t   HDR_LEN  = 14;   // 6 dst + 6 src + 2 ethertype
    static constexpr size_t   PAY_LEN  = 46;   // min payload for 60-byte frame
    static const char         TAG[]    = "hft-test-roundtrip";
    static_assert(sizeof(TAG) <= PAY_LEN, "tag too long");

    try {
        RawSocket receiver("lo");
        RawSocket sender("lo");

        // 2-second receive deadline so the test never hangs.
        struct timeval tv{};
        tv.tv_sec = 2;
        if (::setsockopt(receiver.fd(), SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0)
            throw std::runtime_error("setsockopt(SO_RCVTIMEO) failed");

        // Build TX frame: broadcast dst, zero src, test EtherType, tag payload.
        uint8_t tx[HDR_LEN + PAY_LEN]{};
        std::memset(tx, 0xff, 6);
        const uint16_t et_be = htons(TEST_ET);
        std::memcpy(tx + 12, &et_be, 2);
        std::memcpy(tx + HDR_LEN, TAG, sizeof(TAG));

        std::atomic<bool> found{false};
        std::atomic<bool> rx_ready{false};

        std::thread rx_thread([&]() {
            uint8_t rx[2048];
            rx_ready.store(true);

            while (true) {
                int n = receiver.recv_frame(rx, sizeof(rx));
                if (n < 0) break;  // timeout or error
                if (n < static_cast<int>(HDR_LEN)) continue;

                uint16_t rx_et;
                std::memcpy(&rx_et, rx + 12, 2);
                if (ntohs(rx_et) != TEST_ET) continue;

                const char* pl = reinterpret_cast<const char*>(rx + HDR_LEN);
                if (std::strncmp(pl, TAG, sizeof(TAG) - 1) == 0) {
                    found.store(true);
                    break;
                }
            }
        });

        while (!rx_ready.load()) std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        int sent = sender.send_frame(tx, sizeof(tx));
        rx_thread.join();

        if (sent < 0) {
            fail(name, "send_frame() returned < 0");
            return;
        }
        if (!found.load()) {
            fail(name, "frame not received within 2s timeout");
            return;
        }
        pass(name);

    } catch (const std::exception& e) {
        fail(name, e.what());
    }
}

// main
int main() {
    std::cout << "=== Phase 1: I/O Backend Tests ===\n\n";

    if (!has_cap_net_raw()) {
        std::cout << "  SKIP  All tests — CAP_NET_RAW not available "
                     "(run as root or grant the capability)\n\n"
                     "0 passed, 0 failed, all skipped\n";
        // Exit 0: the missing capability is an environment issue, not a code bug.
        return 0;
    }

    test_bad_iface();
    test_open_lo();
    test_fd_valid();
    test_if_index_positive();
    test_move_ctor();
    test_move_assign();
    test_loopback_roundtrip();

    std::cout << '\n'
              << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
