#include "io/dpdk_port.h"

#include <iostream>
#include <cstring>
#include <thread>
#include <atomic>
#include <vector>

// Phase 5: DPDKPort functional tests
//
// These tests require a DPDK-capable environment.  They are only compiled
// when BACKEND_DPDK=1 is passed to make.
//
// Running in WSL2 (no real NIC):
//   sudo make BACKEND_DPDK=1 test-phase5 \
//        DPDK_EAL_ARGS="--vdev net_tap0,iface=tap0 --no-pci --no-huge --iova-mode=va"
//
// The loopback test (test_dpdk_send_recv_loopback) sends a frame on port 0
// and reads it back through the kernel TAP interface from a second thread.
// This requires two net_tap vdevs:
//   --vdev net_tap0,iface=tap0 --vdev net_tap1,iface=tap1
// with the kernel TAP interfaces bridged:
//   ip link add br0 type bridge
//   ip link set tap0 master br0
//   ip link set tap1 master br0
//   ip link set br0 up
//
// For bare-metal with a loopback cable between two ports pass two PCI addresses.

static int g_pass = 0;
static int g_fail = 0;

static void pass(const char* name) {
    std::cout << "  PASS  " << name << '\n';
    ++g_pass;
}
static void fail(const char* name, const char* reason) {
    std::cout << "  FAIL  " << name << " -- " << reason << '\n';
    ++g_fail;
}

// Build EAL argv from the DPDK_EAL_ARGS environment variable so the test
// binary can be parameterised without recompilation.
// Falls back to WSL2-safe defaults if the variable is not set.
static std::vector<std::string> eal_args_from_env() {
    const char* env = std::getenv("DPDK_EAL_ARGS");
    std::vector<std::string> args;
    args.push_back("test_dpdk_port"); // argv[0]

    if (env) {
        // Split on spaces.
        std::string s(env);
        size_t pos = 0;
        while (pos < s.size()) {
            const size_t sp = s.find(' ', pos);
            args.push_back(s.substr(pos, sp == std::string::npos ? sp : sp - pos));
            if (sp == std::string::npos) break;
            pos = sp + 1;
        }
    } else {
        // WSL2 defaults: virtual TAP device, no real NIC, no hugepages.
        args.push_back("--vdev");
        args.push_back("net_tap0,iface=tap0");
        args.push_back("--no-pci");
        args.push_back("--no-huge");
        args.push_back("--iova-mode=va");
    }
    return args;
}

static void test_dpdk_port_init() {
    const char* name = "DPDKPort: EAL init and port start succeed";

    auto strs = eal_args_from_env();
    std::vector<char*> argv;
    for (auto& s : strs) argv.push_back(s.data());

    try {
        DPDKPort::Config cfg;
        cfg.eal_argc = static_cast<int>(argv.size());
        cfg.eal_argv = argv.data();
        cfg.port_id  = 0;

        DPDKPort port(cfg);
        pass(name);
    } catch (const std::exception& e) {
        fail(name, e.what());
    }
}

static void test_dpdk_send_frame() {
    const char* name = "DPDKPort: send_frame returns frame length";

    auto strs = eal_args_from_env();
    std::vector<char*> argv;
    for (auto& s : strs) argv.push_back(s.data());

    try {
        DPDKPort::Config cfg;
        cfg.eal_argc = static_cast<int>(argv.size());
        cfg.eal_argv = argv.data();

        DPDKPort port(cfg);

        // Minimal Ethernet frame: 14-byte header + 46 bytes padding = 60 bytes.
        uint8_t frame[60]{};
        // Destination MAC: broadcast
        std::memset(frame, 0xFF, 6);
        // Source MAC: 02:00:00:00:00:01 (locally administered)
        frame[6] = 0x02; frame[11] = 0x01;
        // EtherType: 0x0800 (IPv4)
        frame[12] = 0x08; frame[13] = 0x00;

        const int sent = port.send_frame(frame, sizeof(frame));
        if (sent != static_cast<int>(sizeof(frame))) {
            fail(name, "send_frame returned unexpected value"); return;
        }
        pass(name);
    } catch (const std::exception& e) {
        fail(name, e.what());
    }
}

static void test_dpdk_send_recv_loopback() {
    const char* name = "DPDKPort: send on port 0 received on port 1 (loopback)";

    // This test needs two vdev ports bridged at the kernel level.
    // Skip automatically if DPDK_LOOPBACK_TEST is not set.
    if (!std::getenv("DPDK_LOOPBACK_TEST")) {
        std::cout << "  SKIP  " << name
                  << " (set DPDK_LOOPBACK_TEST=1 with two bridged net_tap vdevs)\n";
        return;
    }

    auto strs = eal_args_from_env();
    std::vector<char*> argv;
    for (auto& s : strs) argv.push_back(s.data());

    try {
        DPDKPort::Config tx_cfg;
        tx_cfg.eal_argc = static_cast<int>(argv.size());
        tx_cfg.eal_argv = argv.data();
        tx_cfg.port_id  = 0;

        DPDKPort::Config rx_cfg;
        rx_cfg.eal_argc = static_cast<int>(argv.size());
        rx_cfg.eal_argv = argv.data();
        rx_cfg.port_id  = 1;

        DPDKPort tx_port(tx_cfg);
        DPDKPort rx_port(rx_cfg);

        // Payload to identify the frame.
        uint8_t send_frame[64]{};
        std::memset(send_frame, 0xFF, 6);      // dst MAC: broadcast
        send_frame[6]  = 0x02; send_frame[11] = 0x01; // src MAC
        send_frame[12] = 0x08; send_frame[13] = 0x00; // EtherType IPv4
        for (int i = 14; i < 64; ++i)
            send_frame[i] = static_cast<uint8_t>(i); // recognisable payload

        // Receive on a background thread (busy-polls until a frame arrives).
        std::atomic<bool> received{false};
        std::atomic<bool> payload_ok{false};

        std::thread rx_thread([&]() {
            uint8_t buf[2048]{};
            const int n = rx_port.recv_frame(buf, sizeof(buf));
            received = true;
            if (n == 64 && std::memcmp(buf + 14, send_frame + 14, 50) == 0)
                payload_ok = true;
        });

        // Give the RX thread a moment to enter its poll loop.
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        tx_port.send_frame(send_frame, sizeof(send_frame));

        rx_thread.join();

        if (!received.load()) { fail(name, "no frame received"); return; }
        if (!payload_ok.load()) { fail(name, "received payload does not match sent frame"); return; }
        pass(name);
    } catch (const std::exception& e) {
        fail(name, e.what());
    }
}

int main() {
    std::cout << "=== Phase 5: DPDKPort Tests ===\n\n";

    std::cout << "  -- Initialisation --\n";
    test_dpdk_port_init();

    std::cout << "\n  -- Send --\n";
    test_dpdk_send_frame();

    std::cout << "\n  -- Loopback (optional) --\n";
    test_dpdk_send_recv_loopback();

    std::cout << '\n' << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
