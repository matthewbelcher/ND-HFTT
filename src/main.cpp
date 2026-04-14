#include <iostream>
#include <iomanip>
#include <thread>
#include <atomic>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <stdexcept>

#include <arpa/inet.h>   // htons / ntohs
#include <sys/socket.h>  // setsockopt, SOL_SOCKET, SO_RCVTIMEO
#include <sys/time.h>    // struct timeval

#include "io/raw_socket.h"

// Print an Ethernet frame in a Wireshark-style hex+ASCII dump.
static void print_frame(const char* label, const uint8_t* buf, int len) {
    std::cout << label << " (" << len << " bytes)\n";

    // Header fields
    std::cout << "  dst MAC : ";
    for (int i = 0; i < 6; ++i)
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << static_cast<int>(buf[i]) << (i < 5 ? ":" : "");
    std::cout << '\n';

    std::cout << "  src MAC : ";
    for (int i = 6; i < 12; ++i)
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << static_cast<int>(buf[i]) << (i < 11 ? ":" : "");
    std::cout << '\n';

    uint16_t et;
    std::memcpy(&et, buf + 12, 2);
    std::cout << "  EtherType: 0x" << std::hex << std::setw(4) << std::setfill('0')
              << ntohs(et) << std::dec << '\n';

    // Hex + ASCII dump of the payload
    const uint8_t* payload = buf + 14;
    int payload_len = len - 14;
    std::cout << "  payload (" << payload_len << " bytes):\n";

    for (int row = 0; row < payload_len; row += 16) {
        std::cout << "    " << std::hex << std::setw(4) << std::setfill('0') << row << "  ";
        for (int col = 0; col < 16; ++col) {
            if (row + col < payload_len)
                std::cout << std::hex << std::setw(2) << std::setfill('0')
                          << static_cast<int>(payload[row + col]) << ' ';
            else
                std::cout << "   ";
            if (col == 7) std::cout << ' ';
        }
        std::cout << " |";
        for (int col = 0; col < 16 && row + col < payload_len; ++col) {
            char c = static_cast<char>(payload[row + col]);
            std::cout << (std::isprint(static_cast<unsigned char>(c)) ? c : '.');
        }
        std::cout << std::dec << "|\n";
    }
}

// IEEE Local Experimental Ethertype 1 — avoids confusing the kernel stack
// with a real EtherType (e.g. 0x0800 = IPv4).
static constexpr uint16_t TEST_ETHERTYPE = 0x88B5;
static constexpr size_t   ETH_HDR_LEN    = 14;  // 6 dst + 6 src + 2 ethertype
static constexpr size_t   MIN_PAYLOAD    = 46;  // pad to 60-byte minimum frame

// Phase 1 loopback smoke test.
//
// Opens two independent AF_PACKET/SOCK_RAW sockets on the loopback interface:
//   - receiver: blocks waiting for our test EtherType in a background thread
//   - sender:   builds and transmits the crafted Ethernet frame
//
// Both sockets are bound to 'lo' with ETH_P_ALL, so the receiver sees every
// frame the sender puts on the wire — exercising the full TX→RX path between
// two distinct RawSocket instances.
//
// Run as root (or with CAP_NET_RAW):
//   sudo ./bin/hft-tcpstack
int main() {
    const std::string iface = "lo";

    std::cout << "Phase 1 loopback smoke test on '" << iface << "'\n"
              << "  two independent RawSocket instances (sender + receiver)\n\n";

    try {
        // Open two sockets
        RawSocket receiver(iface);
        RawSocket sender(iface);

        // 2-second receive deadline so the receiver thread never hangs.
        struct timeval tv{};
        tv.tv_sec = 2;
        if (setsockopt(receiver.fd(), SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0)
            throw std::runtime_error("setsockopt(SO_RCVTIMEO) failed");

        // Build TX frame
        // [ dst MAC (6) | src MAC (6) | EtherType (2) | payload (46) ]
        uint8_t tx[ETH_HDR_LEN + MIN_PAYLOAD]{};

        std::memset(tx, 0xff, 6);                          // dst: broadcast
        // src bytes 6-11: stay 0x00 (loopback, no real NIC)
        const uint16_t et_be = htons(TEST_ETHERTYPE);
        std::memcpy(tx + 12, &et_be, 2);

        const char tag[] = "hft-loopback-test";
        static_assert(sizeof(tag) <= MIN_PAYLOAD, "tag too long for payload");
        std::memcpy(tx + ETH_HDR_LEN, tag, sizeof(tag));

        // Receiver thread
        // Loops reading frames from the receiver socket, ignoring anything
        // that isn't our test EtherType, until it finds the matching frame
        // or the 2-second SO_RCVTIMEO fires.
        std::atomic<bool> found{false};
        std::atomic<bool> rx_ready{false};

        std::thread rx_thread([&]() {
            uint8_t rx[2048];
            rx_ready.store(true);

            while (true) {
                int n = receiver.recv_frame(rx, sizeof(rx));
                if (n < 0)  // timeout or error
                    break;
                if (n < static_cast<int>(ETH_HDR_LEN))
                    continue;

                uint16_t rx_et;
                std::memcpy(&rx_et, rx + 12, 2);
                if (ntohs(rx_et) != TEST_ETHERTYPE)
                    continue;

                const char* pl = reinterpret_cast<const char*>(rx + ETH_HDR_LEN);
                if (std::strncmp(pl, tag, sizeof(tag) - 1) == 0) {
                    print_frame("RX (receiver socket)", rx, n);
                    std::cout << "  -> payload matched\n\n";
                    found.store(true);
                    break;
                }
            }
        });

        // Spin-wait until the receiver thread has entered recv_frame.
        // The store to rx_ready happens before the first blocking syscall,
        // so a brief yield loop is enough on any real system.
        while (!rx_ready.load())
            std::this_thread::yield();
        // One extra sleep to let the thread reach the blocking recvfrom(2).
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        // Send
        print_frame("TX (sender socket)", tx, static_cast<int>(sizeof(tx)));
        int sent = sender.send_frame(tx, sizeof(tx));
        if (sent < 0) {
            std::cerr << "  FAIL: send_frame() returned " << sent << '\n';
            rx_thread.join();
            return 1;
        }
        std::cout << "  -> sent " << sent << " bytes\n\n";

        rx_thread.join();

        if (found.load()) {
            std::cout << "  PASS\n";
            return 0;
        }
        std::cerr << "  FAIL: frame not received (timeout)\n";
        return 1;

    } catch (const std::exception& e) {
        std::cerr << "  ERROR: " << e.what() << '\n'
                  << "  (run as root or grant CAP_NET_RAW)\n";
        return 1;
    }
}
