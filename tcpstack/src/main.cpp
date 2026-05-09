#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <random>
#include <stdexcept>

#include <arpa/inet.h>   // htons / htonl / inet_pton
#include <sys/socket.h>

#include "io/raw_socket.h"
#include "net/ethernet.h"
#include "net/ipv4.h"
#include "net/tcp.h"

// Hex+ASCII dump of a buffer, Wireshark-style.
static void hex_dump(const uint8_t* buf, size_t len) {
    for (size_t row = 0; row < len; row += 16) {
        std::cout << "    " << std::hex << std::setw(4) << std::setfill('0') << row << "  ";
        for (size_t col = 0; col < 16; ++col) {
            if (row + col < len)
                std::cout << std::hex << std::setw(2) << std::setfill('0')
                          << static_cast<int>(buf[row + col]) << ' ';
            else
                std::cout << "   ";
            if (col == 7) std::cout << ' ';
        }
        std::cout << " |";
        for (size_t col = 0; col < 16 && row + col < len; ++col) {
            char c = static_cast<char>(buf[row + col]);
            std::cout << (std::isprint(static_cast<unsigned char>(c)) ? c : '.');
        }
        std::cout << std::dec << "|\n";
    }
}

// Build a full Eth/IPv4/TCP SYN frame from 127.0.0.1:src_port to 127.0.0.1:dst_port.
// Returns the total frame length written into `out`.
static size_t build_syn_frame(uint8_t*  out,
                              size_t    out_len,
                              uint16_t  src_port,
                              uint16_t  dst_port,
                              uint32_t  seq_num)
{
    // Ethernet header — loopback has no real MAC, all zeros is accepted.
    EthernetHeader eth{};
    std::memset(eth.dst_mac, 0x00, 6);
    std::memset(eth.src_mac, 0x00, 6);
    eth.ethertype = ETHERTYPE_IPV4;

    const size_t eth_n = ethernet_encode(eth, out, out_len);
    if (eth_n == 0) return 0;

    // IPv4 header — protocol = TCP, DF flag, no fragmentation.
    IPv4Header ip{};
    ip.version_ihl    = (IPV4_VERSION << 4) | IPV4_IHL_MIN;
    ip.tos            = 0;
    ip.total_length   = IPV4_HDR_LEN + TCP_HDR_LEN;   // no payload on SYN
    ip.identification = 0x1234;
    ip.flags_frag     = 0x4000;                       // DF set, offset = 0
    ip.ttl            = IPV4_DEFAULT_TTL;
    ip.protocol       = IPPROTO_TCP_V;
    // 127.0.0.1 — inet_pton writes directly in network byte order.
    if (::inet_pton(AF_INET, "127.0.0.1", &ip.src_ip) != 1) return 0;
    if (::inet_pton(AF_INET, "127.0.0.1", &ip.dst_ip) != 1) return 0;

    const size_t ip_n = ipv4_encode(ip, out + eth_n, out_len - eth_n);
    if (ip_n == 0) return 0;

    // TCP header — SYN only, no payload, no options.
    TCPHeader tcp{};
    tcp.src_port             = src_port;
    tcp.dst_port             = dst_port;
    tcp.seq_num              = seq_num;
    tcp.ack_num              = 0;
    tcp.data_offset_reserved = tcp_pack_data_offset(5);
    tcp.flags                = TCP_FLAG_SYN;
    tcp.window               = 65535;
    tcp.urgent_ptr           = 0;

    const size_t tcp_n = tcp_encode(ip, tcp, nullptr, 0,
                                    out + eth_n + ip_n,
                                    out_len - eth_n - ip_n);
    if (tcp_n == 0) return 0;

    return eth_n + ip_n + tcp_n;
}

// Phase 2 smoke test.
//
// Builds a raw Ethernet+IPv4+TCP SYN directed at 127.0.0.1:8080 using our own
// packet construction code, then transmits it via AF_PACKET/SOCK_RAW on 'lo'.
//
// To observe the packet on the wire, run in another terminal before launching:
//   sudo tcpdump -i lo -nn -xx 'tcp and port 8080'
//
// Expected tcpdump output: one of our SYNs at src_port → 127.0.0.1.8080,
// followed by the kernel's RST (nothing is listening on 8080). The RST is
// proof that our packet was well-formed enough for the kernel's TCP stack
// to accept and respond to it.
//
// Run as root (or with CAP_NET_RAW):
//   sudo ./bin/hft-tcpstack
int main() {
    const std::string iface   = "lo";
    const uint16_t    dst_port = 8080;

    // Seed src_port and initial seq_num from the monotonic clock so successive
    // runs don't confuse kernel TIME_WAIT state from a previous SYN.
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 rng(static_cast<uint32_t>(now));
    const uint16_t src_port = static_cast<uint16_t>(0xC000 | (rng() & 0x3FFF));  // 49152-65535
    const uint32_t seq_num  = rng();

    std::cout << "=== Phase 2 smoke test: raw TCP SYN on 'lo' ===\n\n"
              << "  target  : 127.0.0.1:" << dst_port << '\n'
              << "  src port: "           << src_port << '\n'
              << "  seq num : 0x"         << std::hex << seq_num << std::dec << "\n\n"
              << "  to capture:  sudo tcpdump -i lo -nn -xx 'tcp and port " << dst_port << "'\n\n";

    try {
        RawSocket tx(iface);

        uint8_t frame[ETHERNET_HDR_LEN + IPV4_HDR_LEN + TCP_HDR_LEN];
        const size_t frame_len = build_syn_frame(frame, sizeof(frame),
                                                 src_port, dst_port, seq_num);
        if (frame_len == 0) {
            std::cerr << "  ERROR: frame construction failed\n";
            return 1;
        }

        std::cout << "TX frame (" << frame_len << " bytes):\n";
        hex_dump(frame, frame_len);
        std::cout << '\n';

        const int sent = tx.send_frame(frame, frame_len);
        if (sent < 0) {
            std::cerr << "  ERROR: send_frame() returned " << sent
                      << " (errno " << errno << ": " << std::strerror(errno) << ")\n";
            return 1;
        }

        std::cout << "  -> sent " << sent << " bytes on '" << iface << "'\n"
                  << "     check tcpdump for the captured SYN (and kernel RST reply)\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "  ERROR: " << e.what() << '\n'
                  << "  (run as root or grant CAP_NET_RAW)\n";
        return 1;
    }
}
