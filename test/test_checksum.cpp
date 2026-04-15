#include "net/ethernet.h"
#include "net/ipv4.h"
#include "net/tcp.h"

#include <iostream>
#include <cstring>
#include <cstdint>
#include <vector>

#include <arpa/inet.h>  // htons / htonl / ntohs

// Mini test runner (same style as test_raw_socket.cpp).
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

// =========================================================================
// IPv4 checksum
// =========================================================================

// Known-good IPv4 header from widely-cited networking references
// (also verifiable in Wireshark by pasting the hex into a saved pcap):
//   45 00 00 3c 1c 46 40 00 40 06 [cksum] ac 10 0a 63 ac 10 0a 0c
// Expected header checksum = 0xb1e6.
static void test_ipv4_checksum_known_vector() {
    const char* name = "ipv4_checksum: matches known 0xb1e6 reference vector";

    uint8_t hdr[20] = {
        0x45, 0x00, 0x00, 0x3c, 0x1c, 0x46, 0x40, 0x00,
        0x40, 0x06, 0x00, 0x00,                          // cksum field zeroed
        0xac, 0x10, 0x0a, 0x63,                          // src 172.16.10.99
        0xac, 0x10, 0x0a, 0x0c                           // dst 172.16.10.12
    };

    const uint16_t got_be = ipv4_checksum(hdr, sizeof(hdr));
    const uint16_t got    = ntohs(got_be);
    if (got == 0xb1e6) {
        pass(name);
    } else {
        std::cout << "    expected 0xb1e6, got 0x" << std::hex << got << std::dec << '\n';
        fail(name, "wrong checksum");
    }
}

// Verify property: a header with a correct checksum installed, when re-summed,
// yields 0. This is the standard way receivers validate the checksum.
static void test_ipv4_checksum_verify_property() {
    const char* name = "ipv4_checksum: verify property (sum with cksum in place == 0)";

    IPv4Header ip{};
    ip.version_ihl     = (IPV4_VERSION << 4) | IPV4_IHL_MIN;
    ip.tos             = 0;
    ip.total_length    = 40;
    ip.identification  = 0x1234;
    ip.flags_frag      = 0;
    ip.ttl             = IPV4_DEFAULT_TTL;
    ip.protocol        = IPPROTO_TCP_V;
    ip.src_ip          = htonl(0x0a000001);  // 10.0.0.1
    ip.dst_ip          = htonl(0x0a000002);  // 10.0.0.2

    uint8_t buf[IPV4_HDR_LEN];
    const size_t n = ipv4_encode(ip, buf, sizeof(buf));
    if (n != IPV4_HDR_LEN) {
        fail(name, "ipv4_encode returned wrong length");
        return;
    }

    // Recompute over the encoded bytes — result must be zero.
    const uint16_t verify = ipv4_checksum(buf, IPV4_HDR_LEN);
    if (verify == 0) {
        pass(name);
    } else {
        fail(name, "non-zero verify sum");
    }
}

// =========================================================================
// IPv4 encode/decode round-trip
// =========================================================================
static void test_ipv4_roundtrip() {
    const char* name = "ipv4: encode/decode round-trip";

    IPv4Header ip{};
    ip.version_ihl     = (IPV4_VERSION << 4) | IPV4_IHL_MIN;
    ip.tos             = 0x10;
    ip.total_length    = 1500;
    ip.identification  = 0xbeef;
    ip.flags_frag      = 0x4000;             // DF set, offset = 0
    ip.ttl             = 64;
    ip.protocol        = IPPROTO_TCP_V;
    ip.src_ip          = htonl(0xc0a80001);  // 192.168.0.1
    ip.dst_ip          = htonl(0xc0a80002);  // 192.168.0.2

    uint8_t buf[IPV4_HDR_LEN];
    if (ipv4_encode(ip, buf, sizeof(buf)) != IPV4_HDR_LEN) {
        fail(name, "encode failed");
        return;
    }

    IPv4Header decoded{};
    if (!ipv4_decode(buf, sizeof(buf), decoded)) {
        fail(name, "decode failed");
        return;
    }

    if (decoded.version_ihl    != ip.version_ihl    ||
        decoded.tos            != ip.tos            ||
        decoded.total_length   != ip.total_length   ||
        decoded.identification != ip.identification ||
        decoded.flags_frag     != ip.flags_frag     ||
        decoded.ttl            != ip.ttl            ||
        decoded.protocol       != ip.protocol       ||
        decoded.src_ip         != ip.src_ip         ||
        decoded.dst_ip         != ip.dst_ip)
    {
        fail(name, "decoded fields differ from original");
        return;
    }

    pass(name);
}

// Reject decodes of headers with wrong version or IHL < 5.
static void test_ipv4_decode_rejects_malformed() {
    const char* name = "ipv4_decode: rejects wrong version / short IHL";

    uint8_t buf[IPV4_HDR_LEN] = {};
    buf[0] = 0x60;  // version=6 — not our job to handle
    IPv4Header out{};
    if (ipv4_decode(buf, sizeof(buf), out)) {
        fail(name, "accepted version=6");
        return;
    }

    buf[0] = 0x44;  // version=4, IHL=4 (<5 is illegal)
    if (ipv4_decode(buf, sizeof(buf), out)) {
        fail(name, "accepted IHL=4");
        return;
    }

    pass(name);
}

// =========================================================================
// TCP checksum
// =========================================================================

// Hand-computed TCP checksum reference — derived by summing the pseudo-header
// + an all-zero 20-byte TCP header with data_offset=5:
//
//   pseudo:  C0A8 0001 + C0A8 0002 + 0006 + 0014    = 0x816E
//   tcp hdr: (all zero except data_offset byte 0x50) = 0x5000
//   total:                                            0xD16E
//   invert:                                           0x2E91
//
// This is small enough to sanity-check by hand and catches most common bugs:
// wrong byte order, missing pseudo-header, checksum field not zeroed.
static void test_tcp_checksum_known_vector() {
    const char* name = "tcp_checksum: matches hand-computed 0x2e91 reference";

    IPv4Header ip{};
    ip.src_ip = htonl(0xc0a80001);  // 192.168.0.1
    ip.dst_ip = htonl(0xc0a80002);  // 192.168.0.2

    TCPHeader tcp{};
    tcp.data_offset_reserved = tcp_pack_data_offset(5);  // = 0x50

    const uint16_t got_be = tcp_checksum(ip, tcp, nullptr, 0);
    const uint16_t got    = ntohs(got_be);
    if (got == 0x2e91) {
        pass(name);
    } else {
        std::cout << "    expected 0x2e91, got 0x" << std::hex << got << std::dec << '\n';
        fail(name, "wrong checksum");
    }
}

// Verify property for TCP: sum of (pseudo-header || TCP header with cksum in
// place || payload) must be zero.
static void test_tcp_checksum_verify_property() {
    const char* name = "tcp_checksum: verify property (sum with cksum in place == 0)";

    IPv4Header ip{};
    ip.src_ip = htonl(0x0a000001);
    ip.dst_ip = htonl(0x0a000002);

    TCPHeader tcp{};
    tcp.src_port             = 12345;
    tcp.dst_port             = 80;
    tcp.seq_num              = 0xdeadbeef;
    tcp.ack_num              = 0;
    tcp.data_offset_reserved = tcp_pack_data_offset(5);
    tcp.flags                = TCP_FLAG_SYN;
    tcp.window               = 65535;
    tcp.urgent_ptr           = 0;

    const char payload[] = "abcdefghij";       // 10 bytes
    const size_t pl_len  = sizeof(payload) - 1;

    // Encode the full segment.
    uint8_t segment[TCP_HDR_LEN + 16];
    const size_t enc_len = tcp_encode(ip, tcp, payload, pl_len,
                                      segment, sizeof(segment));
    if (enc_len != TCP_HDR_LEN + pl_len) {
        fail(name, "tcp_encode returned wrong length");
        return;
    }

    // Build the pseudo-header that the receiver would synthesize, then sum it
    // together with the encoded segment and expect 0.
    uint8_t pseudo[12];
    std::memcpy(pseudo + 0, &ip.src_ip, 4);
    std::memcpy(pseudo + 4, &ip.dst_ip, 4);
    pseudo[8] = 0;
    pseudo[9] = IPPROTO_TCP_V;
    const uint16_t tcp_len_be = htons(static_cast<uint16_t>(enc_len));
    std::memcpy(pseudo + 10, &tcp_len_be, 2);

    std::vector<uint8_t> full(sizeof(pseudo) + enc_len);
    std::memcpy(full.data(),                   pseudo,  sizeof(pseudo));
    std::memcpy(full.data() + sizeof(pseudo),  segment, enc_len);

    // ipv4_checksum() is just a generic RFC 1071 Internet checksum.
    const uint16_t verify = ipv4_checksum(full.data(), full.size());
    if (verify == 0)
        pass(name);
    else
        fail(name, "non-zero verify sum");
}

// Odd-length payload must be handled correctly (trailing byte zero-padded).
static void test_tcp_checksum_odd_payload() {
    const char* name = "tcp_checksum: verify property holds for odd-length payload";

    IPv4Header ip{};
    ip.src_ip = htonl(0x7f000001);   // 127.0.0.1
    ip.dst_ip = htonl(0x7f000001);

    TCPHeader tcp{};
    tcp.src_port             = 50000;
    tcp.dst_port             = 9000;
    tcp.data_offset_reserved = tcp_pack_data_offset(5);
    tcp.flags                = TCP_FLAG_ACK;
    tcp.window               = 8192;

    // 7-byte payload — odd length exercises the trailing-byte zero-pad path.
    const uint8_t payload[7] = { 'H', 'F', 'T', '!', 0x01, 0x02, 0x03 };

    uint8_t segment[TCP_HDR_LEN + sizeof(payload)];
    const size_t enc_len = tcp_encode(ip, tcp, payload, sizeof(payload),
                                      segment, sizeof(segment));
    if (enc_len != sizeof(segment)) {
        fail(name, "tcp_encode returned wrong length");
        return;
    }

    uint8_t pseudo[12];
    std::memcpy(pseudo + 0, &ip.src_ip, 4);
    std::memcpy(pseudo + 4, &ip.dst_ip, 4);
    pseudo[8] = 0;
    pseudo[9] = IPPROTO_TCP_V;
    const uint16_t tcp_len_be = htons(static_cast<uint16_t>(enc_len));
    std::memcpy(pseudo + 10, &tcp_len_be, 2);

    std::vector<uint8_t> full(sizeof(pseudo) + enc_len);
    std::memcpy(full.data(),                  pseudo,  sizeof(pseudo));
    std::memcpy(full.data() + sizeof(pseudo), segment, enc_len);

    const uint16_t verify = ipv4_checksum(full.data(), full.size());
    if (verify == 0)
        pass(name);
    else
        fail(name, "non-zero verify sum on odd payload");
}

// =========================================================================
// TCP encode/decode round-trip
// =========================================================================
static void test_tcp_roundtrip() {
    const char* name = "tcp: encode/decode round-trip";

    IPv4Header ip{};
    ip.src_ip = htonl(0x0a000001);
    ip.dst_ip = htonl(0x0a000002);

    TCPHeader tcp{};
    tcp.src_port             = 44444;
    tcp.dst_port             = 8080;
    tcp.seq_num              = 0x12345678;
    tcp.ack_num              = 0x87654321;
    tcp.data_offset_reserved = tcp_pack_data_offset(5);
    tcp.flags                = TCP_FLAG_SYN | TCP_FLAG_ACK;
    tcp.window               = 29200;
    tcp.urgent_ptr           = 0;

    uint8_t buf[TCP_HDR_LEN];
    if (tcp_encode(ip, tcp, nullptr, 0, buf, sizeof(buf)) != TCP_HDR_LEN) {
        fail(name, "encode failed");
        return;
    }

    TCPHeader decoded{};
    if (!tcp_decode(buf, sizeof(buf), decoded)) {
        fail(name, "decode failed");
        return;
    }

    if (decoded.src_port             != tcp.src_port             ||
        decoded.dst_port             != tcp.dst_port             ||
        decoded.seq_num              != tcp.seq_num              ||
        decoded.ack_num              != tcp.ack_num              ||
        decoded.data_offset_reserved != tcp.data_offset_reserved ||
        decoded.flags                != tcp.flags                ||
        decoded.window               != tcp.window               ||
        decoded.urgent_ptr           != tcp.urgent_ptr)
    {
        fail(name, "decoded fields differ from original");
        return;
    }

    if (tcp_unpack_data_offset(decoded.data_offset_reserved) != 5) {
        fail(name, "data offset unpack mismatch");
        return;
    }

    pass(name);
}

// =========================================================================
// Ethernet encode/decode
// =========================================================================
static void test_ethernet_roundtrip() {
    const char* name = "ethernet: encode/decode round-trip";

    EthernetHeader e{};
    const uint8_t dst[6] = { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };
    const uint8_t src[6] = { 0x00, 0x11, 0x22, 0x33, 0x44, 0x55 };
    std::memcpy(e.dst_mac, dst, 6);
    std::memcpy(e.src_mac, src, 6);
    e.ethertype = ETHERTYPE_IPV4;

    uint8_t buf[ETHERNET_HDR_LEN];
    if (ethernet_encode(e, buf, sizeof(buf)) != ETHERNET_HDR_LEN) {
        fail(name, "encode failed");
        return;
    }

    // Confirm the EtherType bytes are on the wire in network byte order
    // (0x08 0x00 for IPv4), independent of host endianness.
    if (buf[12] != 0x08 || buf[13] != 0x00) {
        fail(name, "ethertype not in network byte order");
        return;
    }

    EthernetHeader decoded{};
    if (!ethernet_decode(buf, sizeof(buf), decoded)) {
        fail(name, "decode failed");
        return;
    }

    if (std::memcmp(decoded.dst_mac, dst, 6) != 0 ||
        std::memcmp(decoded.src_mac, src, 6) != 0 ||
        decoded.ethertype != ETHERTYPE_IPV4)
    {
        fail(name, "decoded fields differ from original");
        return;
    }

    pass(name);
}

// Buffer-too-small paths must return 0 / false cleanly.
static void test_short_buffer_rejection() {
    const char* name = "encode/decode: short buffers rejected";

    uint8_t tiny[4] = {};

    EthernetHeader e{};
    if (ethernet_encode(e, tiny, sizeof(tiny)) != 0) {
        fail(name, "ethernet_encode accepted short buffer");
        return;
    }
    EthernetHeader e_out{};
    if (ethernet_decode(tiny, sizeof(tiny), e_out)) {
        fail(name, "ethernet_decode accepted short buffer");
        return;
    }

    IPv4Header ip{};
    ip.version_ihl = (IPV4_VERSION << 4) | IPV4_IHL_MIN;
    ip.protocol    = IPPROTO_TCP_V;
    if (ipv4_encode(ip, tiny, sizeof(tiny)) != 0) {
        fail(name, "ipv4_encode accepted short buffer");
        return;
    }
    IPv4Header ip_out{};
    if (ipv4_decode(tiny, sizeof(tiny), ip_out)) {
        fail(name, "ipv4_decode accepted short buffer");
        return;
    }

    TCPHeader tcp{};
    tcp.data_offset_reserved = tcp_pack_data_offset(5);
    if (tcp_encode(ip, tcp, nullptr, 0, tiny, sizeof(tiny)) != 0) {
        fail(name, "tcp_encode accepted short buffer");
        return;
    }
    TCPHeader tcp_out{};
    if (tcp_decode(tiny, sizeof(tiny), tcp_out)) {
        fail(name, "tcp_decode accepted short buffer");
        return;
    }

    pass(name);
}

// =========================================================================
// main
// =========================================================================
int main() {
    std::cout << "=== Phase 2: Packet Construction & Checksum Tests ===\n\n";

    test_ipv4_checksum_known_vector();
    test_ipv4_checksum_verify_property();
    test_ipv4_roundtrip();
    test_ipv4_decode_rejects_malformed();

    test_tcp_checksum_known_vector();
    test_tcp_checksum_verify_property();
    test_tcp_checksum_odd_payload();
    test_tcp_roundtrip();

    test_ethernet_roundtrip();
    test_short_buffer_rejection();

    std::cout << '\n'
              << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
