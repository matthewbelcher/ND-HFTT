#include "net/tcp.h"

#include <cstring>

#include <arpa/inet.h>  // htons / ntohs / htonl / ntohl

namespace {

// Accumulate an Internet-checksum-style sum (16-bit big-endian words, with
// a trailing odd byte zero-padded on the right). Caller folds and inverts.
uint32_t sum_bytes_be(const uint8_t* p, size_t len, uint32_t start = 0) {
    uint32_t sum = start;
    while (len >= 2) {
        sum += (static_cast<uint32_t>(p[0]) << 8) | p[1];
        p   += 2;
        len -= 2;
    }
    if (len == 1)
        sum += static_cast<uint32_t>(p[0]) << 8;
    return sum;
}

uint16_t fold_and_invert(uint32_t sum) {
    while (sum >> 16)
        sum = (sum & 0xFFFF) + (sum >> 16);
    return static_cast<uint16_t>(~sum & 0xFFFF);
}

}  // namespace

// TCP checksum uses a 12-byte pseudo-header:
//   [ src_ip (4) | dst_ip (4) | zero (1) | protocol (1) | tcp_len (2) ]
// followed by the TCP header (with checksum field = 0) and the payload.
// The sum is a standard RFC 1071 Internet checksum.
uint16_t tcp_checksum(const IPv4Header& ip,
                      const TCPHeader&  tcp_host,
                      const void*       payload,
                      size_t            payload_len)
{
    const uint16_t tcp_total_len = static_cast<uint16_t>(TCP_HDR_LEN + payload_len);

    // Build the 12-byte pseudo-header in a stack buffer, then sum it.
    // IP addresses are already stored in network byte order.
    uint8_t pseudo[12];
    std::memcpy(pseudo + 0, &ip.src_ip, 4);
    std::memcpy(pseudo + 4, &ip.dst_ip, 4);
    pseudo[8] = 0;
    pseudo[9] = IPPROTO_TCP_V;
    const uint16_t tcp_len_be = htons(tcp_total_len);
    std::memcpy(pseudo + 10, &tcp_len_be, 2);

    uint32_t sum = sum_bytes_be(pseudo, sizeof(pseudo));

    // Serialize the TCP header into a 20-byte scratch buffer with checksum=0,
    // then sum it. This keeps us independent of struct padding/alignment and
    // makes the host→network conversions explicit.
    uint8_t tcp_hdr[TCP_HDR_LEN];
    const uint16_t sp_be  = htons(tcp_host.src_port);
    const uint16_t dp_be  = htons(tcp_host.dst_port);
    const uint32_t seq_be = htonl(tcp_host.seq_num);
    const uint32_t ack_be = htonl(tcp_host.ack_num);
    const uint16_t win_be = htons(tcp_host.window);
    const uint16_t urg_be = htons(tcp_host.urgent_ptr);

    std::memcpy(tcp_hdr + 0,  &sp_be,  2);
    std::memcpy(tcp_hdr + 2,  &dp_be,  2);
    std::memcpy(tcp_hdr + 4,  &seq_be, 4);
    std::memcpy(tcp_hdr + 8,  &ack_be, 4);
    tcp_hdr[12] = tcp_host.data_offset_reserved;
    tcp_hdr[13] = tcp_host.flags;
    std::memcpy(tcp_hdr + 14, &win_be, 2);
    tcp_hdr[16] = 0;  // checksum high byte (zero during computation)
    tcp_hdr[17] = 0;  // checksum low byte
    std::memcpy(tcp_hdr + 18, &urg_be, 2);

    sum = sum_bytes_be(tcp_hdr, TCP_HDR_LEN, sum);

    // Payload — treated as an opaque byte stream with the standard odd-byte
    // zero-pad behavior handled inside sum_bytes_be().
    if (payload != nullptr && payload_len > 0)
        sum = sum_bytes_be(static_cast<const uint8_t*>(payload), payload_len, sum);

    const uint16_t cksum_host = fold_and_invert(sum);
    return htons(cksum_host);
}

size_t tcp_encode(const IPv4Header& ip,
                  const TCPHeader&  tcp,
                  const void*       payload,
                  size_t            payload_len,
                  void*             buf,
                  size_t            buf_len)
{
    const size_t total = TCP_HDR_LEN + payload_len;
    if (buf_len < total)
        return 0;

    auto* p = static_cast<uint8_t*>(buf);

    const uint16_t sp_be  = htons(tcp.src_port);
    const uint16_t dp_be  = htons(tcp.dst_port);
    const uint32_t seq_be = htonl(tcp.seq_num);
    const uint32_t ack_be = htonl(tcp.ack_num);
    const uint16_t win_be = htons(tcp.window);
    const uint16_t urg_be = htons(tcp.urgent_ptr);

    std::memcpy(p + 0,  &sp_be,  2);
    std::memcpy(p + 2,  &dp_be,  2);
    std::memcpy(p + 4,  &seq_be, 4);
    std::memcpy(p + 8,  &ack_be, 4);
    p[12] = tcp.data_offset_reserved;
    p[13] = tcp.flags;
    std::memcpy(p + 14, &win_be, 2);
    // Zero the checksum slot first, fill in after payload is in place.
    p[16] = 0;
    p[17] = 0;
    std::memcpy(p + 18, &urg_be, 2);

    if (payload != nullptr && payload_len > 0)
        std::memcpy(p + TCP_HDR_LEN, payload, payload_len);

    // Compute the checksum over the pseudo-header + the header we just wrote
    // + the payload, then splice it into the header's checksum field.
    const uint16_t cksum = tcp_checksum(ip, tcp, payload, payload_len);
    std::memcpy(p + 16, &cksum, 2);

    return total;
}

bool tcp_decode(const void* buf, size_t buf_len, TCPHeader& out) {
    if (buf_len < TCP_HDR_LEN)
        return false;

    const auto* p = static_cast<const uint8_t*>(buf);

    uint16_t sp_be, dp_be, win_be, urg_be;
    uint32_t seq_be, ack_be;

    std::memcpy(&sp_be,  p + 0,  2);
    std::memcpy(&dp_be,  p + 2,  2);
    std::memcpy(&seq_be, p + 4,  4);
    std::memcpy(&ack_be, p + 8,  4);
    std::memcpy(&win_be, p + 14, 2);
    std::memcpy(&urg_be, p + 18, 2);

    out.src_port             = ntohs(sp_be);
    out.dst_port             = ntohs(dp_be);
    out.seq_num              = ntohl(seq_be);
    out.ack_num              = ntohl(ack_be);
    out.data_offset_reserved = p[12];
    out.flags                = p[13];
    out.window               = ntohs(win_be);
    std::memcpy(&out.checksum, p + 16, 2);  // keep checksum in network order
    out.urgent_ptr           = ntohs(urg_be);

    // Reject malformed headers (data offset < 5 would imply a <20 byte header).
    if (tcp_unpack_data_offset(out.data_offset_reserved) < 5)
        return false;

    return true;
}
