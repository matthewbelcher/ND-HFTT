#ifndef HFT_TCPSTACK_TCP_H
#define HFT_TCPSTACK_TCP_H

#include <cstddef>
#include <cstdint>

#include "net/ipv4.h"

// TCP header (20 bytes, no options).
//
// On the wire, all multi-byte fields are in network byte order.
// The encode/decode helpers convert at the boundary; fields in this struct
// (when populated by callers) are expected in HOST byte order for:
//   src_port, dst_port, seq_num, ack_num, window, urgent_ptr
// data_offset_reserved and flags are single-byte bitfields.
struct __attribute__((packed)) TCPHeader {
    uint16_t src_port;
    uint16_t dst_port;
    uint32_t seq_num;
    uint32_t ack_num;

    // data_offset: high 4 bits = header length in 32-bit words (min 5, max 15).
    // low 4 bits are the 3-bit reserved field + the NS flag.
    // We don't use NS, so the low nibble is always 0 in our encoder.
    uint8_t  data_offset_reserved;

    // CWR ECE URG ACK PSH RST SYN FIN — MSB to LSB.
    uint8_t  flags;

    uint16_t window;
    uint16_t checksum;
    uint16_t urgent_ptr;
};

static_assert(sizeof(TCPHeader) == 20, "TCPHeader must be 20 bytes");

constexpr size_t TCP_HDR_LEN = 20;

// TCP flag bits.
constexpr uint8_t TCP_FLAG_FIN = 0x01;
constexpr uint8_t TCP_FLAG_SYN = 0x02;
constexpr uint8_t TCP_FLAG_RST = 0x04;
constexpr uint8_t TCP_FLAG_PSH = 0x08;
constexpr uint8_t TCP_FLAG_ACK = 0x10;
constexpr uint8_t TCP_FLAG_URG = 0x20;
constexpr uint8_t TCP_FLAG_ECE = 0x40;
constexpr uint8_t TCP_FLAG_CWR = 0x80;

// Pack a 4-bit data-offset (header length in 32-bit words) into the high
// nibble of the data_offset_reserved byte. The low nibble (reserved + NS)
// is cleared.
inline uint8_t tcp_pack_data_offset(uint8_t words) {
    return static_cast<uint8_t>((words & 0x0F) << 4);
}

// Extract the data-offset (header length in 32-bit words) from the
// data_offset_reserved byte of a received TCP header.
inline uint8_t tcp_unpack_data_offset(uint8_t data_offset_reserved) {
    return static_cast<uint8_t>((data_offset_reserved >> 4) & 0x0F);
}

// Compute the TCP checksum over the pseudo-header + TCP header + payload,
// per RFC 793.
//
// The pseudo-header is built from:
//   src_ip, dst_ip (from `ip`, in network byte order as stored)
//   protocol       (must be IPPROTO_TCP_V = 6)
//   tcp_length     (TCP header length + payload length)
//
// `tcp` is the on-the-wire TCP header (network byte order, checksum field zero).
// `payload` / `payload_len` is the application data following the header.
//
// Returns the checksum in NETWORK byte order, ready to store into
// `tcp.checksum` directly.
uint16_t tcp_checksum(const IPv4Header& ip,
                      const TCPHeader&  tcp,
                      const void*       payload,
                      size_t            payload_len);

// Write `tcp` + `payload` into `buf` (no IPv4 header — that's the caller's job).
// `tcp` fields are accepted in HOST byte order; the checksum is computed
// using `ip` to build the pseudo-header. Returns bytes written, or 0 on
// insufficient buffer.
size_t tcp_encode(const IPv4Header& ip,
                  const TCPHeader&  tcp,
                  const void*       payload,
                  size_t            payload_len,
                  void*             buf,
                  size_t            buf_len);

// Parse a TCP header from `buf` into `out`. Multi-byte fields are returned
// in HOST byte order. Returns true on success, false if `buf_len < TCP_HDR_LEN`
// or the encoded data offset is less than 5 (malformed).
// Does NOT validate the checksum.
bool tcp_decode(const void* buf, size_t buf_len, TCPHeader& out);

#endif //HFT_TCPSTACK_TCP_H
