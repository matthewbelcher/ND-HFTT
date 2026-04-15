#ifndef HFT_TCPSTACK_IPV4_H
#define HFT_TCPSTACK_IPV4_H

#include <cstddef>
#include <cstdint>

// IPv4 header (20 bytes, no options).
//
// Fields are stored in network byte order (big-endian) on the wire.
// The encode/decode helpers convert to/from host byte order at the boundary;
// callers should populate fields in HOST byte order and call ipv4_encode().
struct __attribute__((packed)) IPv4Header {
    // version (4 bits) + IHL (4 bits) packed into one byte.
    // version is the high nibble, IHL (header length in 32-bit words) is the low.
    uint8_t  version_ihl;

    uint8_t  tos;          // type of service / DSCP + ECN
    uint16_t total_length; // header + payload, network byte order
    uint16_t identification;
    uint16_t flags_frag;   // 3-bit flags (high) + 13-bit fragment offset
    uint8_t  ttl;
    uint8_t  protocol;     // e.g. IPPROTO_TCP (6)
    uint16_t checksum;     // one's-complement header checksum
    uint32_t src_ip;       // network byte order
    uint32_t dst_ip;       // network byte order
};

static_assert(sizeof(IPv4Header) == 20, "IPv4Header must be 20 bytes");

constexpr size_t   IPV4_HDR_LEN   = 20;
constexpr uint8_t  IPV4_VERSION   = 4;
constexpr uint8_t  IPV4_IHL_MIN   = 5;          // 5 * 4 = 20 bytes
constexpr uint8_t  IPV4_DEFAULT_TTL = 64;
constexpr uint8_t  IPPROTO_TCP_V  = 6;          // avoid clashing with <netinet/in.h>

// Bit helpers for version_ihl and flags_frag.
constexpr uint8_t  IPV4_DF_FLAG   = 0x40;  // don't fragment (high byte of flags_frag)
constexpr uint8_t  IPV4_MF_FLAG   = 0x20;  // more fragments

// Standard Internet checksum (RFC 1071): ones' complement sum of 16-bit words.
// Used for both IPv4 header and (with a pseudo-header prefix) TCP segments.
// Returns the checksum in network byte order, ready to store into a header
// field directly (the field itself should be zeroed before calling).
uint16_t ipv4_checksum(const void* hdr, size_t len);

// Write `hdr` into `buf`. Accepts fields in HOST byte order for:
//   total_length, identification, flags_frag, src_ip, dst_ip
// The checksum is computed internally (the caller's `hdr.checksum` is ignored).
// version_ihl, tos, ttl, protocol are single-byte fields and need no conversion.
// Returns the number of bytes written (always IPV4_HDR_LEN), or 0 on buffer
// too small.
size_t ipv4_encode(const IPv4Header& hdr, void* buf, size_t buf_len);

// Parse an IPv4 header from `buf` into `out`. Multi-byte fields in `out` are
// returned in HOST byte order. Returns true on success, false if `buf_len`
// is short or the header is malformed (bad version / IHL < 5).
// Does NOT validate the checksum — callers that care should call
// ipv4_checksum() on the raw bytes and expect 0.
bool ipv4_decode(const void* buf, size_t buf_len, IPv4Header& out);

#endif //HFT_TCPSTACK_IPV4_H
