#include "net/ipv4.h"

#include <cstring>

#include <arpa/inet.h>  // htons / htonl / ntohs / ntohl

// RFC 1071 Internet checksum: sum all 16-bit words, fold carries, one's
// complement. Handles odd-length buffers by treating the trailing byte as
// the high byte of a zero-padded 16-bit word (standard behavior).
//
// Returns the result in network byte order. When storing into a header's
// checksum field, the field itself must be zero during the computation.
uint16_t ipv4_checksum(const void* hdr, size_t len) {
    const auto* p = static_cast<const uint8_t*>(hdr);
    uint32_t sum = 0;

    // Process 16-bit words. We read big-endian (network order) to build the
    // sum; the final one's-complement is returned as a raw 16-bit value which
    // is already in network byte order when written back out.
    while (len >= 2) {
        sum += (static_cast<uint32_t>(p[0]) << 8) | p[1];
        p   += 2;
        len -= 2;
    }

    // Trailing odd byte — treat as high byte of a zero-padded word.
    if (len == 1)
        sum += static_cast<uint32_t>(p[0]) << 8;

    // Fold carries into the low 16 bits (may itself produce a carry).
    while (sum >> 16)
        sum = (sum & 0xFFFF) + (sum >> 16);

    const uint16_t checksum_host = static_cast<uint16_t>(~sum & 0xFFFF);
    return htons(checksum_host);
}

size_t ipv4_encode(const IPv4Header& hdr, void* buf, size_t buf_len) {
    if (buf_len < IPV4_HDR_LEN)
        return 0;

    auto* p = static_cast<uint8_t*>(buf);

    // Byte 0: version (high nibble) + IHL (low nibble). Caller owns both.
    p[0] = hdr.version_ihl;
    p[1] = hdr.tos;

    // 16-bit fields: host → network.
    const uint16_t total_be = htons(hdr.total_length);
    std::memcpy(p + 2, &total_be, 2);

    const uint16_t id_be = htons(hdr.identification);
    std::memcpy(p + 4, &id_be, 2);

    const uint16_t flags_frag_be = htons(hdr.flags_frag);
    std::memcpy(p + 6, &flags_frag_be, 2);

    p[8] = hdr.ttl;
    p[9] = hdr.protocol;

    // Zero the checksum slot before computing — the algorithm requires it.
    p[10] = 0;
    p[11] = 0;

    // src/dst IPs are stored as uint32 in network byte order already (that's
    // the convention for addresses built via inet_pton / htonl). We simply
    // copy the bytes through.
    std::memcpy(p + 12, &hdr.src_ip, 4);
    std::memcpy(p + 16, &hdr.dst_ip, 4);

    // Compute the checksum over the 20 bytes we just wrote, then store it
    // back in the slot we zeroed.
    const uint16_t cksum = ipv4_checksum(p, IPV4_HDR_LEN);
    std::memcpy(p + 10, &cksum, 2);

    return IPV4_HDR_LEN;
}

bool ipv4_decode(const void* buf, size_t buf_len, IPv4Header& out) {
    if (buf_len < IPV4_HDR_LEN)
        return false;

    const auto* p = static_cast<const uint8_t*>(buf);

    out.version_ihl = p[0];
    const uint8_t version = (out.version_ihl >> 4) & 0x0F;
    const uint8_t ihl     = out.version_ihl & 0x0F;
    if (version != IPV4_VERSION || ihl < IPV4_IHL_MIN)
        return false;

    out.tos = p[1];

    uint16_t total_be;
    std::memcpy(&total_be, p + 2, 2);
    out.total_length = ntohs(total_be);

    uint16_t id_be;
    std::memcpy(&id_be, p + 4, 2);
    out.identification = ntohs(id_be);

    uint16_t flags_frag_be;
    std::memcpy(&flags_frag_be, p + 6, 2);
    out.flags_frag = ntohs(flags_frag_be);

    out.ttl      = p[8];
    out.protocol = p[9];

    std::memcpy(&out.checksum, p + 10, 2);  // leave in network byte order
    std::memcpy(&out.src_ip,   p + 12, 4);  // addresses stay in network order
    std::memcpy(&out.dst_ip,   p + 16, 4);

    return true;
}
