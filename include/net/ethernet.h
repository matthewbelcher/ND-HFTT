#ifndef HFT_TCPSTACK_ETHERNET_H
#define HFT_TCPSTACK_ETHERNET_H

#include <cstddef>
#include <cstdint>

// Ethernet II frame header (14 bytes, no 802.1Q VLAN tag).
//
// All multi-byte fields are stored in network byte order (big-endian).
// Use helpers ethernet_encode() / ethernet_decode() below to keep the
// byte-order conversions localized.
struct __attribute__((packed)) EthernetHeader {
    uint8_t  dst_mac[6];
    uint8_t  src_mac[6];
    uint16_t ethertype;   // network byte order
};

static_assert(sizeof(EthernetHeader) == 14, "EthernetHeader must be 14 bytes");

// Common EtherType values (host byte order — convert with htons() before writing).
constexpr uint16_t ETHERTYPE_IPV4 = 0x0800;
constexpr uint16_t ETHERTYPE_ARP  = 0x0806;
constexpr uint16_t ETHERTYPE_IPV6 = 0x86DD;

// Length of an Ethernet II header on the wire.
constexpr size_t ETHERNET_HDR_LEN = 14;

// Write `hdr` into `buf` (which must have at least ETHERNET_HDR_LEN bytes).
// `hdr.ethertype` is expected in HOST byte order and will be converted.
// Returns the number of bytes written (always ETHERNET_HDR_LEN).
size_t ethernet_encode(const EthernetHeader& hdr, void* buf, size_t buf_len);

// Parse an Ethernet header from `buf` into `out`.
// `out.ethertype` is returned in HOST byte order.
// Returns true on success, false if `buf_len < ETHERNET_HDR_LEN`.
bool ethernet_decode(const void* buf, size_t buf_len, EthernetHeader& out);

#endif //HFT_TCPSTACK_ETHERNET_H
