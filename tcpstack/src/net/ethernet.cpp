#include "net/ethernet.h"

#include <cstring>

#include <arpa/inet.h>  // htons / ntohs

size_t ethernet_encode(const EthernetHeader& hdr, void* buf, size_t buf_len) {
    if (buf_len < ETHERNET_HDR_LEN)
        return 0;

    auto* p = static_cast<uint8_t*>(buf);
    std::memcpy(p + 0, hdr.dst_mac, 6);
    std::memcpy(p + 6, hdr.src_mac, 6);

    // ethertype arrives in host byte order; convert at the boundary.
    const uint16_t et_be = htons(hdr.ethertype);
    std::memcpy(p + 12, &et_be, 2);

    return ETHERNET_HDR_LEN;
}

bool ethernet_decode(const void* buf, size_t buf_len, EthernetHeader& out) {
    if (buf_len < ETHERNET_HDR_LEN)
        return false;

    const auto* p = static_cast<const uint8_t*>(buf);
    std::memcpy(out.dst_mac, p + 0, 6);
    std::memcpy(out.src_mac, p + 6, 6);

    uint16_t et_be;
    std::memcpy(&et_be, p + 12, 2);
    out.ethertype = ntohs(et_be);

    return true;
}
