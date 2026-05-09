#pragma once
#include "pkt_headers.hpp"

/* Compute checksum for count bytes starting at addr, using one's complement of one's complement sum*/
unsigned short compute_checksum(unsigned short *addr, unsigned int count)
{
    uint32_t sum = 0;

    while (count > 1)
    {
        sum += ntohs(*addr++);
        count -= 2;
    }

    // Handle leftover byte
    if (count > 0)
    {
        sum += *((uint8_t *)addr) << 8; // pad high byte
    }

    // Fold to 16 bits
    while (sum >> 16)
    {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }

    return (unsigned short)(~sum);
}

void compute_ip_checksum(struct ip_hdr *ip_hdr)
{
    ip_hdr->check = 0;
    ip_hdr->check = compute_checksum((unsigned short *)ip_hdr, 20);
}

uint16_t tcp_checksum(struct pkt_hdr *pkt, size_t payload_len, size_t total_len)
{
    uint32_t sum = 0;
    const uint16_t *p;
    size_t len;

    // Step 1: Zero the checksum field
    ((struct tcp_hdr *)&pkt->tcp)->check = 0;

    // Step 2: Sum TCP header
    p = (const uint16_t *)&pkt->tcp;
    len = (uint32_t)((pkt->tcp.data_off_reserved >> 4) * 4);
    while (len > 1)
    {
        sum += ntohs(*p++);
        len -= 2;
    }

    // Step 3: Sum payload
    //dump_buffer((const uint8_t *)pkt, sizeof(struct pkt_hdr) + payload_len);
    len = payload_len;
    while (len > 1)
    {
        sum += ntohs(*p++);
        len -= 2;
    }
    if (len == 1)
    {
        sum += *((const uint8_t *)p) << 8; // ✅ shift into upper byte (pad low with zero)
    }

    // Step 4: Sum pseudo-header
    uint32_t src = ntohl(pkt->ip.src_addr);
    uint32_t dst = ntohl(pkt->ip.dst_addr);

    sum += (src >> 16) & 0xFFFF;
    sum += src & 0xFFFF;
    sum += (dst >> 16) & 0xFFFF;
    sum += dst & 0xFFFF;

    sum += pkt->ip.protocol;                     // protocol is 8 bits, promoted to 16
    sum += (uint32_t)((pkt->tcp.data_off_reserved >> 4) * 4) + payload_len; // TCP length

    // Step 5: Fold 32-bit sum to 16-bit
    while (sum >> 16)
        sum = (sum & 0xFFFF) + (sum >> 16);

    return (uint16_t)~sum;
}

/**
 * Builds a TCP packet with the given payload and payload length.
 * The packet is built in the buffer passed as argument. The passed buffer is populated with the complete packet.
 *
 * @param pkt_hdr: Pointer to the packet header.
 * @param payload: Pointer to the payload.
 * @param payload_len: Length of the payload.
 * @param buffer: Buffer to store the packet.
 * */
void build_tcp_packet(const char *payload, size_t payload_len, uint8_t flags, uint32_t seq, uint32_t ack, char *buffer)
{
    struct pkt_hdr pkt_hdr;
    std::cout << "ack: " << ack << std::endl;
    pkt_hdr.ip.tot_len = htons((uint16_t)(sizeof(struct ip_hdr) + sizeof(struct tcp_hdr) + payload_len));
    std::cout << "IP TOTAL LEN: " << payload_len << std::endl;
    compute_ip_checksum(&pkt_hdr.ip);

    pkt_hdr.tcp.src_port = htons(1234);  // MANUAL
    pkt_hdr.tcp.dst_port = htons(12345); // MANUAL
    pkt_hdr.tcp.seq_num = htonl(seq);    // MANUAL
    pkt_hdr.tcp.ack_num = htonl(ack);    // MANUAL
    pkt_hdr.tcp.flags = flags;           // MANUAL
    pkt_hdr.tcp.check = tcp_checksum(&pkt_hdr, payload_len, sizeof(struct pkt_hdr) + payload_len);

    std::cout << "TCP ACK NUM: " << ntohl(pkt_hdr.tcp.ack_num) << std::endl;

    memcpy(buffer, &pkt_hdr, sizeof(struct pkt_hdr));
    if (payload_len > 0)
    {
        memcpy(buffer + sizeof(struct pkt_hdr), payload, payload_len);
    }

    return;
}
