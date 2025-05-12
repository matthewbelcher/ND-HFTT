#include <stdint.h>
#include <string.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <net/ethernet.h>
#include <iostream>

/* Ethernet header */

enum class TCP_FLAGS: uint8_t {
    SYN = 0b00000010,
    ACK = 0b00010000,
    FIN = 0b00000001,
    RST = 0b00000100,
    PSH = 0b00001000,
};

struct eth_hdr
{
    // swap the mac addresses
    uint8_t dst_mac[ETH_ALEN] = {0x00, 0x0f, 0x53, 0x4b, 0xe6, 0xb1}; /* Destination MAC address */
    uint8_t src_mac[ETH_ALEN] = {0x00, 0x0f, 0x53, 0x5a, 0x4d, 0xa1}; /* Source MAC address */
    uint16_t ether_type = htons(ETH_P_IP);                            /* EtherType (e.g., ETH_P_IP) */
} __attribute__((packed));

/* IP header */
struct ip_hdr
{
    uint8_t version_ihl = 0x45; /* Version and IHL 4 indicates ipv4, 5 indicates 32*5 = 160 bits*/
    uint8_t dscp_ecn = 0x00;
    uint16_t tot_len; /* Total Length */              // # MODIFICATION
    uint16_t id = htons(0x00fc); /* Identification */ // No fragmentation, so don't care
    uint16_t flags_frag_off = htons(0x0000);          /* Flags and Fragment Offset, assuming no fragmentation */
    uint8_t ttl = 0x40;                               /* Time to Live */
    uint8_t protocol = IPPROTO_TCP;                   /* Protocol (e.g., IPPROTO_TCP) */
    uint16_t check = 0; /* Header Checksum */         // MODIFICATION
    uint32_t src_addr = htonl(0xc0a80d17);            /* Source Address */
    uint32_t dst_addr = htonl(0xc0a80d0a);            /* Destination Address */
} __attribute__((packed));

/* TCP header */
struct tcp_hdr
{
    uint16_t src_port; /* Source Port */              // MODIFICATION
    uint16_t dst_port; /* Destination Port */         // MODIFICATION
    uint32_t seq_num; /* Sequence Number */           // MODIFICATION
    uint32_t ack_num; /* Acknowledgment Number */     // MODIFICATION
    uint8_t data_off_reserved = 0b01010000;           /* Data Offset and Reserved */
    uint8_t flags; /* TCP Flags */                    // MODIFICATION
    uint16_t window = htons(UINT16_MAX); /* Window */ // MODIFICATION
    uint16_t check = 0; /* Checksum */                // MODIFICATION
    uint16_t urg_ptr = 0;                             /* Urgent Pointer */
} __attribute__((packed));

struct pkt_hdr
{
    struct eth_hdr eth;
    struct ip_hdr ip;
    struct tcp_hdr tcp;
} __attribute__((packed));

/* Compute checksum for count bytes starting at addr, using one's complement of one's complement sum*/
unsigned short compute_checksum(unsigned short *addr, unsigned int count);
void compute_ip_checksum(struct ip_hdr *ip_hdr);
uint16_t tcp_checksum(struct pkt_hdr *pkt, size_t payload_len, size_t total_len);

/**
/**
 * Builds a TCP packet with the given payload and payload length.
 * The packet is built in the buffer passed as argument. The passed buffer is populated with the complete packet.
 *
 * @param pkt_hdr: Pointer to the packet header.
 * @param payload: Pointer to the payload.
 * @param payload_len: Length of the payload.
 * @param buffer: Buffer to store the packet.
 * */
void build_tcp_packet(const char *payload, size_t payload_len, uint8_t flags, uint32_t seq, uint32_t ack, char *buffer);