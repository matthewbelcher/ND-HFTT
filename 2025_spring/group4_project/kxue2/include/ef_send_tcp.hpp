#include <etherfabric/vi.h>
#include <etherfabric/pd.h>
#include <etherfabric/memreg.h>
#include <etherfabric/capabilities.h>
#include "utils.h"
#include "pkt_headers.hpp"
#include <iostream>
#include <tuple>
#include <bitset>
#include <chrono>
#include <queue>
#define PKT_BUF_SIZE 2048                                            // Size of each packet buffer
#define RX_DMA_OFF ROUND_UP(sizeof(struct pkt_buf), EF_VI_DMA_ALIGN) // Offset of the RX DMA address
#define RX_RING_SIZE 512                                             // Maximum number of receive requests in the RX ring
#define TX_RING_SIZE 2048                                            // Maximum number of transmit requests in the TX ring
#define REFILL_BATCH_SIZE 64                                         // Minimum number of buffers to refill the ring

struct pkt_buf
{
    ef_addr rx_ef_addr;
    ef_addr tx_ef_addr;
    int64_t id;
    struct pkt_buf *next;
} __attribute__((packed));

struct pkt_bufs
{
    void *mem;
    size_t mem_size;
    int num;
    struct pkt_buf *free_pool;
    int free_pool_n;
};

struct vi
{
    ef_driver_handle dh;
    ef_pd pd;
    ef_vi vi;
    ef_memreg memreg;
    unsigned int tx_outstanding;
    uint64_t n_pkts;
};

class TcpResetException : public std::exception {
public:
    const char* what() const noexcept override {
        return "TCP connection reset (RST received)";
    }
};
/*
    This function returns a pointer to the packet buffer at index pkt_buf_i.
    It casts the memory pointer to a pointer to a pkt_buf struct.
    It then returns the pointer to the pkt_buf struct.
    id -> pkt_buf struct
*/
static inline struct pkt_buf *pkt_buf_from_id(int pkt_buf_i);
/*
    This function returns the offset of the DMA address from the packet buffer.
    It returns the offset of the DMA address from the packet buffer.
    pkt_buf_i -> offset (not entirely important)
*/
static inline int addr_offset_from_id(int pkt_buf_i);
/*
    This function refills the RX ring.
    It checks if the RX ring has enough space to refill the ring.
    It also checks if there are enough free buffers to refill the ring.
    If it does, it refills the ring.
    If it doesn't, it returns.
*/
static void vi_refill_rx_ring(void);
/*
    This function frees a packet buffer.
    It adds the packet buffer to the free pool.
    pkt_buf -> free pool
*/
static inline void pkt_buf_free(struct pkt_buf *pkt_buf);
/*
    This function initializes the packet buffers.
    It sets the number of packet buffers to the sum of the RX and TX ring sizes.
    It then sets the memory size to the number of packet buffers times the size of each packet buffer.
    It then maps the memory to the packet buffers.
    It then initializes the packet buffers.
*/
static int init_pkts_memory(void);
/*
    This function initializes the virtual interface.
    It sets the flags to the default flags.
    It then opens the driver.
    It then allocates the PD.
    It then allocates the VI.
    It then allocates the memory register.
    It then initializes the packet buffers.
    It then sets the filters to receive TCP packets.
    It then returns 0.
*/
static int init(const char *intf);
/**
 * @brief Send a packet with the given payload, payload length, flags, sequence number, and acknowledgment number and frees the buffer
 * Note: seq and ack are numbers to be sent with the packet
 * @param payload
 * @param payload_len
 * @param flags
 * @param seq
 * @param ack
 */
static void send_packet(char *payload, int payload_len, uint8_t flags, uint32_t seq, uint32_t ack);
/*
 * Receive a packet and verify the seq, ack, and flags are as expected
 */
static std::tuple<struct pkt_hdr *, uint32_t, uint8_t> receive_packet(uint8_t flags, uint32_t seq, uint32_t ack);
/*
 * Send a connection handshake
 */
static void send_connection_handshake();
/*
 * Send a TCP teardown
 */
static void send_tcp_teardown();
/*
 * Send a hello world packet
 */
static void send_hello_world();
/*
* Initialize the EF_VI TCP interface
*/
void ef_init_tcp_client();
/*
 * Connect to the server
 */
void ef_connect();
/*
 * Disconnect from the server
 */
void ef_disconnect();
/*
 * Read a packet
 */
ssize_t ef_read(char* buf, int count);
/*
 * Reset the variables
 */
void reset_variables();
/*
 * Send a reset
 */
static void send_reset();
/*
 * Send a packet
 */
ssize_t ef_send(char* buf, int len);
/*
 * Poll events for incoming packets when data is immediately wanted
 */
static void poll_events(char *buf, ssize_t& read, int len);
/*
 * Poll events for incoming packets when data is not immediately wanted
 */
static void poll_events();

void dump_buffer(const uint8_t *buf, size_t len);







