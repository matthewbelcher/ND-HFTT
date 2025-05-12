#include "ef_send_tcp.hpp"

/* Future Changes
1. Make a parser to handle reads in an event driven callback way
2. Make sure when numbers don't align (snd_nxt, rcv_nxt, snd_una) that we handle gracefully with right logic
3. Allow for retransmissions, related to 2
4. Handle window updates
5. Handle congestion control
6. State machine to better handle TCP states?
7. Debug duplex messaging and regular messaging more
*/

static struct vi vi;
static struct pkt_bufs pbs;
static uint32_t snd_nxt = 16000000;
static uint32_t rcv_nxt = 0;
static uint32_t snd_una = 0;
static std::queue<std::tuple<char *, ssize_t, ssize_t>> data_queue;
/*
    This function returns a pointer to the packet buffer at index pkt_buf_i.
    It casts the memory pointer to a pointer to a pkt_buf struct.
    It then returns the pointer to the pkt_buf struct.
    id -> pkt_buf struct
*/
static inline struct pkt_buf *pkt_buf_from_id(int pkt_buf_i)
{
    assert((unsigned)pkt_buf_i < (unsigned)pbs.num);
    return (struct pkt_buf *)((char *)pbs.mem + (size_t)pkt_buf_i * PKT_BUF_SIZE);
}
/*
    This function returns the offset of the DMA address from the packet buffer.
    It returns the offset of the DMA address from the packet buffer.
    pkt_buf_i -> offset (not entirely important)
*/
static inline int addr_offset_from_id(int pkt_buf_i)
{
    return (pkt_buf_i % 2) * EF_VI_DMA_ALIGN;
}
/*
    This function refills the RX ring.
    It checks if the RX ring has enough space to refill the ring.
    It also checks if there are enough free buffers to refill the ring.
    If it does, it refills the ring.
    If it doesn't, it returns.
*/
static void vi_refill_rx_ring(void)
{
    ef_vi *vi_ptr = &vi.vi;
    struct pkt_buf *pkt_buf;
    int i;

    if (ef_vi_receive_space(vi_ptr) < REFILL_BATCH_SIZE ||
        pbs.free_pool_n < REFILL_BATCH_SIZE)
        return;

    for (i = 0; i < REFILL_BATCH_SIZE; ++i)
    {
        pkt_buf = pbs.free_pool;
        pbs.free_pool = pbs.free_pool->next;
        --pbs.free_pool_n;
        ef_vi_receive_init(vi_ptr, pkt_buf->rx_ef_addr, pkt_buf->id);
    }
    ef_vi_receive_push(vi_ptr);
}
/*
    This function frees a packet buffer.
    It adds the packet buffer to the free pool.
    pkt_buf -> free pool
*/
static inline void pkt_buf_free(struct pkt_buf *pkt_buf)
{
    pkt_buf->next = pbs.free_pool;
    pbs.free_pool = pkt_buf;
    ++pbs.free_pool_n;
}

void reset_variables()
{
    data_queue.empty();
    snd_nxt = 16000000;
    rcv_nxt = 0;
    snd_una = 0;
}

void set_variables()
{
    data_queue.empty();
    snd_nxt = 16000000;
    rcv_nxt = 0;
    snd_una = 0;
}
/*
    This function initializes the packet buffers.
    It sets the number of packet buffers to the sum of the RX and TX ring sizes.
    It then sets the memory size to the number of packet buffers times the size of each packet buffer.
    It then maps the memory to the packet buffers.
    It then initializes the packet buffers.
*/
static int init_pkts_memory(void)
{
    int64_t i;
    pbs.num = RX_RING_SIZE + TX_RING_SIZE;
    pbs.mem_size = pbs.num * PKT_BUF_SIZE;
    pbs.mem_size = ROUND_UP(pbs.mem_size, huge_page_size);

    pbs.mem = mmap(NULL, pbs.mem_size, PROT_READ | PROT_WRITE,
                   MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB, -1, 0);
    if (pbs.mem == MAP_FAILED)
    {
        fprintf(stderr, "mmap() failed. Are huge pages configured?\n");
        TEST(posix_memalign(&pbs.mem, huge_page_size, pbs.mem_size) == 0);
    }

    for (i = 0; i < pbs.num; ++i)
    {
        struct pkt_buf *pkt_buf = pkt_buf_from_id(i);
        pkt_buf->id = i;
        pkt_buf_free(pkt_buf);
    }
    return 0;
}
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
static int init()
{
    int i;
    unsigned int vi_flags = EF_VI_FLAGS_DEFAULT;

    TRY(ef_driver_open(&vi.dh));
    TRY(ef_pd_alloc_by_name(&vi.pd, vi.dh, "enp1s0f1", EF_PD_DEFAULT));
    TRY(ef_vi_alloc_from_pd(&vi.vi, vi.dh, &vi.pd, vi.dh, -1,
                            RX_RING_SIZE, TX_RING_SIZE, NULL, -1,
                            (enum ef_vi_flags)vi_flags));

    TRY(ef_memreg_alloc(&vi.memreg, vi.dh, &vi.pd, vi.dh,
                        pbs.mem, pbs.mem_size));

    for (i = 0; i < pbs.num; ++i)
    {
        struct pkt_buf *pkt_buf = pkt_buf_from_id(i);
        pkt_buf->rx_ef_addr = ef_memreg_dma_addr(&vi.memreg, i * PKT_BUF_SIZE + RX_DMA_OFF + addr_offset_from_id(i));
        pkt_buf->tx_ef_addr = ef_memreg_dma_addr(&vi.memreg, i * PKT_BUF_SIZE + RX_DMA_OFF + ef_vi_receive_prefix_len(&vi.vi) + addr_offset_from_id(i));
    }

    assert(ef_vi_receive_capacity(&vi.vi) == RX_RING_SIZE - 1);
    assert(ef_vi_transmit_capacity(&vi.vi) == TX_RING_SIZE - 1);

    while (ef_vi_receive_space(&vi.vi) > REFILL_BATCH_SIZE)
        vi_refill_rx_ring();

    // Set up filters to receive all TCP packets
    ef_filter_spec fs;
    ef_filter_spec_init(&fs, EF_FILTER_FLAG_NONE);
    TRY(ef_filter_spec_set_ip4_full(&fs, IPPROTO_TCP, htonl(0xc0a80d17), htons(1234), htonl(0xc0a80d0a), htons(12345)));
    TRY(ef_vi_filter_add(&vi.vi, vi.dh, &fs, NULL));

    return 0;
}

/**
 * @brief Send a packet with the given payload, payload length, flags, sequence number, and acknowledgment number and frees the buffer
 * Note: seq and ack are numbers to be sent with the packet
 * @param payload
 * @param payload_len
 * @param flags
 * @param seq
 * @param ack
 */

static void send_packet(char *payload, int payload_len, uint8_t flags, uint32_t seq, uint32_t ack)
{
    // initialize packet buffer
    struct pkt_buf *pkt_buf = pbs.free_pool;
    pbs.free_pool = pbs.free_pool->next;
    --pbs.free_pool_n;
    // build packet
    build_tcp_packet(payload, payload_len, flags, seq, ack, (char *)pkt_buf + RX_DMA_OFF + addr_offset_from_id(pkt_buf->id) + ef_vi_receive_prefix_len(&vi.vi));
    // initialize transmit
    int rc = ef_vi_transmit(&vi.vi, pkt_buf->tx_ef_addr, RX_DMA_OFF + addr_offset_from_id(pkt_buf->id) + ef_vi_receive_prefix_len(&vi.vi) + payload_len + sizeof(struct pkt_hdr), pkt_buf->id);
    if (rc != 0)
    {
        throw std::runtime_error("Failed to transmit");
        return;
    }
    pkt_buf_free(pkt_buf);

    return;
}

static void verify_incoming_checksums(struct pkt_hdr *hdr)
{
    uint16_t ip_checksum = ntohs(hdr->ip.check);
    uint16_t tcpchecksum = ntohs(hdr->tcp.check);
    hdr->ip.check = 0;
    hdr->tcp.check = 0;
    assert(ip_checksum == compute_checksum((unsigned short *)&hdr->ip, ((hdr->ip.version_ihl & 0x0F) * 4)));
    assert(tcpchecksum == tcp_checksum(hdr, ntohs(hdr->ip.tot_len) - (uint32_t)((hdr->ip.version_ihl & 0x0F) * 4) - (uint32_t)((hdr->tcp.data_off_reserved >> 4) * 4), ntohs(hdr->ip.tot_len)));
    hdr->ip.check = htons(ip_checksum);
    hdr->tcp.check = htons(tcpchecksum);
}
/*
 * Receive a packet and verify the seq, ack, and flags are as expected, but must free buffer after
 */
static std::tuple<struct pkt_hdr *, uint32_t, uint8_t> receive_packet(uint8_t flags, uint32_t seq, uint32_t ack)
{
    ef_event evs[EF_VI_EVENT_POLL_MIN_EVS];
    uint8_t received_flags = 0;
    while (true)
    {
        int n_ev = ef_eventq_poll(&vi.vi, evs, sizeof(evs) / sizeof(evs[0]));
        for (int i = 0; i < n_ev; ++i)
        {
            switch (EF_EVENT_TYPE(evs[i]))
            {
            case EF_EVENT_TYPE_TX:
                break;
            case EF_EVENT_TYPE_TX_WITH_TIMESTAMP:
                break;
            case EF_EVENT_TYPE_TX_ERROR:
                throw std::runtime_error("Transmit failed");
            case EF_EVENT_TYPE_RX:
            {
                auto id = EF_EVENT_RX_RQ_ID(evs[i]);
                struct pkt_buf *pkt_buf = pkt_buf_from_id(id);
                char *tcp_pkt = (char *)pkt_buf + RX_DMA_OFF + addr_offset_from_id(pkt_buf->id) + ef_vi_receive_prefix_len(&vi.vi);
                struct pkt_hdr *hdr = (struct pkt_hdr *)tcp_pkt;
                verify_incoming_checksums(hdr);
                received_flags = received_flags | hdr->tcp.flags;
                if ((received_flags & flags) == flags)
                {
                    return std::make_tuple(hdr, (uint32_t)ntohs(hdr->ip.tot_len) - (uint32_t)((hdr->ip.version_ihl & 0x0F) * 4) - (uint32_t)((hdr->tcp.data_off_reserved >> 4) * 4), id);
                }
                break;
                /*if ((flags & (uint8_t)TCP_FLAGS::SYN) && (hdr->tcp.flags & (uint8_t)TCP_FLAGS::SYN) == 0)
                {
                    throw std::runtime_error("SYN not received when expected");
                }
                if ((flags & (uint8_t)TCP_FLAGS::ACK) && (hdr->tcp.flags & (uint8_t)TCP_FLAGS::ACK) == 0)
                {
                    throw std::runtime_error("ACK not received when expected");
                }*/
            }
            default:
                throw std::runtime_error("Unexpected event type: " + std::to_string(EF_EVENT_TYPE(evs[i])));
                break;
            }
        }
    }
}

static void send_connection_handshake()
{
    // Send SYN packet
    char *payload = NULL;
    uint32_t payload_len = 0;
    uint8_t flags = 0b00000010;
    send_packet(payload, payload_len, flags, snd_nxt, rcv_nxt);

    // handle SYN-ACK
    flags = (uint8_t)TCP_FLAGS::SYN | (uint8_t)TCP_FLAGS::ACK;
    uint32_t s_seq = 0; // don't care
    uint32_t s_ack = snd_nxt + 1;
    std::cout << "Trying to receive SYN-ACK" << std::endl;
    auto [tcp_pkt, len, id] = receive_packet(flags, s_seq, s_ack);
    std::cout << "Received SYN-ACK" << std::endl;
    uint32_t server_seq = ntohl(tcp_pkt->tcp.seq_num); // this is the seq number of the server
    pkt_buf_free(pkt_buf_from_id(id));
    vi_refill_rx_ring();
    std::cout << "Refilled RX ring" << std::endl;
    snd_nxt += 1;
    rcv_nxt = server_seq + 1;

    // send ACK
    flags = (uint8_t)TCP_FLAGS::ACK;
    snd_una = snd_nxt;
    send_packet(payload, payload_len, flags, snd_nxt, rcv_nxt);

    // send_hello_world();
}

static void send_tcp_teardown()
{

    // send FIN-ACK
    uint8_t flags = (uint8_t)TCP_FLAGS::FIN | (uint8_t)TCP_FLAGS::ACK;
    char *payload = NULL;
    uint32_t payload_len = 0;
    send_packet(payload, payload_len, flags, snd_nxt, rcv_nxt);
    snd_nxt += 1;

    // receive FIN-ACK
    flags = (uint8_t)TCP_FLAGS::ACK | (uint8_t)TCP_FLAGS::FIN;
    auto [tcp_pkt, len, id] = receive_packet(flags, rcv_nxt, snd_nxt);
    pkt_buf_free(pkt_buf_from_id(id));
    vi_refill_rx_ring();
    rcv_nxt += 1;

    // send ACK
    flags = (uint8_t)TCP_FLAGS::ACK;
    payload = NULL;
    payload_len = 0;
    send_packet(payload, payload_len, flags, snd_nxt, rcv_nxt);
    reset_variables();
}

static void send_hello_world()
{
    uint8_t flags = (uint8_t)TCP_FLAGS::ACK;
    char *payload = "Hello World\n";
    size_t payload_len = strlen(payload);
    send_packet(payload, payload_len, flags, snd_nxt, rcv_nxt);
    snd_nxt += payload_len;
    // rcv_next += TODO update rcv_next with response
}

void ef_init_tcp_client()
{

    TRY(init_pkts_memory());
    TRY(init());
    return;
}

void ef_connect()
{
    set_variables();
    send_connection_handshake();
    return;
}

void ef_disconnect()
{
    send_tcp_teardown();
    return;
}

static void send_reset()
{
    uint8_t flags = (uint8_t)TCP_FLAGS::RST;
    char *payload = NULL;
    uint32_t payload_len = 0;
    send_packet(payload, payload_len, flags, snd_nxt, rcv_nxt);
}
/*
    Don't use for buf > 15000
    Futures changes: Event driven system specifically updating state and using a callback to allow strategy to process
*/

static void poll_events(char *buf, ssize_t &read, int len)
{
    ef_event evs[EF_VI_EVENT_POLL_MIN_EVS];
    while (true)
    {
        int n_ev = ef_eventq_poll(&vi.vi, evs, sizeof(evs) / sizeof(evs[0]));
        if (n_ev == 0 || read == len)
        {
            break;
        }
        for (int i = 0; i < n_ev; ++i)
        {
            switch (EF_EVENT_TYPE(evs[i]))
            {
            case EF_EVENT_TYPE_TX:
                break;
            case EF_EVENT_TYPE_TX_WITH_TIMESTAMP:
                break;
            case EF_EVENT_TYPE_TX_ERROR:
                throw std::runtime_error("Transmit failed");
            case EF_EVENT_TYPE_RX:
            {
                auto id = EF_EVENT_RX_RQ_ID(evs[i]);
                struct pkt_buf *pkt_buf = pkt_buf_from_id(id);
                uint32_t offset = RX_DMA_OFF + addr_offset_from_id(id) + ef_vi_receive_prefix_len(&vi.vi);
                struct pkt_hdr *hdr = (struct pkt_hdr *)((char *)pkt_buf + offset);
                verify_incoming_checksums(hdr);
                if (hdr->tcp.flags & (uint8_t)TCP_FLAGS::RST)
                {
                    reset_variables();
                    throw TcpResetException();
                }
                if (hdr->tcp.flags & (uint8_t)TCP_FLAGS::FIN)
                {
                    send_reset();
                    reset_variables();
                    throw TcpResetException();
                }
                if (hdr->tcp.flags & (uint8_t)TCP_FLAGS::ACK)
                {
                    uint32_t ack_num = ntohl(hdr->tcp.ack_num);
                    if (ack_num > snd_una && ack_num <= snd_nxt)
                        snd_una = ack_num;
                    if (ack_num < snd_una)
                    {
                        throw std::runtime_error("Duplicate ACK received");
                    }
                    if (ack_num > snd_nxt)
                    {
                        throw std::runtime_error("Invalid or malicious ACK received");
                    }
                }
                // not factoring in congestion window or window scaling, but this is another check
                uint32_t seq_num = ntohl(hdr->tcp.seq_num);
                ssize_t pay_len = (size_t)ntohs(hdr->ip.tot_len) - (size_t)((hdr->ip.version_ihl & 0x0F) * 4) - (uint32_t)((hdr->tcp.data_off_reserved >> 4) * 4);
                if (seq_num == rcv_nxt)
                {
                    if (hdr->tcp.flags & (uint8_t)TCP_FLAGS::SYN)
                    {
                        throw std::runtime_error("Did not expect SYN since handshake was completed");
                        rcv_nxt += 1;
                    }
                    if (pay_len > 0)
                    {
                        rcv_nxt += pay_len;
                        char *payload = NULL;
                        int payload_len = 0;
                        uint8_t flags = (uint8_t)TCP_FLAGS::ACK;
                        send_packet(payload, payload_len, flags, snd_nxt, rcv_nxt);
                    }
                }
                else if (seq_num > rcv_nxt)
                {
                    throw std::runtime_error("Lost packets somewhere, seeing packet in the future");
                }
                else
                {
                    throw std::runtime_error("Seeing packet from past, sender is retransmitting");
                }
                if (len == read)
                {
                    data_queue.push(std::make_tuple((char *)hdr + sizeof(struct pkt_hdr), pay_len, id));
                }
                else if (read > len)
                {
                    throw std::runtime_error("Read more than len, should not be reachable state");
                }
                else
                {
                    if (len < pay_len + read)
                    {
                        memcpy(buf, (char *)hdr + sizeof(struct pkt_hdr), len - read);
                        read = len;
                        data_queue.push(std::make_tuple((char *)hdr + sizeof(struct pkt_hdr) + len - read, pay_len - (len - read), id));
                        // rest is pay_len - (len - read)
                    }
                    else
                    {
                        memcpy(buf, (char *)hdr + sizeof(struct pkt_hdr), pay_len);
                        read += pay_len;
                        pkt_buf_free(pkt_buf);
                        vi_refill_rx_ring();
                    }
                }
                break;
            }
            default:
                throw std::runtime_error("Unexpected event type: " + std::to_string(EF_EVENT_TYPE(evs[i])));
                break;
            }
        }
    }
}

static void poll_events()
{
    ef_event evs[EF_VI_EVENT_POLL_MIN_EVS];
    while (true)
    {
        int n_ev = ef_eventq_poll(&vi.vi, evs, sizeof(evs) / sizeof(evs[0]));
        if (n_ev == 0)
        {
            break;
        }
        for (int i = 0; i < n_ev; ++i)
        {
            switch (EF_EVENT_TYPE(evs[i]))
            {
            case EF_EVENT_TYPE_TX:
                break;
            case EF_EVENT_TYPE_TX_WITH_TIMESTAMP:
                break;
            case EF_EVENT_TYPE_TX_ERROR:
                throw std::runtime_error("Transmit failed");
            case EF_EVENT_TYPE_RX:
            {
                auto id = EF_EVENT_RX_RQ_ID(evs[i]);
                struct pkt_buf *pkt_buf = pkt_buf_from_id(id);
                uint32_t offset = RX_DMA_OFF + addr_offset_from_id(id) + ef_vi_receive_prefix_len(&vi.vi);
                struct pkt_hdr *hdr = (struct pkt_hdr *)((char *)pkt_buf + offset);
                verify_incoming_checksums(hdr);
                if (hdr->tcp.flags & (uint8_t)TCP_FLAGS::RST)
                {
                    reset_variables();
                    throw TcpResetException();
                }
                if (hdr->tcp.flags & (uint8_t)TCP_FLAGS::FIN)
                {
                    send_reset();
                    reset_variables();
                    throw TcpResetException();
                }
                if (hdr->tcp.flags & (uint8_t)TCP_FLAGS::ACK)
                {
                    uint32_t ack_num = ntohl(hdr->tcp.ack_num);
                    if (ack_num > snd_una && ack_num <= snd_nxt)
                        snd_una = ack_num;
                    if (ack_num < snd_una)
                    {
                        throw std::runtime_error("Duplicate ACK received");
                    }
                    if (ack_num > snd_nxt)
                    {
                        throw std::runtime_error("Invalid or malicious ACK received");
                    }
                }
                // not factoring in congestion window or window scaling, but this is another check
                uint32_t seq_num = ntohl(hdr->tcp.seq_num);
                ssize_t pay_len = (size_t)ntohs(hdr->ip.tot_len) - (size_t)((hdr->ip.version_ihl & 0x0F) * 4) - (size_t)((hdr->tcp.data_off_reserved >> 4) * 4);
                if (seq_num == rcv_nxt)
                {
                    if (hdr->tcp.flags & (uint8_t)TCP_FLAGS::SYN)
                    {
                        throw std::runtime_error("Did not expect SYN since handshake was completed");
                        rcv_nxt += 1;
                    }
                    if (pay_len > 0)
                    {
                        rcv_nxt += pay_len;
                        char *payload = NULL;
                        int payload_len = 0;
                        uint8_t flags = (uint8_t)TCP_FLAGS::ACK;
                        send_packet(payload, payload_len, flags, snd_nxt, rcv_nxt);
                    }
                }
                else if (seq_num > rcv_nxt)
                {
                    throw std::runtime_error("Lost packets somewhere, seeing packet in the future");
                }
                else
                {
                    throw std::runtime_error("Seeing packet from past, sender is retransmitting");
                }
                data_queue.push(std::make_tuple((char *)hdr + sizeof(struct pkt_hdr), pay_len, id));
                break;
            }
            default:
                throw std::runtime_error("Unexpected event type: " + std::to_string(EF_EVENT_TYPE(evs[i])));
                break;
            }
        }
    }
}

ssize_t ef_read(char *buf, int len)
{ // in theory can use a parser to read the packet and apply a callback to strategy, but beyond scope here

    ssize_t read = 0;
    while (read < len && !data_queue.empty())
    {
        auto [payload, payload_len, id] = data_queue.front();
        if (payload_len > len - read)
        {
            memcpy(buf + read, payload, len - read);
            read = len;
            std::get<1>(data_queue.front()) -= len - read;
            std::get<0>(data_queue.front()) += len - read;
        }
        else
        {
            memcpy(buf + read, payload, payload_len);
            read += payload_len;
            data_queue.pop();
            pkt_buf_free(pkt_buf_from_id(id));
            vi_refill_rx_ring();
        }
    }

    poll_events(buf, read, len);
    return read;
}

ssize_t ef_send(char *buf, int len)
{
    if (len >= 1460)
    {
        throw std::runtime_error("Payload length too large");
    }
    uint8_t flags = (uint8_t)TCP_FLAGS::ACK | (uint8_t)TCP_FLAGS::PSH;
    char *payload = buf;
    uint32_t payload_len = len;
    send_packet(payload, payload_len, flags, snd_nxt, rcv_nxt);
    snd_nxt += payload_len;
    poll_events();
    return 0;
}

/*
For use with generalized event driven rx handling
if (EF_EVENT_TYPE(evs[i]) == EF_EVENT_TYPE_RX) {
            auto id = EF_EVENT_RX_RQ_ID(evs[i]);
            char* pkt = pkt_bufs + id * BUF_SIZE; // need to use the other function
            size_t len = EF_EVENT_RX_BYTES(evs[i]);
            handle_packet(pkt, len);
            ef_vi_receive_init(&vi, ef_memreg_dma_addr(&mr, id * BUF_SIZE), id);
        }

*/

void dump_buffer(const uint8_t *buf, size_t len)
{
    for (size_t i = 0; i < len; i++)
    {
        if (i % 16 == 0)
            printf("%04zx: ", i);
        printf("%02x ", buf[i]);
        if (i % 16 == 15 || i == len - 1)
            printf("\n");
    }
}