#include "io/dpdk_port.h"

#include <stdexcept>
#include <string>
#include <cstring>

#include <rte_eal.h>
#include <rte_errno.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_mempool.h>

static constexpr uint16_t RX_QUEUE_ID  = 0;
static constexpr uint16_t TX_QUEUE_ID  = 0;
static constexpr uint16_t NB_QUEUES    = 1;
static constexpr uint16_t BURST_SIZE   = 32;
static constexpr uint16_t DATA_ROOM_SZ = RTE_MBUF_DEFAULT_BUF_SIZE; // 2176 bytes

// EAL is a process-lifetime resource; init only once.
static bool s_eal_initialised = false;

DPDKPort::DPDKPort(const Config& cfg)
    : port_id_(cfg.port_id), mempool_(nullptr)
{
    if (!s_eal_initialised) {
        const int ret = rte_eal_init(cfg.eal_argc, cfg.eal_argv);
        if (ret < 0)
            throw std::runtime_error(
                std::string("rte_eal_init failed: ") + rte_strerror(rte_errno));
        s_eal_initialised = true;
    }

    if (!rte_eth_dev_is_valid_port(port_id_))
        throw std::runtime_error(
            "DPDK port " + std::to_string(port_id_) + " not found");

    // Mbuf pool: each entry holds one Ethernet frame up to DATA_ROOM_SZ bytes.
    mempool_ = rte_pktmbuf_pool_create(
        "hft_pool", cfg.pool_size, /*cache_size=*/256,
        /*priv_size=*/0, DATA_ROOM_SZ, rte_socket_id());
    if (!mempool_)
        throw std::runtime_error(
            std::string("rte_pktmbuf_pool_create failed: ") + rte_strerror(rte_errno));

    // Configure port with default settings (no offloads needed for correctness).
    struct rte_eth_conf port_conf{};
    int ret = rte_eth_dev_configure(port_id_, NB_QUEUES, NB_QUEUES, &port_conf);
    if (ret < 0)
        throw std::runtime_error(
            "rte_eth_dev_configure failed: " + std::to_string(-ret));

    // RX queue.
    ret = rte_eth_rx_queue_setup(
        port_id_, RX_QUEUE_ID, cfg.rx_descs,
        rte_eth_dev_socket_id(port_id_), nullptr, mempool_);
    if (ret < 0)
        throw std::runtime_error(
            "rte_eth_rx_queue_setup failed: " + std::to_string(-ret));

    // TX queue.
    ret = rte_eth_tx_queue_setup(
        port_id_, TX_QUEUE_ID, cfg.tx_descs,
        rte_eth_dev_socket_id(port_id_), nullptr);
    if (ret < 0)
        throw std::runtime_error(
            "rte_eth_tx_queue_setup failed: " + std::to_string(-ret));

    ret = rte_eth_dev_start(port_id_);
    if (ret < 0)
        throw std::runtime_error(
            "rte_eth_dev_start failed: " + std::to_string(-ret));

    // Receive all frames regardless of destination MAC.
    rte_eth_promiscuous_enable(port_id_);
}

DPDKPort::~DPDKPort() {
    rte_eth_dev_stop(port_id_);
    rte_eth_dev_close(port_id_);
    if (mempool_) {
        rte_mempool_free(mempool_);
        mempool_ = nullptr;
    }
    // EAL cleanup is process-lifetime; call once via atexit or at program end.
    // Calling rte_eal_cleanup() here would break any other live DPDKPort.
}

int DPDKPort::send_frame(const void* buf, size_t len) {
    struct rte_mbuf* m = rte_pktmbuf_alloc(mempool_);
    if (!m) return -1;

    if (len > rte_pktmbuf_tailroom(m)) {
        rte_pktmbuf_free(m);
        return -1;
    }

    std::memcpy(rte_pktmbuf_mtod(m, void*), buf, len);
    m->data_len = static_cast<uint16_t>(len);
    m->pkt_len  = static_cast<uint32_t>(len);

    const uint16_t sent = rte_eth_tx_burst(port_id_, TX_QUEUE_ID, &m, 1);
    if (sent == 0) {
        rte_pktmbuf_free(m);
        return -1;  // TX ring full; caller may retry
    }
    return static_cast<int>(len);
}

int DPDKPort::recv_frame(void* buf, size_t len) {
    struct rte_mbuf* mbufs[BURST_SIZE];

    // Busy-poll: no interrupt, no sleep — this is the DPDK model.
    uint16_t nb_rx = 0;
    while (nb_rx == 0)
        nb_rx = rte_eth_rx_burst(port_id_, RX_QUEUE_ID, mbufs, BURST_SIZE);

    // Deliver the first frame; discard any extras from this burst.
    // For high-throughput paths use recv_burst() to drain the full burst.
    const uint32_t frame_len = mbufs[0]->pkt_len;
    int result = -1;
    if (static_cast<size_t>(frame_len) <= len) {
        std::memcpy(buf, rte_pktmbuf_mtod(mbufs[0], const void*), frame_len);
        result = static_cast<int>(frame_len);
    }

    for (uint16_t i = 0; i < nb_rx; ++i)
        rte_pktmbuf_free(mbufs[i]);

    return result;
}

void DPDKPort::recv_burst(const std::function<void(const void*, size_t)>& cb) {
    struct rte_mbuf* mbufs[BURST_SIZE];

    uint16_t nb_rx = 0;
    while (nb_rx == 0)
        nb_rx = rte_eth_rx_burst(port_id_, RX_QUEUE_ID, mbufs, BURST_SIZE);

    for (uint16_t i = 0; i < nb_rx; ++i) {
        // cb receives a pointer directly into the mbuf — no memcpy.
        cb(rte_pktmbuf_mtod(mbufs[i], const void*),
           static_cast<size_t>(mbufs[i]->data_len));
        rte_pktmbuf_free(mbufs[i]);
    }
}
