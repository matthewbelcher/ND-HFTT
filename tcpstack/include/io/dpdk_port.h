#ifndef HFT_TCPSTACK_DPDK_PORT_H
#define HFT_TCPSTACK_DPDK_PORT_H

#include "io_backend.h"

#include <cstdint>
#include <functional>

struct rte_mempool;  // forward-declared to avoid pulling DPDK headers transitively

// DPDK poll-mode driver backend.
// Replaces RawSocket with kernel-bypass I/O for minimum-latency packet I/O.
//
// Typical usage (WSL2 development with net_tap, no real NIC):
//
//   const char* args[] = {
//       "prog",
//       "--vdev", "net_tap0,iface=tap0",
//       "--no-pci", "--no-huge", "--iova-mode=va",
//       nullptr
//   };
//   DPDKPort::Config cfg;
//   cfg.eal_argc = 6;
//   cfg.eal_argv = const_cast<char**>(args);
//   DPDKPort port(cfg);
//   port.send_frame(buf, len);
//   port.recv_frame(buf, sizeof(buf));  // busy-polls until a frame arrives
//
// Bare-metal usage (real NIC, hugepages configured):
//   Bind NIC to vfio-pci or uio_pci_generic first:
//     dpdk-devbind.py --bind=vfio-pci 0000:01:00.0
//   Then pass EAL args without --no-huge / --no-pci.
//
// Thread safety: not thread-safe. Call all methods from the same thread.
//
// recv_frame busy-polls the NIC in a tight loop (no interrupts, no blocking).
// Pin the calling thread to a dedicated core for best latency results.
class DPDKPort : public IOBackend {
public:
    struct Config {
        int       eal_argc;               // passed verbatim to rte_eal_init
        char**    eal_argv;               // passed verbatim to rte_eal_init
        uint16_t  port_id   = 0;          // DPDK port index (usually 0)
        uint16_t  rx_descs  = 512;        // RX ring descriptor count
        uint16_t  tx_descs  = 512;        // TX ring descriptor count
        uint32_t  pool_size = 4095;       // mbuf pool size; best as 2^n - 1
    };

    // Initialises EAL, creates the mbuf pool, configures and starts the port.
    // Throws std::runtime_error on any failure.
    // EAL is initialised at most once per process; subsequent DPDKPort
    // instances reuse the existing EAL context.
    explicit DPDKPort(const Config& cfg);
    ~DPDKPort() override;

    DPDKPort(const DPDKPort&)            = delete;
    DPDKPort& operator=(const DPDKPort&) = delete;

    // IOBackend interface

    // Copies buf into an mbuf and calls rte_eth_tx_burst.
    // Returns bytes sent on success, -1 if the TX ring is full or allocation fails.
    int send_frame(const void* buf, size_t len) override;

    // Busy-polls rte_eth_rx_burst until a frame arrives, copies the first frame
    // into buf (up to len bytes), frees all mbufs in the burst.
    // Returns bytes received on success, -1 if the frame exceeds len.
    // Note: frames beyond the first in a burst are discarded; use recv_burst()
    // to drain a full burst without extra copies.
    int recv_frame(void* buf, size_t len) override;

    // Zero-copy RX: drains one burst and invokes cb(data, len) for each frame
    // with a pointer directly into the mbuf data room. The pointer is only valid
    // during cb; do not store it. Avoids the memcpy in recv_frame().
    void recv_burst(const std::function<void(const void*, size_t)>& cb);

    uint16_t port_id() const { return port_id_; }

private:
    uint16_t     port_id_;
    rte_mempool* mempool_;
};

#endif // HFT_TCPSTACK_DPDK_PORT_H
