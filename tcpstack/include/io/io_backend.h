#ifndef HFT_TCPSTACK_IO_BACKEND_H
#define HFT_TCPSTACK_IO_BACKEND_H

#include <cstddef>

// Abstract I/O interface for raw Ethernet frame I/O.
// Implementations: RawSocket (AF_PACKET, Phase 1), DPDKPort (Phase 5).
class IOBackend {
public:
    virtual ~IOBackend() = default;

    // Send a raw Ethernet frame.
    // Returns bytes sent on success, or -1 on error (errno set).
    virtual int send_frame(const void* buf, size_t len) = 0;

    // Receive a raw Ethernet frame into buf (capacity len bytes).
    // Blocks until a frame arrives.
    // Returns bytes received, or -1 on error (errno set).
    virtual int recv_frame(void* buf, size_t len) = 0;
};

#endif //HFT_TCPSTACK_IO_BACKEND_H
