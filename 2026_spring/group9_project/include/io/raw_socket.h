#ifndef HFT_TCPSTACK_RAW_SOCKET_H
#define HFT_TCPSTACK_RAW_SOCKET_H

#include "io_backend.h"
#include <string>

// AF_PACKET / SOCK_RAW backend.
// Sends and receives raw Ethernet frames on a named network interface.
// Requires CAP_NET_RAW (run as root or with the capability set).
class RawSocket : public IOBackend {
public:
    // Opens AF_PACKET/SOCK_RAW and binds to `iface` (e.g. "eth0", "lo").
    // Throws std::runtime_error if the socket cannot be created or bound.
    explicit RawSocket(const std::string& iface);
    ~RawSocket() override;

    RawSocket(const RawSocket&)            = delete;
    RawSocket& operator=(const RawSocket&) = delete;
    RawSocket(RawSocket&&) noexcept;
    RawSocket& operator=(RawSocket&&) noexcept;

    // IOBackend interface
    int send_frame(const void* buf, size_t len) override;
    int recv_frame(void*       buf, size_t len) override;

    int fd()       const { return fd_; }
    int if_index() const { return if_index_; }

private:
    int         fd_;
    int         if_index_;
    std::string iface_;
};

#endif //HFT_TCPSTACK_RAW_SOCKET_H
