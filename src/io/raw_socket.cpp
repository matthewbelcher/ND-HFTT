#include "io/raw_socket.h"

#include <stdexcept>
#include <cerrno>
#include <cstring>
#include <utility>

#include <unistd.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <linux/if_packet.h>
#include <linux/if_ether.h>  // ETH_P_ALL, ETH_ALEN (avoid <net/ethernet.h> which
                             // collides with our include/net/ethernet.h)

RawSocket::RawSocket(const std::string& iface)
    : fd_(-1), if_index_(-1), iface_(iface)
{
    // ETH_P_ALL captures every Ethernet frame regardless of EtherType.
    fd_ = ::socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (fd_ < 0)
        throw std::runtime_error(std::string("socket(): ") + std::strerror(errno));

    // Resolve interface name → ifindex.
    struct ifreq ifr{};
    std::strncpy(ifr.ifr_name, iface.c_str(), IFNAMSIZ - 1);
    if (::ioctl(fd_, SIOCGIFINDEX, &ifr) < 0) {
        ::close(fd_);
        throw std::runtime_error(std::string("ioctl(SIOCGIFINDEX): ") + std::strerror(errno));
    }
    if_index_ = ifr.ifr_ifindex;

    // Bind so we only see/send frames on this interface.
    struct sockaddr_ll sll{};
    sll.sll_family   = AF_PACKET;
    sll.sll_protocol = htons(ETH_P_ALL);
    sll.sll_ifindex  = if_index_;

    if (::bind(fd_, reinterpret_cast<struct sockaddr*>(&sll), sizeof(sll)) < 0) {
        ::close(fd_);
        throw std::runtime_error(std::string("bind(): ") + std::strerror(errno));
    }
}

RawSocket::~RawSocket() {
    if (fd_ >= 0)
        ::close(fd_);
}

RawSocket::RawSocket(RawSocket&& other) noexcept
    : fd_(other.fd_), if_index_(other.if_index_), iface_(std::move(other.iface_))
{
    other.fd_       = -1;
    other.if_index_ = -1;
}

RawSocket& RawSocket::operator=(RawSocket&& other) noexcept {
    if (this != &other) {
        if (fd_ >= 0)
            ::close(fd_);
        fd_       = other.fd_;
        if_index_ = other.if_index_;
        iface_    = std::move(other.iface_);
        other.fd_       = -1;
        other.if_index_ = -1;
    }
    return *this;
}

int RawSocket::send_frame(const void* buf, size_t len) {
    // sockaddr_ll tells the kernel which interface to egress on.
    // For SOCK_RAW the full Ethernet frame (including dst/src MAC) is already
    // in buf; sll_addr mirrors the dst MAC for the kernel's L2 routing path.
    struct sockaddr_ll sll{};
    sll.sll_family  = AF_PACKET;
    sll.sll_ifindex = if_index_;
    sll.sll_halen   = ETH_ALEN;
    if (len >= static_cast<size_t>(ETH_ALEN))
        std::memcpy(sll.sll_addr, buf, ETH_ALEN);

    ssize_t n = ::sendto(fd_, buf, len, 0,
                         reinterpret_cast<const struct sockaddr*>(&sll), sizeof(sll));
    return static_cast<int>(n);
}

int RawSocket::recv_frame(void* buf, size_t len) {
    ssize_t n = ::recvfrom(fd_, buf, len, 0, nullptr, nullptr);
    return static_cast<int>(n);
}
