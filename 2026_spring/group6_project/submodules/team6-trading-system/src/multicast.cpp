#include "../include/hft/multicast.h"

#include <arpa/inet.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sys/fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

namespace hft::mcast {
/* Class Method Implementations */

/**
 * Create a UDP multicast listener object
 *
 * @param mcast_ip_addr IP address of multicast group
 * @param mcast_port UDP port for multicast group
 * @param local_ip_addr IP address of local network interface
 *
 * @throws std::runtime_error If multicast operations fail
 */
MulticastListener::MulticastListener(const in_addr_t mcast_ip_addr,
                                     const int mcast_port,
                                     const in_addr_t local_ip_addr) {

  this->socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (this->socket_fd < 0) {
    throw std::runtime_error("Failed to create multicast socket");
  }

  // Set to non-blocking
  const int flags = fcntl(this->socket_fd, F_GETFL, 0);
  if (flags < 0) {
    throw std::runtime_error("Failed to get socket flags");
  }
  if (fcntl(this->socket_fd, F_SETFL, flags | O_NONBLOCK) < 0) {
    throw std::runtime_error("Failed to set socket flags");
  }

  // Set reuse addr/port option
  int val = 1;
  int ret1 = setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &val, sizeof(val));
  if (ret1 < 0) {
    throw std::runtime_error("Failed to set SO_REUSADDR on multicast socket");
  }

  // Bind to the multicast group (only this multicast group will be read on this
  // socket)
  sockaddr_in mcast_addr = {};
  mcast_addr.sin_addr.s_addr = mcast_ip_addr;
  mcast_addr.sin_port = htons(mcast_port);
  mcast_addr.sin_family = AF_INET;

  if (bind(this->socket_fd, reinterpret_cast<sockaddr *>(&mcast_addr),
           sizeof(mcast_addr)) < 0) {
    throw std::runtime_error("Failed to bind multicast socket");
  }

  // Send the multicast subscription
  this->mcast_req = {};
  this->mcast_req.imr_multiaddr.s_addr = mcast_ip_addr;
  this->mcast_req.imr_interface.s_addr = local_ip_addr;

  if (setsockopt(this->socket_fd, IPPROTO_IP, IP_ADD_MEMBERSHIP,
                 &this->mcast_req, sizeof(this->mcast_req)) < 0) {
    throw std::runtime_error("Failed to join multicast group");
  }
}

/**
 * Shut down UDP multicast listener
 */
MulticastListener::~MulticastListener() {
  if (setsockopt(this->socket_fd, IPPROTO_IP, IP_DROP_MEMBERSHIP,
                 &this->mcast_req, sizeof(this->mcast_req)) < 0) {
    std::cerr << "Failed to drop multicast group membership: "
              << strerror(errno) << std::endl;
  }
  close(this->socket_fd);
}

/**
 * Receive available data from multicast
 *
 * @param buf Location to write data
 * @param len Maximum length to write
 * @return Length of data received
 */
ssize_t MulticastListener::receive(uint8_t *buf, const ssize_t len) const {
  const ssize_t n = recv(this->socket_fd, buf, len, 0);
  if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
    throw std::runtime_error("Error receiving data from multicast group");
  }

  return n;
}
} // namespace hft::mcast
