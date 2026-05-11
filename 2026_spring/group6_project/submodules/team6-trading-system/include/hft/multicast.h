#ifndef TRADING_PROJECT_MULTICAST_H
#define TRADING_PROJECT_MULTICAST_H

#include <cstdint>
#include <arpa/inet.h>

namespace hft::mcast {
    /* Class Templates */

    /**
     * Multicast UDP listener object
     */
    class MulticastListener {
        ip_mreq mcast_req{};

        public:
            int socket_fd;
            MulticastListener(in_addr_t mcast_ip_addr, int mcast_port, in_addr_t local_ip_addr);
            ~MulticastListener();

            ssize_t receive(uint8_t *buf, ssize_t len) const;
    };
}


#endif //TRADING_PROJECT_MULTICAST_H