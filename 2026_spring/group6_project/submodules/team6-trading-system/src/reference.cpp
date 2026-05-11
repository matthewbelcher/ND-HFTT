#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>
#include <sys/epoll.h>
#include <fcntl.h>

#include <vector>
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <cstring>

int main(int argc, char* argv[]) {

    if (argc < 4) {
        std::cerr << "Usage " << argv[0] << "(<mcast addr> <mcast port>)+ <local ip>" << std::endl;
        return 1;
    }

    std::vector<int> mcast_socks;

    for (int i = 1; i < argc - 1; i +=2) {
        // create socket
        int mcast_fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (mcast_fd < 0) {
            std::cerr << "Failed to create socket" << strerror(errno) << std::endl;
            return 1;
        }

        // set socket to non-blocking
        int flags = fcntl(mcast_fd, F_GETFL, 0);
        if (flags < 0) {
            std::cerr << "Could not get flags " << strerror(errno) << std::endl;
            return 1;
        }

        flags = fcntl(mcast_fd, F_SETFL, flags | O_NONBLOCK);
        if (flags< 0 ) {
            std::cerr << "Could not set socket to non-blocking " << std::endl;
            return 1;
        }

        int val = 1;
        int ret = setsockopt(mcast_fd, SOL_SOCKET, SO_REUSEPORT, &val, sizeof(int));

        // bind to the multicast group (only this multicast group will be read on this socket)
        struct sockaddr_in mcast_addr;
        mcast_addr.sin_addr.s_addr = inet_addr(argv[i]);
        mcast_addr.sin_port = htons(atoi(argv[i+1]));
        mcast_addr.sin_family = AF_INET;

        int rc = bind(mcast_fd, (const sockaddr*) &mcast_addr, sizeof(mcast_addr));
        if (rc < 0) {
            std::cerr << "Failed to bind mcast address " << strerror(errno) << std::endl;
            return 1;
        }

        // send the multicast subscription
        struct ip_mreq mreq;
        mreq.imr_multiaddr.s_addr = inet_addr(argv[i]);
        mreq.imr_interface.s_addr = inet_addr(argv[argc-1]);

        rc = setsockopt(mcast_fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq));
        if (rc < 0) {
            std::cerr << "Failed to join mcast group " << strerror(errno) << std::endl;
            return 1;
        }
        mcast_socks.push_back(mcast_fd);
    }

    // creat the epoll fd
    int epoll_fd = epoll_create(128);

    for (int sock : mcast_socks) {

        // add each mcast sock to the epoll set looking for EPOLLIN events
        // will trigger whenever read is available
        epoll_event event;
        event.data.fd = sock;
        event.events = EPOLLIN;
        int ret = epoll_ctl(epoll_fd, EPOLL_CTL_ADD, sock, &event );
        if (ret < 0 ) {
            std::cerr << "epoll add failed " << strerror(errno) << std::endl;
        }

    }

    char buf[1500];
    while (true) {
        epoll_event events[16];
        // wait on epoll events
        int nfds = epoll_wait(epoll_fd, events, 16, 0);
        if (nfds > 0) {
            for (int i = 0; i < nfds; ++i) {
                ssize_t bytes = read(events[i].data.fd, buf, sizeof(buf));
                if (bytes < 0 && errno != EAGAIN) {
                    std::cerr << "Error on read " << strerror(errno) << std::endl;
                    return 1;
                }
                if (bytes > 0) {
                    std::cout << "Read " << bytes << std::endl;
                    for (int b = 0; b < bytes; ++b) {
                        printf("%hhx:", buf[b]);
                    }
                    std::cout << std::endl;
                }
            }
        }
    }
    return 0;
}