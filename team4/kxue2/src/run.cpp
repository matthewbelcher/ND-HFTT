#include "ef_send_tcp.hpp"
#include <signal.h>

static bool running = true;

void signal_handler(int signum) {
    if (signum == SIGINT) {
        std::cout << "\nReceived Ctrl+C, disconnecting..." << std::endl;
        running = false;
        //ef_disconnect();
        exit(0);
    }
}

int main(int argc, char *argv[])
{

    // 1. Sample Client to Server Communication
    /* 
    ef_init_tcp_client();
    ef_connect();
    ef_send("Hello HFTT Class\n", 17);
    ef_send("My name is Kevin\n", 17);
    ef_send("What is your name?\n", 19);
    ef_disconnect();
    return 0;
    */

    // 2. Sample Server to Client Communication
    // Set up signal handler
    
    signal(SIGINT, signal_handler);
    ef_init_tcp_client();
    ef_connect();
    char buf[1500];
    while (true) {
        ssize_t bytes = ef_read(buf, 1500);
        if (bytes > 0) {
            buf[bytes] = '\0';
            std::cout << buf;
        }
    }
    ef_disconnect();
    return 0;
}