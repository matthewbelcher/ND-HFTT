#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "tcp_gateway.h"

namespace {

std::uint16_t parse_port(const char* value, const char* name) {
    if (value == nullptr) {
        throw std::runtime_error(std::string(name) + " is required");
    }

    const auto parsed = std::strtoul(value, nullptr, 10);
    if (parsed == 0 || parsed > 65535) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }

    return static_cast<std::uint16_t>(parsed);
}

void print_usage() {
    std::cerr << "Usage: gateway [port] [market_data_host] [market_data_port] [snapshot_port]\n"
              << "Defaults: 1234 239.255.0.1 12345 1235\n";
}

}  // namespace

int main(int argc, char** argv) {
    using namespace ghostbook::gateway;

    std::uint16_t port = 1234;
    std::string market_data_host = "239.255.0.1";
    std::uint16_t market_data_port = 12345;
    std::uint16_t snapshot_port = 1235;

    try {
        if (argc > 1) {
            port = parse_port(argv[1], "port");
        }
        if (argc > 2) {
            market_data_host = argv[2];
        }
        if (argc > 3) {
            market_data_port = parse_port(argv[3], "market_data_port");
        }
        if (argc > 4) {
            snapshot_port = parse_port(argv[4], "snapshot_port");
        }
        if (argc > 5) {
            print_usage();
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        print_usage();
        return 1;
    }

    auto gateway = std::make_unique<TcpGateway>(0);

    if (!gateway->start(port, market_data_host, market_data_port, snapshot_port)) {
        std::cerr << "Failed to start gateway" << std::endl;
        return 1;
    }

    std::cout << "Gateway started successfully" << std::endl;

    // Keep running
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}
