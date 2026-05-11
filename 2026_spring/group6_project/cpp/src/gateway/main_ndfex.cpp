#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include "ndfex_gateway.h"

namespace {

std::uint16_t parse_port(const char *value, const char *name) {
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
  std::cerr << "Usage: ndfex_gateway [oe_port] [md_host] [md_port] [snap_host] "
               "[snap_port] [udp_interface]\n"
            << "Defaults: 3000 239.255.0.1 12345 239.255.0.2 12346 127.0.0.1\n";
}

} // namespace

int main(int argc, char **argv) {
  using namespace ghostbook::gateway;

  std::uint16_t oe_port = 3000;
  std::string md_host = "239.255.0.1";
  std::uint16_t md_port = 12345;
  std::string snap_host = "239.255.0.2";
  std::uint16_t snap_port = 12346;
  std::string udp_interface = "127.0.0.1";

  try {
    if (argc > 1)
      oe_port = parse_port(argv[1], "oe_port");
    if (argc > 2)
      md_host = argv[2];
    if (argc > 3)
      md_port = parse_port(argv[3], "md_port");
    if (argc > 4)
      snap_host = argv[4];
    if (argc > 5)
      snap_port = parse_port(argv[5], "snap_port");
    if (argc > 6)
      udp_interface = argv[6];
    if (argc > 7) {
      print_usage();
      return 1;
    }
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n";
    print_usage();
    return 1;
  }

  auto gateway = std::make_unique<NdfexGateway>(0);
  if (!gateway->start(oe_port, md_host, md_port, snap_host, snap_port,
                      udp_interface)) {
    std::cerr << "Failed to start NDFEX gateway\n";
    return 1;
  }

  std::cout << "NDFEX gateway started — OE port " << oe_port << ", MD "
            << md_host << ":" << md_port << ", snapshot " << snap_host << ":"
            << snap_port << " (interface: " << udp_interface << ")\n";

  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  return 0;
}
