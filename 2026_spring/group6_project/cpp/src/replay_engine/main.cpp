#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

#include "ndfex_gateway.h"
#include "replay_engine.h"

namespace {
volatile std::sig_atomic_t g_running = 1;
void on_signal(int) { g_running = 0; }
} // namespace

static void usage(const char *prog) {
  std::cerr
      << "Usage: " << prog
      << " [oe_port md_host md_port snap_host snap_port udp_iface]"
         " <live_feed.bin> <snapshot.bin>\n"
      << "  Defaults: 3000 239.255.0.1 12345 239.255.0.2 12346 127.0.0.1\n";
}

int main(int argc, char **argv) {
  std::uint16_t oe_port = 3000;
  std::string md_host = "239.255.0.1";
  std::uint16_t md_port = 12345;
  std::string snap_host = "239.255.0.2";
  std::uint16_t snap_port = 12346;
  std::string udp_iface = "127.0.0.1";

  std::filesystem::path live_path;
  std::filesystem::path snap_path;

  if (argc == 3) {
    live_path = argv[1];
    snap_path = argv[2];
  } else if (argc == 9) {
    oe_port = static_cast<std::uint16_t>(std::stoi(argv[1]));
    md_host = argv[2];
    md_port = static_cast<std::uint16_t>(std::stoi(argv[3]));
    snap_host = argv[4];
    snap_port = static_cast<std::uint16_t>(std::stoi(argv[5]));
    udp_iface = argv[6];
    live_path = argv[7];
    snap_path = argv[8];
  } else {
    usage(argv[0]);
    return 1;
  }

  ghostbook::replay::ReplayEngine engine;

  if (!engine.load_snapshot(snap_path)) {
    std::cerr << "error: failed to load snapshot " << snap_path << "\n";
    return 1;
  }
  std::cout << "snapshot loaded: " << snap_path << "\n";

  if (!engine.open_feed(live_path)) {
    std::cerr << "error: failed to open feed " << live_path << "\n";
    return 1;
  }
  std::cout << "feed opened: " << live_path << "\n";

  ghostbook::gateway::NdfexGateway gw(std::move(engine));

  if (!gw.start(oe_port, md_host, md_port, snap_host, snap_port, udp_iface)) {
    std::cerr << "error: gateway start failed\n";
    return 1;
  }
  std::cout << "gateway running (OE :" << oe_port << ")\n";

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  while (g_running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  gw.shutdown();
  std::cout << "shutdown complete\n";
  return 0;
}
