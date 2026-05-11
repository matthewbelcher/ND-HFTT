#include <arpa/inet.h>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <hft/ExchangeClient.h>
#include <hft/PositionTracker.h>
#include <hft/RiskSystem.h>
#include <hft/multicast.h>
#include <hft/orderbook.h>

#include "strategies/Liquidate.h"

constexpr uint32_t    NDFEX_CLIENT_ID = 6;
constexpr const char *NDFEX_USERNAME  = "team6";
constexpr const char *NDFEX_PASSWORD  = "56vjGN5S";
constexpr int         CPU_ID          = 5;

const hft::client::ConnectionSpec NDFEX_SPEC{
    inet_addr("192.168.13.14"),
    {inet_addr("239.0.0.1"), 12345},
    {inet_addr("239.0.0.2"), 12345},
    {inet_addr("192.168.13.100"), 1234}};

// Generous limits: we are only liquidating, not building position.
// PNL shutdown is disabled (large negative) so an injected zero-cost
// basis doesn't trigger a premature halt.
const hft::risk::RiskLimits RISK_LIMITS{
    .max_qty_per_order        = 10,
    .max_qty_per_side         = 10,
    .max_exposure_per_side    = 2000,
    .max_position             = 10,
    .max_abs_position_shutdown = 100,   // won't fire during normal liquidation
    .min_pnl_shutdown         = -1e9,   // disabled — injected fills have price 0
    .max_orders_per_second    = 10,
    .max_orders_per_md_update = 10,
    .max_inflight_orders      = 10,
    .min_valid_price          = 2,
    .max_valid_price          = 100000};

std::atomic<bool> interrupted(false);

static void interruption_handler(int sig) {
  if (sig == SIGINT)
    interrupted = true;
}

static void setup_affinity(int cpu_id) {
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  CPU_SET(cpu_id, &cpu_set);
  if (sched_setaffinity(0, sizeof(cpu_set), &cpu_set) == -1)
    std::cerr << "[WARN] sched_setaffinity failed: " << strerror(errno) << "\n";
}

int main(int argc, const char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <iface_addr> <sym>:<pos> [<sym>:<pos> ...]\n"
              << "  e.g. " << argv[0] << " 192.168.1.5 2:+3 4:-1 13:+2\n";
    return EXIT_FAILURE;
  }

  // Parse sym:pos pairs from command line.
  std::vector<std::pair<hft::msg::symbol_id_t, int64_t>> positions;
  for (int i = 2; i < argc; ++i) {
    const char *arg = argv[i];
    const char *sep = strchr(arg, ':');
    if (!sep) {
      std::cerr << "[ERROR] Bad argument '" << arg << "': expected sym:pos\n";
      return EXIT_FAILURE;
    }
    const auto sym = static_cast<hft::msg::symbol_id_t>(strtoul(arg, nullptr, 10));
    const int64_t pos = strtoll(sep + 1, nullptr, 10);
    if (pos != 0)
      positions.emplace_back(sym, pos);
  }

  if (positions.empty()) {
    std::cout << "[INFO] No non-zero positions given — nothing to do.\n";
    return EXIT_SUCCESS;
  }

  setup_affinity(CPU_ID);
  signal(SIGINT, interruption_handler);

  hft::client::ConnectionSpec connection = NDFEX_SPEC;
  connection.iface_ip = inet_addr(argv[1]);

  auto client = hft::client::ExchangeClient(connection, RISK_LIMITS);
  client.login(NDFEX_CLIENT_ID, NDFEX_USERNAME, NDFEX_PASSWORD);

  auto &tracker = client.get_position_tracker();

  // Inject synthetic positions.  Price 0 is used because we don't know the
  // original fill price; PNL accuracy is irrelevant here.
  std::vector<hft::msg::symbol_id_t> symbols;
  for (const auto &[sym, pos] : positions) {
    const auto qty = static_cast<hft::msg::quantity_t>(std::abs(pos));
    const hft::msg::SIDE side = (pos > 0) ? hft::msg::SIDE::BUY : hft::msg::SIDE::SELL;
    tracker.on_fill(sym, side, qty, 0);
    symbols.push_back(sym);
    std::cout << "[INFO] sym=" << sym << "  injected position " << pos << "\n";
  }

  hft::strategy::impl::Liquidate liquidate(
      client, client.get_orderbook(), tracker,
      symbols, RISK_LIMITS.max_qty_per_order);

  std::cout << "[INFO] Liquidating " << symbols.size() << " symbol(s)...\n";

  while (!interrupted) {
    client.update();
    liquidate.on_tick();

    if (liquidate.is_done()) {
      std::cout << "[INFO] All positions flat — exiting.\n";
      return EXIT_SUCCESS;
    }
  }

  std::cout << "[WARN] Interrupted before all positions were cleared.\n";
  return EXIT_FAILURE;
}
