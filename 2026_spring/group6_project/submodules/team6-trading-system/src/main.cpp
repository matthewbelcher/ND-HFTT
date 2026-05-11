#include <arpa/inet.h>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <hft/ExchangeClient.h>
#include <hft/ExchangeOrder.h>
#include <hft/PositionTracker.h>
#include <hft/RiskSystem.h>
#include <hft/Strategy.h>
#include <hft/multicast.h>
#include <hft/orderbook.h>

#include "hft/msg/common.h"
#include "strategies/ETFArbitrage.h"
#include "strategies/Liquidate.h"
#include "strategies/Logger.h"
#include "strategies/MarketMaker.h"
#include "strategies/MomentumScalper.h"
#include "strategies/SpreadScalper.h"

constexpr const uint32_t NDFEX_CLIENT_ID = (6);
constexpr const char *NDFEX_USERNAME = ("team6");
constexpr const char *NDFEX_PASSWORD = ("56vjGN5S");

const hft::client::ConnectionSpec NDFEX_SPEC{
    inet_addr("192.168.13.14"),
    {inet_addr("239.0.0.1"), 12345},
    {inet_addr("239.0.0.2"), 12345},
    {inet_addr("192.168.13.100"), 1234}};

const hft::client::ConnectionSpec GHOSTBOOK_SPEC{
    inet_addr("192.168.13.14"),
    {inet_addr("239.255.255.1"), 12345},
    {inet_addr("239.255.255.2"), 12345},
    {inet_addr("127.0.0.1"), 10000}};

const hft::risk::RiskLimits RISK_LIMITS{.max_qty_per_order = 10,
                                        .max_qty_per_side = 30,
                                        .max_exposure_per_side = 2000,
                                        .max_position = 10,
                                        .max_abs_position_shutdown = 10,
                                        .min_pnl_shutdown = -4500,
                                        .max_orders_per_second = 10,
                                        .max_orders_per_md_update = 10,
                                        .max_inflight_orders = 50,
                                        .min_valid_price = 2,
                                        .max_valid_price = 100000};

constexpr const int CPU_ID = 5;

// Interrupted flag
std::atomic<bool> interrupted(false);

/* Helper functions */

bool setup_hot_thread(int cpu_id) {
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  CPU_SET(cpu_id, &cpu_set);
  if (sched_setaffinity(0, sizeof(cpu_set), &cpu_set) == -1) {
    std::cerr << "sched_setaffinity failed: " << strerror(errno) << std::endl;
    return false;
  }

  return true;
}

void interruptionHandler(int signal) {
  if (signal == SIGINT) {
    std::cout << "[INTERRUPT] Beginning cleanup..." << std::endl;
    interrupted = true;
  }
}

/* Main Entry Point */

int main(const int argc, const char *argv[]) {
  // Process CLI arguments
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <local inferface addr>" << std::endl;
    return EXIT_FAILURE;
  }
  const in_addr_t iface_ip = inet_addr(argv[1]);
  hft::client::ConnectionSpec connection = GHOSTBOOK_SPEC;
  connection.iface_ip = iface_ip;

  // Set CPU affinity
  if (!setup_hot_thread(CPU_ID)) {
    return EXIT_FAILURE;
  }

  // Register interrupt handler
  signal(SIGINT, interruptionHandler);

  // Disable std::cout
  // std::streambuf *orig_cout = std::cout.rdbuf(nullptr);
  // std::streambuf *orig_cerr = std::cerr.rdbuf(nullptr);

  // Create exchange order-entry client
  auto client = hft::client::ExchangeClient(connection, RISK_LIMITS);
  client.login(NDFEX_CLIENT_ID, NDFEX_USERNAME, NDFEX_PASSWORD);

  auto &position_tracker = client.get_position_tracker();

  // STRATEGY SELECTION

  // hft::strategy::impl::MarketMaker market_maker(
  //     client, /*symbol=*/1, /*qty=*/4, /*half_spread=*/5,
  //     /*max_skew=*/3, /*stop_loss=*/-500.0);

  // hft::strategy::impl::MomentumScalper momentum(
  //     client, /*symbol=*/4, /*qty=*/2, /*window=*/100,
  //     /*threshold=*/0.003, /*take_profit=*/10, /*stop_loss=*/10,
  //     /*cooldown=*/200);

  // hft::strategy::impl::ETFArbitrage etf_arb(client, /*threshold=*/15,
  // /*qty=*/1,
  //                                           /*revert_threshold=*/5);

  // hft::strategy::impl::SpreadScalper spread_scalper(
  //     client, /*symbol=*/9, /*qty=*/2, /*min_spread=*/10,
  //     /*target_profit=*/10, /*stop_loss_per_trade=*/15);

  std::vector<std::unique_ptr<hft::strategy::Strategy>> strategies{};

  strategies.push_back(std::make_unique<hft::strategy::impl::MarketMaker>(
      client, 1, 4, 5, 3, -1000));
  strategies.push_back(std::make_unique<hft::strategy::impl::MarketMaker>(
      client, 2, 4, 5, 3, -1000));
  strategies.push_back(std::make_unique<hft::strategy::impl::MarketMaker>(
      client, 3, 4, 5, 3, -1000));
  strategies.push_back(std::make_unique<hft::strategy::impl::MarketMaker>(
      client, 4, 4, 5, 3, -1000));
  strategies.push_back(std::make_unique<hft::strategy::impl::MarketMaker>(
      client, 5, 4, 5, 3, -1000));
  strategies.push_back(std::make_unique<hft::strategy::impl::MarketMaker>(
      client, 6, 4, 5, 3, -1000));
  // strategies.push_back(std::make_unique<hft::strategy::impl::MarketMaker>(
  //     client, 7, 4, 5, 3, -1000));
  // strategies.push_back(std::make_unique<hft::strategy::impl::MarketMaker>(
  //     client, 8, 4, 5, 3, -1000));
  // strategies.push_back(std::make_unique<hft::strategy::impl::MarketMaker>(
  //     client, 9, 4, 5, 3, -1000));
  // strategies.push_back(std::make_unique<hft::strategy::impl::MarketMaker>(
  //     client, 10, 4, 5, 3, -1000));
  // strategies.push_back(std::make_unique<hft::strategy::impl::MarketMaker>(
  //     client, 11, 4, 5, 3, -1000));
  // strategies.push_back(std::make_unique<hft::strategy::impl::MarketMaker>(
  //     client, 12, 4, 5, 3, -1000));
  // strategies.push_back(std::make_unique<hft::strategy::impl::MarketMaker>(
  //     client, 13, 4, 5, 3, -1000));

  // strategies.push_back(std::make_unique<hft::strategy::impl::MomentumScalper>(
  //     client, 8, 2, 100, 0.003, 10, 10, 200));
  // strategies.push_back(std::make_unique<hft::strategy::impl::MomentumScalper>(
  //     client, 9, 2, 100, 0.003, 10, 10, 200));
  // strategies.push_back(std::make_unique<hft::strategy::impl::MomentumScalper>(
  //     client, 10, 2, 100, 0.003, 10, 10, 200));

  // Log configuration
  static const std::vector<hft::msg::symbol_id_t> LOG_SYMS = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  uint64_t tick_count = 0;

  // Main event loop. Strategy logic fires inside client.update() via MD
  // callbacks.
  while (!interrupted) {
    client.update();

    if (++tick_count % 10000 == 0) {
      hft::strategy::log::log_pnl(position_tracker, LOG_SYMS, client);
    }
  }

  // Restore std::cout
  // std::cout.rdbuf(orig_cout);
  // std::cerr.rdbuf(orig_cerr);

  // Liquidate all positions across every listed symbol and exit when flat.
  client.cancel_all_orders();
  hft::strategy::impl::Liquidate liquidate(
      client, client.get_orderbook(), position_tracker,
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
      RISK_LIMITS.max_qty_per_order);

  // position_tracker.on_fill(1, hft::msg::SIDE::BUY, 4, 0);

  while (true) {
    client.update();
    liquidate.on_tick();

    if (liquidate.is_done()) {
      std::cout << "[INFO] All positions flat... exiting" << std::endl;
      break;
    }
  }

  return EXIT_SUCCESS;
}
