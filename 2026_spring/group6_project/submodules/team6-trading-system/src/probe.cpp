/*
 * Simple program to send a test order to the exchange
 **/

#include "hft/msg/common.h"
#include "hft/msg/market_data.h"
#include "hft/msg/order_entry.h"
#include <arpa/inet.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>
#include <unistd.h>
#include <vector>

#include <hft/ExchangeClient.h>
#include <hft/ExchangeOrder.h>
#include <hft/PositionTracker.h>
#include <hft/RiskSystem.h>
#include <hft/Strategy.h>
#include <hft/multicast.h>
#include <hft/orderbook.h>

using namespace std;
using namespace hft;

/* Constants */

// constexpr const char *EXCH_DATA_LIVE_IP = "239.255.255.1";
// constexpr const int EXCH_DATA_LIVE_PORT = 12345;
// constexpr const char *EXCH_DATA_REPLAY_IP = "239.255.255.1";
// constexpr const int EXCH_DATA_REPLAY_PORT = 12345;

constexpr const char *EXCH_ORDER_ENTRY_IP = "127.0.0.1";
constexpr const int EXCH_ORDER_ENTRY_PORT = 10000;

/* Main entry point */

int main(const int argc, const char *argv[]) {

  // Process CLI arguments
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <local IP address>" << std::endl;
    return EXIT_FAILURE;
  }
  // const in_addr_t local_udp_ip = inet_addr(argv[1]);

  auto client = client::ExchangeClient(inet_addr(EXCH_ORDER_ENTRY_IP),
                                       EXCH_ORDER_ENTRY_PORT);
  client.login(12345, "jackoconnor", "jackoconnor");

  // Create UDP multicast listener objects
  // const auto live_listener = hft::mcast::MulticastListener(
  //     inet_addr(EXCH_DATA_LIVE_IP), EXCH_DATA_LIVE_PORT, local_udp_ip);

  // const auto replay_listener = hft::mcast::MulticastListener(
  //     inet_addr(EXCH_DATA_REPLAY_IP), EXCH_DATA_REPLAY_PORT, local_udp_ip);

  const int num_orders = 100;
  vector<unique_ptr<order::ExchangeOrder>> orders{};

  while (true) {
    client.update();

    const int count = orders.size();
    if (count < num_orders) {
      auto order = make_unique<order::ExchangeOrder>(
          client, count + 10, 1, msg::SIDE::BUY, 10, 1,
          msg::oe::ORDER_FLAGS::NONE);
      orders.push_back(std::move(order));
      orders.back()->commit();
    } else {
      break;
    }

    this_thread::sleep_for(chrono::milliseconds(100));
  }

  return EXIT_SUCCESS;
}
