/*
 * NDFEX TCP client definitions
 */

#ifndef HFT_CLIENT_H
#define HFT_CLIENT_H

#include <arpa/inet.h>
#include <unordered_map>
#include <vector>

// Forward declare to avoid circular include (Strategy.h includes ExchangeClient.h)
namespace hft::strategy { class Strategy; }

#include "hft/PositionTracker.h"
#include "hft/RiskSystem.h"
#include "hft/orderbook.h"
#include "msg/common.h"
#include "msg/market_data.h"
#include "msg/order_entry.h"
#include "multicast.h"

namespace hft::order {
class ExchangeOrder;
}

namespace hft::risk {
class RiskSystem;
}

namespace hft::client {

struct Address {
  in_addr_t ip;
  in_port_t port;
};

struct ConnectionSpec {
  in_addr_t iface_ip;
  Address live_mcast_host;
  Address replay_mcast_host;
  Address order_entry_host;
};

struct BidAndOffer {
  msg::price_t bid_price;
  msg::quantity_t bid_quantity;
  msg::price_t ask_price;
  msg::quantity_t ask_quantity;
};

class ExchangeClient {
private:
  int socket_fd;
  int epoll_fd_;

  msg::oe::client_id_t client_id = 0;
  msg::seq_num_t seq_num = 0;
  msg::oe::session_id_t session_id = 0;

  const mcast::MulticastListener live_listener_;
  const mcast::MulticastListener replay_listener_;

  orderbook::MultiSymbolOrderBook orderbook_;

  std::unordered_map<msg::order_id_t, order::ExchangeOrder *> orders{};
  risk::PositionTracker position_tracker_;
  risk::RiskSystem risk_system_;

  // Market data subscriptions: symbol → list of subscribed strategies
  std::unordered_map<msg::symbol_id_t, std::vector<strategy::Strategy *>> md_subscribers_;

  void handle_response(const msg::oe::Response &response);
  void fire_md_callbacks(msg::symbol_id_t symbol, const msg::md::MdMessage &msg);
  void fire_md_callbacks_all(const msg::md::MdMessage &msg);

public:
  ExchangeClient(const ConnectionSpec &spec, const risk::RiskLimits &limits);
  ~ExchangeClient();

  void update();

  msg::order_id_t register_order(msg::order_id_t order_id,
                                 order::ExchangeOrder &order);

  void submit_request(msg::oe::MSG_TYPE type,
                      const msg::oe::RequestBody &content);
  bool receive_response(msg::oe::Response *, bool blocking = false) const;

  void login(uint32_t _client_id, const char *username, const char *password);
  void logout();

  bool new_order(const msg::oe::NewOrderRequest &request);
  bool modify_order(const msg::oe::ModifyOrderRequest &request);
  void delete_order(const msg::oe::DeleteOrderRequest &request);

  void cancel_all_orders();
  void flatten_positions();

  [[nodiscard]] const std::unordered_map<msg::order_id_t,
                                         order::ExchangeOrder *> &
  get_orders() const;

  [[nodiscard]] orderbook::MultiSymbolOrderBook &get_orderbook();
  [[nodiscard]] risk::PositionTracker &get_position_tracker();

  BidAndOffer get_bbo(msg::symbol_id_t symbol);

  // Market data subscription API — called by Strategy::subscribe_md()
  void subscribe_md(msg::symbol_id_t symbol, strategy::Strategy *s);
  void unsubscribe_md(msg::symbol_id_t symbol, strategy::Strategy *s);
};

}; // namespace hft::client

#endif // HFT_CLIENT_H
