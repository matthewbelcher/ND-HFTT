#include "../include/hft/ExchangeClient.h"

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

#include "hft/ExchangeOrder.h"
#include "hft/PositionTracker.h"
#include "hft/Strategy.h"
#include "hft/RiskSystem.h"
#include "hft/msg/common.h"
#include "hft/msg/market_data.h"
#include "hft/msg/order_entry.h"
#include "hft/multicast.h"
#include "hft/orderbook.h"

constexpr const int MAX_EVENTS = 16;

namespace hft::client {
/**
 * Handle reception of response message
 *
 * @param response Response struct from exchange
 */
void ExchangeClient::handle_response(const msg::oe::Response &response) {

  // msg::oe::dump_oe_response(&response);

  // Update sequence number
  seq_num = std::max(seq_num, response.header.seq_num) + 1;

  switch (response.header.msg_type) {
  case msg::oe::MSG_TYPE::LOGIN_RESPONSE:
    // Shouldn't get a login response after initial connect
    std::cerr << "[ERROR] Received unexpected login response" << std::endl;
    break;
  case msg::oe::MSG_TYPE::ERROR:
    std::cerr << "[ERROR] Received error response from server: " << std::endl;
    // msg::oe::dump_oe_response(&response);
    break;
  default:
    // Response is order-related
    const auto order_id = response.body.order_closed_response.order_id;
    if (!orders.contains(order_id)) {
      std::cerr << "[ERROR] Received order response with unknown order_id: "
                << order_id << std::endl;
      break;
    }

    orders.at(order_id)->on_response(response);

    // Notify risk system of order lifecycle events
    auto &tracker = risk_system_.get_tracker();
    auto *order = orders.at(order_id);

    switch (response.header.msg_type) {
    case msg::oe::MSG_TYPE::ACK:
      risk_system_.on_order_acked();
      break;
    case msg::oe::MSG_TYPE::FILL: {
      const auto &fill = response.body.order_fill_response;
      tracker.on_fill(order->local_state.symbol, order->local_state.side,
                      fill.quantity, fill.price);
      tracker.on_order_fill(order_id, fill.quantity);
      break;
    }
    case msg::oe::MSG_TYPE::CLOSE:
      risk_system_.on_order_closed();
      tracker.on_order_closed(order_id);
      break;
    case msg::oe::MSG_TYPE::REJECT:
      risk_system_.on_order_closed();
      tracker.on_order_closed(order_id);
      break;
    default:
      break;
    }
    break;
  }
}

/**
 * Create a NDFEX exchange client
 *
 * @param spec Exchange connection information struct
 *
 * @throw std::runtime_error If connection fails
 */
ExchangeClient::ExchangeClient(const ConnectionSpec &spec,
                               const risk::RiskLimits &limits)
    : live_listener_(spec.live_mcast_host.ip, spec.live_mcast_host.port,
                     spec.iface_ip),
      replay_listener_(spec.replay_mcast_host.ip, spec.replay_mcast_host.port,
                       spec.iface_ip),
      orderbook_(13), position_tracker_(),
      risk_system_(limits, position_tracker_, orderbook_) {
  // Open socket to exchange order entry service
  const int fd = socket(AF_INET, SOCK_STREAM, 0);

  sockaddr_in exchange_addr{};
  exchange_addr.sin_family = AF_INET;
  exchange_addr.sin_addr.s_addr = spec.order_entry_host.ip;
  exchange_addr.sin_port = htons(spec.order_entry_host.port);

  if (connect(fd, (struct sockaddr *)&exchange_addr, sizeof(exchange_addr)) <
      0) {
    std::cerr << "Error establishing TCP connection " << strerror(errno)
              << std::endl;
    throw std::runtime_error("Could not connect to the exchange");
  }

  // Set socket to non-blocking mode
  const int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0) {
    std::cerr << "Error getting socket flags " << strerror(errno) << std::endl;
    throw std::runtime_error("Could not get socket flags");
  }
  if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
    std::cerr << "Error setting socket flags " << strerror(errno) << std::endl;
    throw std::runtime_error("Could not set socket flags");
  }

  socket_fd = fd;

  // Create epoll instance
  epoll_fd_ = epoll_create1(0);
  if (epoll_fd_ < 0) {
    std::cerr << "Failed to create epoll instance: " << strerror(errno)
              << std::endl;
    throw std::runtime_error("Failed to create epoll fd");
  }

  struct epoll_event ev;
  ev.events = EPOLLIN;
  // Add live mcast listener to epoll
  ev.data.fd = live_listener_.socket_fd;
  epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, live_listener_.socket_fd, &ev);
  // Add TCP order entry interface to epoll
  ev.data.fd = socket_fd;
  epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, socket_fd, &ev);

  // Synchronize orderbooks
  orderbook_.synchronize(live_listener_, replay_listener_);
}

/**
 * Disconnect from NDFEX exchange
 */
ExchangeClient::~ExchangeClient() {
  logout();

  // Gracefully close TCP connection
  shutdown(socket_fd, SHUT_WR);
  char buffer[1024];
  while (recv(socket_fd, buffer, sizeof(buffer), 0) > 0) {
  }
  close(socket_fd);
}

/**
 * Read incoming messages from exchange
 */
void ExchangeClient::update() {

  struct epoll_event events[MAX_EVENTS];
  int nfds = epoll_wait(epoll_fd_, events, MAX_EVENTS, 0);

  for (int n = 0; n < nfds; n++) {
    const auto ev = events[n];
    if (ev.data.fd == live_listener_.socket_fd) {
      // Market data update
      uint8_t buf[1024];
      msg::md::MdMessage m;
      try {
        const ssize_t revc_size = live_listener_.receive(buf, sizeof(buf));
        ssize_t parsed = 0;
        while (parsed < revc_size) {
          const auto n =
              msg::md::parse_message(buf + parsed, sizeof(buf) - parsed, &m);
          if (n <= 0)
            break;

          // msg::md::dump_message(&m);
          orderbook_.apply_message(m);
          risk_system_.on_md_update();

          // Route MD callbacks.  NEW_ORDER and TRADE_SUMMARY carry an explicit
          // symbol so we can fire only the relevant subscribers.  DELETE_ORDER,
          // MODIFY_ORDER, and TRADE do not — rather than maintaining an
          // order_id→symbol side-table (heap allocation per message), we fire
          // all subscribers.  Each callback is a cheap BBO check that is a
          // no-op when nothing on its symbol changed.
          switch (m.header.msg_type) {
          case msg::md::MSG_TYPE::NEW_ORDER:
            fire_md_callbacks(m.body.new_order.symbol, m);
            break;
          case msg::md::MSG_TYPE::TRADE_SUMMARY:
            fire_md_callbacks(m.body.trade_summary.symbol, m);
            break;
          case msg::md::MSG_TYPE::DELETE_ORDER:
          case msg::md::MSG_TYPE::MODIFY_ORDER:
          case msg::md::MSG_TYPE::TRADE:
            fire_md_callbacks_all(m);
            break;
          default:
            break;
          }

          parsed += n;
        }
      } catch (std::exception &e) {
        std::cerr << "[ERROR] Market data update failed: " << e.what()
                  << std::endl;
        cancel_all_orders();
        throw e;
      }

    } else if (ev.data.fd == socket_fd) {
      // Order entry response
      try {
        msg::oe::Response response;
        while (receive_response(&response, false)) {
          handle_response(response);
        }
      } catch (std::exception &e) {
        std::cerr << "[ERROR] Order entry response handling failed: "
                  << e.what() << std::endl;
        cancel_all_orders();
        throw e;
      }
    }
  }
}

/**
 * Register a new `ExchangeOrder` object with the client
 *
 * @param order_id Personal order ID to use (assigned an open one if 0)
 * @param order Reference to ExchangeOrder object
 * @throws std::runtime_error If order_id already exists in orders map
 */
msg::order_id_t ExchangeClient::register_order(msg::order_id_t order_id,
                                               order::ExchangeOrder &order) {
  // Assign an order_id if left blank
  if (order_id == 0) {
    order_id = orders.size() + 1;
  }

  if (orders.contains(order_id)) {
    throw std::runtime_error("Exchange order already registered");
  }

  orders.insert(std::make_pair(order_id, &order));
  return order_id;
}

/**
 * Send a request to the exchange
 *
 * @param type Order entry message type enum
 * @param content Request body struct
 * @throws std::runtime_error If request is not sent
 */
void ExchangeClient::submit_request(const msg::oe::MSG_TYPE type,
                                    const msg::oe::RequestBody &content) {

  const auto request_size =
      sizeof(msg::oe::RequestHeader) + msg::oe::get_request_size(type);
  const msg::oe::RequestHeader header = {
      .length = static_cast<uint16_t>(request_size),
      .msg_type = type,
      .version = msg::oe::OE_PROTOCOL_VERSION,
      .seq_num = (seq_num++),
      .client_id = client_id,
      .session_id = session_id,
  };

  msg::oe::Request request = {.header = header, .body = content};

  // DEBUG
  // std::cout << "[DEBUG] Submitting request..." << std::endl;
  // msg::oe::dump_oe_request(&request);

  const char *buf = reinterpret_cast<char *>(&request);
  size_t total_bytes_sent = 0;
  while (total_bytes_sent < request_size) {
    const ssize_t bytes_sent =
        send(socket_fd, buf + total_bytes_sent, request_size - total_bytes_sent,
             MSG_NOSIGNAL);

    if (bytes_sent == 0) {
      std::cerr << "Error writing to socket " << strerror(errno) << std::endl;
      throw std::runtime_error("Exchange disconnected");
    }
    if (bytes_sent < 0) {
      std::cerr << "Error writing to socket " << strerror(errno) << std::endl;
      throw std::runtime_error("Error while sending request");
    }

    total_bytes_sent += bytes_sent;
  }
}

/**
 * Receive a single response message from the exchange
 *
 * @param response Output struct to populate
 * @param blocking If true, spin until a response arrives; if false, return
 * immediately if no data
 * @return true if a response was received, false if no data was available
 * (non-blocking only)
 * @throw std::runtime_error If a socket error occurred
 */
bool ExchangeClient::receive_response(msg::oe::Response *response,
                                      const bool blocking) const {

  // Phase 1: read header
  const size_t header_size = sizeof(msg::oe::ResponseHeader);
  size_t header_offset = 0;

  while (header_offset < header_size) {
    const ssize_t bytes_received = recv(
        socket_fd, reinterpret_cast<char *>(&response->header) + header_offset,
        header_size - header_offset, MSG_NOSIGNAL);

    if (bytes_received < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        // If non-blocking and we haven't started reading yet, signal no data
        if (!blocking && header_offset == 0)
          return false;
        continue; // partial read or blocking — keep spinning
      }
      std::cerr << "Error reading from socket " << strerror(errno) << std::endl;
      throw std::runtime_error("Error reading response header from socket");
    }

    if (bytes_received == 0) {
      throw std::runtime_error("Exchange disconnected");
    }

    header_offset += bytes_received;
  }

  // Phase 2: read body (always spin — we must complete the message we started)
  const size_t body_size = response->header.length - header_size;
  size_t body_offset = 0;

  while (body_offset < body_size) {
    const ssize_t bytes_received =
        recv(socket_fd, reinterpret_cast<char *>(&response->body) + body_offset,
             body_size - body_offset, MSG_NOSIGNAL);

    if (bytes_received < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      std::cerr << "Error reading from socket " << strerror(errno) << std::endl;
      throw std::runtime_error("Error reading response body from socket");
    }

    if (bytes_received == 0) {
      throw std::runtime_error("Exchange disconnected");
    }

    body_offset += bytes_received;
  }

  return true;
}

/**
 * Authenticate client with NDFEX order entry server
 *
 * @param _client_id NDFEX client id
 * @param username NDFEX order entry username
 * @param password NDFEX order entry password
 */
void ExchangeClient::login(const uint32_t _client_id, const char *username,
                           const char *password) {

  if (session_id) {
    return;
  } // Already logged in

  client_id = _client_id;
  const msg::oe::LoginRequest request = {};
  strncpy((char *)request.username, username, sizeof(request.username));
  strncpy((char *)request.password, password, sizeof(request.password));

  submit_request(msg::oe::MSG_TYPE::LOGIN, {.login_request = request});

  // Handle login response
  msg::oe::Response response = {};
  do {
    receive_response(&response, true);
  } while (response.header.msg_type != msg::oe::MSG_TYPE::LOGIN_RESPONSE);

  const auto response_body = response.body.login_response;
  if (response_body.status != msg::oe::LOGIN_STATUS::SUCCESS) {
    std::cerr << "Login failed: "
              << msg::oe::login_status_string(response_body.status)
              << std::endl;
    throw std::runtime_error("Login failed");
  }

  session_id = response_body.session_id;
}

/**
 * Send logout message to exchange
 *
 * @todo Not sure how to do this yet
 */
void ExchangeClient::logout() {
  // TODO
}

/**
 * Send a 'new order' request to exchange, subject to risk checks
 *
 * @param request New order request body
 * @return true if order was sent, false if rejected by risk system
 */
bool ExchangeClient::new_order(const msg::oe::NewOrderRequest &request) {
  const auto result = risk_system_.evaluate_order(request);
  if (!result.accepted) {
    fprintf(stderr, "[RISK] New order REJECTED (order_id=%" PRIu64 "): %s\n",
            static_cast<uint64_t>(request.order_id),
            risk::reject_reason_string(result.reason));
    return false;
  }

  submit_request(msg::oe::MSG_TYPE::NEW_ORDER, {.new_order_request = request});

  risk_system_.on_order_sent();
  risk_system_.get_tracker().on_order_sent(request.order_id, request.symbol,
                                           request.side, request.quantity);

  return true;
}

/**
 * Send a 'modify order' request to exchange, subject to risk checks
 *
 * @param request Modify order request body
 * @return true if modification was sent, false if rejected by risk system
 */
bool ExchangeClient::modify_order(const msg::oe::ModifyOrderRequest &request) {
  // Look up current order state for delta calculation
  if (orders.contains(request.order_id)) {
    const auto *order = orders.at(request.order_id);
    const auto result = risk_system_.evaluate_modify(
        request, order->exchange_state.symbol, order->exchange_state.side,
        order->exchange_state.quantity);
    if (!result.accepted) {
      fprintf(stderr,
              "[RISK] Modify order REJECTED (order_id=%" PRIu64 "): %s\n",
              static_cast<uint64_t>(request.order_id),
              risk::reject_reason_string(result.reason));
      return false;
    }
  }

  submit_request(msg::oe::MSG_TYPE::MODIFY_ORDER,
                 {.modify_order_request = request});

  risk_system_.on_order_sent();
  // Update exposure tracking for the modified order
  risk_system_.get_tracker().on_order_modified(request.order_id, request.side,
                                               request.quantity);

  return true;
}

/**
 * Send a 'delete order' request to exchange
 *
 * @param request Delete order request body
 */
void ExchangeClient::delete_order(const msg::oe::DeleteOrderRequest &request) {
  submit_request(msg::oe::MSG_TYPE::DELETE_ORDER,
                 {.delete_order_request = request});
}

/**
 * Cancel all open orders on the exchange
 */
void ExchangeClient::cancel_all_orders() {
  for (auto &[id, order] : orders) {
    if (order->status != order::ORDER_STATUS::CLOSED &&
        order->status != order::ORDER_STATUS::CANCEL_IN_FLIGHT) {
      order->cancel();
    }
  }
}

/**
 * Get a read-only view of the orders map
 */
const std::unordered_map<msg::order_id_t, order::ExchangeOrder *> &
ExchangeClient::get_orders() const {
  return orders;
}

/**
 * Get the best bid and offer for a symbol
 *
 * @param symbol Symbol ID to look up
 **/
BidAndOffer ExchangeClient::get_bbo(msg::symbol_id_t symbol) {
  const auto best_bid = orderbook_.get_best_price_level(symbol, msg::SIDE::BUY);
  const auto best_ask =
      orderbook_.get_best_price_level(symbol, msg::SIDE::SELL);

  return {
      best_bid ? best_bid->price : 0,
      best_bid ? best_bid->quantity : 0,
      best_ask ? best_ask->price : 0,
      best_ask ? best_ask->quantity : 0,
  };
}

orderbook::MultiSymbolOrderBook &ExchangeClient::get_orderbook() {
  return orderbook_;
}

risk::PositionTracker &ExchangeClient::get_position_tracker() {
  return risk_system_.get_tracker();
}

void ExchangeClient::subscribe_md(const msg::symbol_id_t symbol,
                                  strategy::Strategy *s) {
  md_subscribers_[symbol].push_back(s);
}

void ExchangeClient::unsubscribe_md(const msg::symbol_id_t symbol,
                                    strategy::Strategy *s) {
  auto it = md_subscribers_.find(symbol);
  if (it == md_subscribers_.end()) return;
  auto &vec = it->second;
  vec.erase(std::remove(vec.begin(), vec.end(), s), vec.end());
}

void ExchangeClient::fire_md_callbacks(const msg::symbol_id_t symbol,
                                       const msg::md::MdMessage &msg) {
  auto it = md_subscribers_.find(symbol);
  if (it == md_subscribers_.end()) return;
  for (auto *s : it->second) {
    s->on_market_data(symbol, msg);
  }
}

void ExchangeClient::fire_md_callbacks_all(const msg::md::MdMessage &msg) {
  for (auto &[sym, subs] : md_subscribers_) {
    for (auto *s : subs) {
      s->on_market_data(sym, msg);
    }
  }
}

} // namespace hft::client
