#include "ndfex_gateway.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <deque>
#include <map>
#include <memory>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/asio.hpp>

#include "../matching_engine/matching_main.h"
#include "../replay_engine/replay_engine.h"
#include "ghostbook/ndfex/protocol.h"

namespace ghostbook::gateway {
namespace {

namespace asio = boost::asio;
using boost::system::error_code;
using tcp = asio::ip::tcp;
using udp = asio::ip::udp;

// --- Protocol conversion helpers ---

matching::Side ndfex_side_to_matching(ndfex::SIDE side) {
  return side == ndfex::SIDE::BUY ? matching::Side::Buy : matching::Side::Sell;
}

ndfex::SIDE matching_side_to_ndfex(matching::Side side) {
  return side == matching::Side::Buy ? ndfex::SIDE::BUY : ndfex::SIDE::SELL;
}

matching::TimeInForce ndfex_flags_to_tif(ndfex::oe::ORDER_FLAGS flags) {
  return flags == ndfex::oe::ORDER_FLAGS::IOC ? matching::TimeInForce::IOC
                                              : matching::TimeInForce::Day;
}

ndfex::oe::REJECT_REASON matching_reason_to_ndfex(const std::string &reason) {
  if (reason.find("duplicate") != std::string::npos) {
    return ndfex::oe::REJECT_REASON::DUPLICATE_ORDER_ID;
  }
  if (reason.find("price") != std::string::npos) {
    return ndfex::oe::REJECT_REASON::INVALID_PRICE;
  }
  if (reason.find("quantity") != std::string::npos ||
      reason.find("qty") != std::string::npos) {
    return ndfex::oe::REJECT_REASON::INVALID_QUANTITY;
  }
  if (reason.find("unknown") != std::string::npos ||
      reason.find("not found") != std::string::npos) {
    return ndfex::oe::REJECT_REASON::UNKNOWN_ORDER_ID;
  }
  return ndfex::oe::REJECT_REASON::INVALID_ORDER;
}

std::uint64_t now_nanoseconds() {
  struct timespec ts{};
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
         static_cast<std::uint64_t>(ts.tv_nsec);
}

// --- Wire framing helpers ---

std::size_t oe_response_body_size(ndfex::oe::MSG_TYPE type) {
  switch (type) {
  case ndfex::oe::MSG_TYPE::LOGIN_RESPONSE:
    return sizeof(ndfex::oe::LoginResponse);
  case ndfex::oe::MSG_TYPE::ACK:
    return sizeof(ndfex::oe::OrderAckResponse);
  case ndfex::oe::MSG_TYPE::REJECT:
    return sizeof(ndfex::oe::OrderRejectResponse);
  case ndfex::oe::MSG_TYPE::FILL:
    return sizeof(ndfex::oe::OrderFillResponse);
  case ndfex::oe::MSG_TYPE::CLOSE:
    return sizeof(ndfex::oe::OrderClosedResponse);
  case ndfex::oe::MSG_TYPE::ERROR:
    return sizeof(ndfex::oe::ErrorResponse);
  default:
    return 0;
  }
}

ndfex::oe::ResponseHeader make_oe_response_header(
    ndfex::oe::MSG_TYPE msg_type, ndfex::oe::client_id_t client_id,
    ndfex::seq_num_t req_seq_num, ndfex::seq_num_t last_seq_num) {
  const auto body_sz =
      static_cast<std::uint16_t>(oe_response_body_size(msg_type));
  ndfex::oe::ResponseHeader hdr{};
  hdr.length =
      static_cast<std::uint16_t>(sizeof(ndfex::oe::ResponseHeader) + body_sz);
  hdr.msg_type = msg_type;
  hdr.version = ndfex::oe::OE_PROTOCOL_VERSION;
  hdr.seq_num = req_seq_num;
  hdr.last_seq_num = last_seq_num;
  hdr.client_id = client_id;
  return hdr;
}

std::vector<std::uint8_t>
build_oe_bytes(ndfex::oe::MSG_TYPE msg_type, ndfex::oe::client_id_t client_id,
               ndfex::seq_num_t req_seq_num, ndfex::seq_num_t last_seq_num,
               const void *body, std::size_t body_size) {
  const auto hdr =
      make_oe_response_header(msg_type, client_id, req_seq_num, last_seq_num);
  std::vector<std::uint8_t> frame(sizeof(hdr) + body_size);
  std::memcpy(frame.data(), &hdr, sizeof(hdr));
  if (body_size > 0 && body != nullptr) {
    std::memcpy(frame.data() + sizeof(hdr), body, body_size);
  }
  return frame;
}

ndfex::md::Header make_md_header(ndfex::md::MSG_TYPE msg_type,
                                 ndfex::seq_num_t seq, std::size_t body_size,
                                 bool snapshot = false) {
  ndfex::md::Header hdr{};
  hdr.magic_number =
      snapshot ? ndfex::md::SNAPSHOT_MAGIC_NUMBER : ndfex::md::MAGIC_NUMBER;
  hdr.length =
      static_cast<std::uint16_t>(sizeof(ndfex::md::Header) + body_size);
  hdr.seq_num = seq;
  hdr.timestamp = now_nanoseconds();
  hdr.msg_type = msg_type;
  return hdr;
}

// --- Lock-free SPSC queue (identical to tcp_gateway) ---

template <typename T, std::size_t Capacity> class SpscQueue {
public:
  static_assert(Capacity > 1, "Capacity must be > 1");

  bool push(const T &value) {
    const auto head = head_.load(std::memory_order_relaxed);
    const auto next = (head + 1) % Capacity;
    if (next == tail_.load(std::memory_order_acquire)) {
      return false;
    }
    data_[head] = value;
    head_.store(next, std::memory_order_release);
    return true;
  }

  bool pop(T *out) {
    const auto tail = tail_.load(std::memory_order_relaxed);
    if (tail == head_.load(std::memory_order_acquire)) {
      return false;
    }
    *out = data_[tail];
    tail_.store((tail + 1) % Capacity, std::memory_order_release);
    return true;
  }

private:
  std::array<T, Capacity> data_{};
  std::atomic<std::size_t> head_{0};
  std::atomic<std::size_t> tail_{0};
};

} // namespace

// =============================================================================
// NdfexGateway::ImplBase — abstract base for pimpl
// =============================================================================

class NdfexGateway::ImplBase {
public:
  virtual ~ImplBase() = default;
  virtual bool start(std::uint16_t oe_port, const std::string &md_host,
                     std::uint16_t md_port, const std::string &snap_host,
                     std::uint16_t snap_port,
                     const std::string &udp_interface) = 0;
  virtual void shutdown() = 0;
  virtual void process_inline(session_id_t session_id,
                              const ndfex::oe::Request &req) = 0;
  virtual session_id_t create_test_session(ndfex::oe::client_id_t) = 0;
  virtual void advance_feed() {} // no-op for MatchingEngine

  NdfexMarketDataCallback md_callback_;
  NdfexResponseCallback ndfex_response_callback_;
};

// =============================================================================
// NdfexGateway::Impl<EngineT>
// =============================================================================

template <typename EngineT>
class NdfexGateway::Impl : public NdfexGateway::ImplBase {
public:
  // Constructor for MatchingEngine (pass start_clock).
  explicit Impl(NdfexGateway *owner, logical_clock_t start_clock)
      : owner_(owner), io_ctx_(), work_guard_(asio::make_work_guard(io_ctx_)),
        oe_acceptor_(io_ctx_), md_socket_(io_ctx_), snap_socket_(io_ctx_),
        outbound_timer_(io_ctx_), snap_timer_(io_ctx_), running_(false),
        exchange_id_counter_(1000), engine_(start_clock) {}

  // Constructor for ReplayEngine (move-in a pre-configured engine).
  explicit Impl(NdfexGateway *owner, EngineT engine)
      : owner_(owner), io_ctx_(), work_guard_(asio::make_work_guard(io_ctx_)),
        oe_acceptor_(io_ctx_), md_socket_(io_ctx_), snap_socket_(io_ctx_),
        outbound_timer_(io_ctx_), snap_timer_(io_ctx_), running_(false),
        exchange_id_counter_(1000), engine_(std::move(engine)) {
    // Wire up feed messages as market data to live clients.
    if constexpr (std::is_same_v<EngineT, replay::ReplayEngine>) {
      engine_.set_feed_message_callback(
          [this](const ndfex::md::Message &msg) { on_feed_message(msg); });
    }
  }

  ~Impl() { shutdown(); }

  bool start(std::uint16_t oe_port, const std::string &md_host,
             std::uint16_t md_port, const std::string &snap_host,
             std::uint16_t snap_port, const std::string &udp_interface) {
    if (running_.exchange(true, std::memory_order_acq_rel)) {
      return false;
    }

    error_code ec;
    setup_md_socket(md_host, md_port, udp_interface, &ec);
    if (ec) {
      running_.store(false, std::memory_order_release);
      return false;
    }

    setup_snap_socket(snap_host, snap_port, udp_interface, &ec);
    if (ec) {
      running_.store(false, std::memory_order_release);
      return false;
    }

    setup_acceptor(oe_acceptor_, oe_port, &ec);
    if (ec) {
      running_.store(false, std::memory_order_release);
      return false;
    }

    start_accept_orders();
    schedule_outbound_pump();
    schedule_snapshot_timer();

    io_thread_ = std::thread([this]() { io_ctx_.run(); });
    engine_thread_ = std::thread([this]() { engine_loop(); });
    return true;
  }

  void shutdown() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) {
      return;
    }

    io_ctx_.post([this]() {
      error_code ec;
      outbound_timer_.cancel(ec);
      snap_timer_.cancel(ec);
      oe_acceptor_.close(ec);
      md_socket_.close(ec);
      snap_socket_.close(ec);
      for (auto &[_, s] : sessions_) {
        s->socket.close(ec);
      }
      sessions_.clear();
      sessions_by_id_.clear();
    });

    work_guard_.reset();
    io_ctx_.stop();
    if (io_thread_.joinable())
      io_thread_.join();
    if (engine_thread_.joinable())
      engine_thread_.join();
  }

  // Inline path: process an NDFEX OE request synchronously (used by tests and
  // the async TCP receiver).
  void process_inline(session_id_t session_id, const ndfex::oe::Request &req) {
    const auto msg_type = req.header.msg_type;
    const auto req_seq = req.header.seq_num;
    const auto client_id = inline_client_id(session_id, req.header.client_id);

    switch (msg_type) {
    case ndfex::oe::MSG_TYPE::LOGIN:
      handle_login_inline(session_id, req, client_id, req_seq);
      break;
    case ndfex::oe::MSG_TYPE::NEW_ORDER:
      handle_new_order_inline(session_id, req.body.new_order_request, client_id,
                              req_seq);
      break;
    case ndfex::oe::MSG_TYPE::DELETE_ORDER:
      handle_delete_order_inline(session_id, req.body.delete_order_request,
                                 client_id, req_seq);
      break;
    case ndfex::oe::MSG_TYPE::MODIFY_ORDER:
      handle_modify_order_inline(session_id, req.body.modify_order_request,
                                 client_id, req_seq);
      break;
    default:
      owner_->stats_.protocol_errors++;
      break;
    }

    owner_->logical_clock_ = engine_.logical_clock();
  }

  session_id_t create_test_session(ndfex::oe::client_id_t client_id) {
    const session_id_t sid = owner_->session_mgr_.create_session(0, 0, 0, 0);
    inline_session_client_ids_[sid] = client_id;
    inline_session_seq_nums_[sid] = 0;
    return sid;
  }

private:
  // ----- per-TCP-socket state -----
  struct Session : public std::enable_shared_from_this<Session> {
    explicit Session(asio::io_context &io)
        : socket(io), recv_buf(), write_queue() {}
    tcp::socket socket;
    std::vector<std::uint8_t> recv_buf;
    bool logged_in{false};
    session_id_t session_id{0};
    ndfex::oe::client_id_t client_id{0};
    ndfex::seq_num_t last_req_seq{0};
    std::array<std::uint8_t, 2048> read_chunk{};
    std::deque<std::vector<std::uint8_t>> write_queue;
    bool write_in_progress{false};
  };

  // ----- inter-thread message types -----
  enum class CmdType : std::uint8_t { NewOrder, DeleteOrder, ModifyOrder };

  struct EngineCommand {
    CmdType type{CmdType::NewOrder};
    session_id_t session_id{};
    ndfex::oe::client_id_t client_id{};
    ndfex::seq_num_t req_seq{};

    struct NewOrderData {
      ndfex::order_id_t order_id{};
      ndfex::symbol_id_t symbol{};
      ndfex::SIDE side{};
      ndfex::quantity_t quantity{};
      ndfex::price_t price{};
      ndfex::oe::ORDER_FLAGS flags{};
    } new_order{};

    struct DeleteOrderData {
      ndfex::order_id_t order_id{};
    } delete_order{};

    struct ModifyOrderData {
      ndfex::order_id_t original_id{};
      matching::order_id_t synthetic_id{};
      ndfex::SIDE side{};
      ndfex::quantity_t quantity{};
      ndfex::price_t price{};
    } modify_order{};
  };

  struct OeReport {
    session_id_t session_id{};
    ndfex::oe::client_id_t client_id{};
    ndfex::seq_num_t req_seq{};
    ndfex::seq_num_t last_seq{};
    ndfex::oe::Response response{};
    std::uint16_t body_size{};
  };

  struct MdReport {
    ndfex::md::Message message{};
  };

  // ----- market data order tracking -----
  struct MdOrderState {
    ndfex::order_id_t order_id{};
    ndfex::symbol_id_t symbol{};
    ndfex::SIDE side{};
    ndfex::quantity_t quantity{};
    ndfex::price_t price{};
  };

  static constexpr std::size_t kQueueCapacity = 1 << 15;

  NdfexGateway *owner_;

  asio::io_context io_ctx_;
  asio::executor_work_guard<asio::io_context::executor_type> work_guard_;
  tcp::acceptor oe_acceptor_;
  udp::socket md_socket_;
  udp::endpoint md_endpoint_;
  bool md_is_multicast_{false};
  udp::socket snap_socket_;
  udp::endpoint snap_endpoint_;
  bool snap_is_multicast_{false};
  asio::steady_timer outbound_timer_;
  asio::steady_timer snap_timer_;
  std::atomic<bool> snapshot_due_{false};

  std::unordered_map<std::uintptr_t, std::shared_ptr<Session>> sessions_;
  std::unordered_map<session_id_t, std::weak_ptr<Session>> sessions_by_id_;

  std::atomic<bool> running_;
  std::thread io_thread_;
  std::thread engine_thread_;

  SpscQueue<EngineCommand, kQueueCapacity> command_queue_;
  SpscQueue<OeReport, kQueueCapacity> oe_report_queue_;
  SpscQueue<MdReport, kQueueCapacity> md_report_queue_;

  // Engine-thread-only order state
  EngineT engine_;
  std::unordered_map<ndfex::order_id_t, session_id_t> order_owner_session_;
  std::unordered_map<ndfex::order_id_t, ndfex::oe::client_id_t>
      order_owner_client_id_;
  std::unordered_map<ndfex::order_id_t, ndfex::seq_num_t> order_req_seq_;
  std::unordered_map<matching::order_id_t, ndfex::order_id_t>
      synthetic_to_ndfex_id_;
  std::unordered_set<ndfex::order_id_t> pending_modify_original_ids_;
  matching::order_id_t exchange_id_counter_;

  // Market data state (engine-thread-only in async mode, inline otherwise)
  std::unordered_map<ndfex::order_id_t, MdOrderState> md_orders_;
  std::atomic<ndfex::seq_num_t> md_seq_{1};

  // Inline-mode session metadata (not used in async mode)
  std::unordered_map<session_id_t, ndfex::oe::client_id_t>
      inline_session_client_ids_;
  std::unordered_map<session_id_t, ndfex::seq_num_t> inline_session_seq_nums_;

  // ===========================================================
  // Networking
  // ===========================================================

  void setup_acceptor(tcp::acceptor &acc, std::uint16_t port, error_code *ec) {
    acc.open(tcp::v4(), *ec);
    if (*ec)
      return;
    acc.set_option(asio::socket_base::reuse_address(true), *ec);
    if (*ec)
      return;
    acc.bind(tcp::endpoint(tcp::v4(), port), *ec);
    if (*ec)
      return;
    acc.listen(asio::socket_base::max_listen_connections, *ec);
  }

  void setup_md_socket(const std::string &host, std::uint16_t port,
                       const std::string &interface, error_code *ec) {
    md_socket_.open(udp::v4(), *ec);
    if (*ec)
      return;
    md_endpoint_ = udp::endpoint(asio::ip::make_address(host, *ec), port);
    if (*ec)
      return;
    md_is_multicast_ = md_endpoint_.address().is_multicast();
    if (md_is_multicast_) {
      md_socket_.set_option(asio::ip::multicast::hops(1), *ec);

      const auto local_interface = boost::asio::ip::make_address_v4(interface);
      md_socket_.set_option(
          boost::asio::ip::multicast::outbound_interface(local_interface));
    }
  }

  void setup_snap_socket(const std::string &host, std::uint16_t port,
                         const std::string &interface, error_code *ec) {
    snap_socket_.open(udp::v4(), *ec);
    if (*ec)
      return;
    snap_endpoint_ = udp::endpoint(asio::ip::make_address(host, *ec), port);
    if (*ec)
      return;
    snap_is_multicast_ = snap_endpoint_.address().is_multicast();
    if (snap_is_multicast_) {
      snap_socket_.set_option(asio::ip::multicast::hops(1), *ec);

      const auto local_interface = boost::asio::ip::make_address_v4(interface);
      snap_socket_.set_option(
          boost::asio::ip::multicast::outbound_interface(local_interface));
    }
  }

  void start_accept_orders() {
    auto session = std::make_shared<Session>(io_ctx_);
    oe_acceptor_.async_accept(session->socket, [this,
                                                session](const error_code &ec) {
      if (!ec && running_.load(std::memory_order_acquire)) {
        sessions_[reinterpret_cast<std::uintptr_t>(session.get())] = session;
        start_read(session);
      }
      if (running_.load(std::memory_order_acquire)) {
        start_accept_orders();
      }
    });
  }

  void schedule_snapshot_timer() {
    snap_timer_.expires_after(std::chrono::milliseconds(100));
    snap_timer_.async_wait([this](const error_code &ec) {
      if (ec || !running_.load(std::memory_order_acquire)) {
        return;
      }
      snapshot_due_.store(true, std::memory_order_release);
      schedule_snapshot_timer();
    });
  }

  void start_read(const std::shared_ptr<Session> &session) {
    session->socket.async_read_some(
        asio::buffer(session->read_chunk),
        [this, session](const error_code &ec, std::size_t bytes) {
          if (ec || bytes == 0) {
            close_session(session);
            return;
          }
          session->recv_buf.insert(session->recv_buf.end(),
                                   session->read_chunk.data(),
                                   session->read_chunk.data() + bytes);
          parse_and_dispatch(session);
          if (running_.load(std::memory_order_acquire)) {
            start_read(session);
          }
        });
  }

  void parse_and_dispatch(const std::shared_ptr<Session> &session) {
    auto &buf = session->recv_buf;
    while (buf.size() >= ndfex::oe::request_header_size) {
      std::uint16_t msg_len{};
      std::memcpy(&msg_len, buf.data(), sizeof(msg_len));
      if (msg_len < ndfex::oe::request_header_size || buf.size() < msg_len) {
        break;
      }

      ndfex::oe::Request req{};
      const std::size_t copy_sz = std::min<std::size_t>(msg_len, sizeof(req));
      std::memcpy(&req, buf.data(), copy_sz);
      buf.erase(buf.begin(), buf.begin() + msg_len);

      owner_->stats_.total_messages_received++;
      handle_oe_message(session, req);
    }
  }

  void handle_oe_message(const std::shared_ptr<Session> &session,
                         const ndfex::oe::Request &req) {
    const auto type = req.header.msg_type;

    if (type == ndfex::oe::MSG_TYPE::LOGIN) {
      handle_logon_async(session, req);
      return;
    }

    if (!session->logged_in) {
      // Send an error response for unauthenticated messages
      ndfex::oe::ErrorResponse err{};
      err.error_code = 1;
      const char *msg = "not logged in";
      err.error_message_length = static_cast<std::uint16_t>(std::strlen(msg));
      std::memcpy(err.error_message, msg, err.error_message_length);
      auto bytes = build_oe_bytes(ndfex::oe::MSG_TYPE::ERROR,
                                  req.header.client_id, req.header.seq_num,
                                  req.header.seq_num, &err, sizeof(err));
      queue_write(session, std::move(bytes));
      return;
    }

    session->last_req_seq = req.header.seq_num;

    EngineCommand cmd{};
    cmd.session_id = session->session_id;
    cmd.client_id = session->client_id;
    cmd.req_seq = req.header.seq_num;

    switch (type) {
    case ndfex::oe::MSG_TYPE::NEW_ORDER:
      if (req.header.length <
          ndfex::oe::request_header_size + sizeof(ndfex::oe::NewOrderRequest)) {
        owner_->stats_.protocol_errors++;
        return;
      }
      cmd.type = CmdType::NewOrder;
      cmd.new_order.order_id = req.body.new_order_request.order_id;
      cmd.new_order.symbol = req.body.new_order_request.symbol;
      cmd.new_order.side = req.body.new_order_request.side;
      cmd.new_order.quantity = req.body.new_order_request.quantity;
      cmd.new_order.price = req.body.new_order_request.price;
      cmd.new_order.flags = req.body.new_order_request.flags;
      owner_->stats_.total_orders_submitted++;
      enqueue_command(cmd);
      break;
    case ndfex::oe::MSG_TYPE::DELETE_ORDER:
      if (req.header.length < ndfex::oe::request_header_size +
                                  sizeof(ndfex::oe::DeleteOrderRequest)) {
        owner_->stats_.protocol_errors++;
        return;
      }
      cmd.type = CmdType::DeleteOrder;
      cmd.delete_order.order_id = req.body.delete_order_request.order_id;
      owner_->stats_.total_cancels_submitted++;
      enqueue_command(cmd);
      break;
    case ndfex::oe::MSG_TYPE::MODIFY_ORDER: {
      if (req.header.length < ndfex::oe::request_header_size +
                                  sizeof(ndfex::oe::ModifyOrderRequest)) {
        owner_->stats_.protocol_errors++;
        return;
      }
      const auto synthetic = exchange_id_counter_++;
      cmd.type = CmdType::ModifyOrder;
      cmd.modify_order.original_id = req.body.modify_order_request.order_id;
      cmd.modify_order.synthetic_id = synthetic;
      cmd.modify_order.side = req.body.modify_order_request.side;
      cmd.modify_order.quantity = req.body.modify_order_request.quantity;
      cmd.modify_order.price = req.body.modify_order_request.price;
      owner_->stats_.total_modifies_submitted++;
      enqueue_command(cmd);
      break;
    }
    default:
      owner_->stats_.protocol_errors++;
      break;
    }
  }

  void handle_logon_async(const std::shared_ptr<Session> &session,
                          const ndfex::oe::Request &req) {
    if (session->logged_in) {
      return;
    }

    const session_id_t sid = owner_->session_mgr_.create_session(0, 0, 0, 0);
    session->logged_in = true;
    session->session_id = sid;
    session->client_id = req.header.client_id;
    sessions_by_id_[sid] = session;

    ndfex::oe::LoginResponse resp{};
    resp.session_id = static_cast<ndfex::oe::session_id_t>(sid);
    resp.status = ndfex::oe::LOGIN_STATUS::SUCCESS;

    auto bytes = build_oe_bytes(ndfex::oe::MSG_TYPE::LOGIN_RESPONSE,
                                req.header.client_id, req.header.seq_num,
                                req.header.seq_num, &resp, sizeof(resp));
    queue_write(session, std::move(bytes));
  }

  void queue_write(const std::shared_ptr<Session> &session,
                   std::vector<std::uint8_t> &&frame) {
    session->write_queue.push_back(std::move(frame));
    if (!session->write_in_progress) {
      do_write(session);
    }
  }

  void do_write(const std::shared_ptr<Session> &session) {
    if (session->write_queue.empty()) {
      session->write_in_progress = false;
      return;
    }
    session->write_in_progress = true;
    asio::async_write(session->socket,
                      asio::buffer(session->write_queue.front()),
                      [this, session](const error_code &ec, std::size_t) {
                        if (ec) {
                          close_session(session);
                          return;
                        }
                        session->write_queue.pop_front();
                        do_write(session);
                      });
  }

  void close_session(const std::shared_ptr<Session> &session) {
    error_code ec;
    session->socket.close(ec);
    if (session->session_id != 0) {
      sessions_by_id_.erase(session->session_id);
      owner_->session_mgr_.close_session(session->session_id);
    }
    sessions_.erase(reinterpret_cast<std::uintptr_t>(session.get()));
  }

  void enqueue_command(const EngineCommand &cmd) {
    while (running_.load(std::memory_order_acquire) &&
           !command_queue_.push(cmd)) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  }

  void enqueue_oe_report(const OeReport &r) {
    while (running_.load(std::memory_order_acquire) &&
           !oe_report_queue_.push(r)) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  }

  void enqueue_md_report(const MdReport &r) {
    while (running_.load(std::memory_order_acquire) &&
           !md_report_queue_.push(r)) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  }

  // ===========================================================
  // Engine thread
  // ===========================================================

  void engine_loop() {
    while (running_.load(std::memory_order_acquire)) {
      EngineCommand cmd{};
      if (!command_queue_.pop(&cmd)) {
        if (snapshot_due_.exchange(false, std::memory_order_acq_rel)) {
          flush_snapshot_to_queue();
        }
        if constexpr (std::is_same_v<EngineT, replay::ReplayEngine>) {
          if (!engine_.feed_exhausted()) {
            engine_.advance();
            flush_engine_events_async(0, 0, 0);
            // When the next message isn't due yet, yield to the IO thread
            // so the outbound pump can dispatch client responses without
            // waiting for the replay to catch up to real time.
            if (engine_.has_deferred_message()) {
              std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
          } else {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
          }
        } else {
          std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        continue;
      }
      process_engine_command(cmd);
      flush_engine_events_async(cmd.session_id, cmd.client_id, cmd.req_seq);
      if (snapshot_due_.exchange(false, std::memory_order_acq_rel)) {
        flush_snapshot_to_queue();
      }
    }
  }

  // Inline path: advance the replay feed one step and flush resulting events.
  // No-op for non-replay engines.
  void advance_feed() override {
    if constexpr (std::is_same_v<EngineT, replay::ReplayEngine>) {
      if (engine_.advance()) {
        inline_flush_events();
      }
    }
  }

  // Called by the feed message callback (ReplayEngine only): forward historical
  // feed messages to the live client's market data channel.
  void on_feed_message(const ndfex::md::Message &msg) {
    if constexpr (std::is_same_v<EngineT, replay::ReplayEngine>) {
      const auto seq = md_seq_.fetch_add(1, std::memory_order_relaxed);
      MdReport r{};
      r.message = msg;
      r.message.header.seq_num = seq;
      enqueue_md_report(r);

      // Keep md_orders_ in sync so the snapshot timer reflects historical
      // state.
      switch (msg.header.msg_type) {
      case ndfex::md::MSG_TYPE::NEW_ORDER: {
        const auto &no = msg.body.new_order;
        md_orders_[no.order_id] = MdOrderState{no.order_id, no.symbol, no.side,
                                               no.quantity, no.price};
        break;
      }
      case ndfex::md::MSG_TYPE::DELETE_ORDER:
        md_orders_.erase(msg.body.delete_order.order_id);
        break;
      case ndfex::md::MSG_TYPE::MODIFY_ORDER: {
        auto it = md_orders_.find(msg.body.modify_order.order_id);
        if (it != md_orders_.end()) {
          it->second.quantity = msg.body.modify_order.quantity;
        }
        break;
      }
      default:
        break;
      }
    }
  }

  void process_engine_command(const EngineCommand &cmd) {
    switch (cmd.type) {
    case CmdType::NewOrder: {
      const auto exch_id = exchange_id_counter_++;
      order_owner_session_[cmd.new_order.order_id] = cmd.session_id;
      order_owner_client_id_[cmd.new_order.order_id] = cmd.client_id;
      order_req_seq_[cmd.new_order.order_id] = cmd.req_seq;
      synthetic_to_ndfex_id_[exch_id] = cmd.new_order.order_id;

      matching::NewOrderCommand c{
          .client_order_id = cmd.new_order.order_id,
          .instrument_id = cmd.new_order.symbol,
          .side = ndfex_side_to_matching(cmd.new_order.side),
          .order_type = matching::OrderType::Limit,
          .tif = ndfex_flags_to_tif(cmd.new_order.flags),
          .price_tick =
              static_cast<matching::price_tick_t>(cmd.new_order.price),
          .quantity = cmd.new_order.quantity,
      };
      engine_.submit(c);
      break;
    }
    case CmdType::DeleteOrder: {
      order_owner_session_[cmd.delete_order.order_id] = cmd.session_id;
      order_owner_client_id_[cmd.delete_order.order_id] = cmd.client_id;
      order_req_seq_[cmd.delete_order.order_id] = cmd.req_seq;

      matching::CancelOrderCommand c{
          .client_order_id = cmd.delete_order.order_id,
          .target_order_id = cmd.delete_order.order_id,
          .cancel_quantity = 0,
      };
      engine_.cancel(c);
      break;
    }
    case CmdType::ModifyOrder: {
      const auto orig = cmd.modify_order.original_id;
      const auto syn = cmd.modify_order.synthetic_id;

      order_owner_session_[syn] = cmd.session_id;
      order_owner_client_id_[syn] = cmd.client_id;
      order_req_seq_[syn] = cmd.req_seq;
      synthetic_to_ndfex_id_[syn] = orig;
      pending_modify_original_ids_.insert(orig);

      matching::ModifyOrderCommand c{
          .original_order_id = orig,
          .new_order_id = syn,
          .new_price_tick =
              static_cast<matching::price_tick_t>(cmd.modify_order.price),
          .new_quantity = cmd.modify_order.quantity,
      };
      engine_.modify(c);
      break;
    }
    }
  }

  void flush_engine_events_async(session_id_t /*cmd_sid*/,
                                 ndfex::oe::client_id_t /*cmd_client*/,
                                 ndfex::seq_num_t /*cmd_seq*/) {
    const auto events = engine_.drain_events();
    for (const auto &event : events) {
      process_event_async(event);
    }
    owner_->logical_clock_ = engine_.logical_clock();
  }

  void process_event_async(const matching::ExecutionEvent &event) {
    // Resolve NDFEX order_id and session metadata
    const bool is_synthetic = synthetic_to_ndfex_id_.count(event.order_id) > 0;
    const ndfex::order_id_t ndfex_id =
        is_synthetic ? synthetic_to_ndfex_id_[event.order_id] : event.order_id;

    const auto session_it = order_owner_session_.find(event.order_id);
    if (session_it == order_owner_session_.end()) {
      return;
    }
    const session_id_t sid = session_it->second;
    const ndfex::oe::client_id_t cid =
        order_owner_client_id_.count(event.order_id)
            ? order_owner_client_id_[event.order_id]
            : 0;
    const ndfex::seq_num_t req_seq = order_req_seq_.count(event.order_id)
                                         ? order_req_seq_[event.order_id]
                                         : 0;

    // Suppress the Cancel event that is the internal cancel step of a modify
    if (event.type == matching::EventType::Cancel &&
        pending_modify_original_ids_.count(event.order_id) > 0) {
      pending_modify_original_ids_.erase(event.order_id);
      // Don't delete md_orders_ entry yet; Ack for synthetic will update it.
      return;
    }

    // Build and enqueue OE response
    OeReport oe{};
    oe.session_id = sid;
    oe.client_id = cid;
    oe.req_seq = req_seq;
    oe.last_seq = req_seq;
    build_oe_response(event, ndfex_id, is_synthetic, &oe);
    enqueue_oe_report(oe);

    // Build and enqueue MD reports
    emit_md_reports_async(event, ndfex_id, is_synthetic);

    // Cleanup
    if (event.type == matching::EventType::Fill &&
        event.remaining_quantity == 0) {
      order_owner_session_.erase(event.order_id);
      order_owner_client_id_.erase(event.order_id);
      order_req_seq_.erase(event.order_id);
    }
    if (event.type == matching::EventType::Cancel) {
      order_owner_session_.erase(event.order_id);
      order_owner_client_id_.erase(event.order_id);
      order_req_seq_.erase(event.order_id);
    }
    if (is_synthetic) {
      synthetic_to_ndfex_id_.erase(event.order_id);
    }
  }

  void build_oe_response(const matching::ExecutionEvent &event,
                         ndfex::order_id_t ndfex_id, bool is_synthetic,
                         OeReport *out) {
    switch (event.type) {
    case matching::EventType::Ack: {
      ndfex::oe::OrderAckResponse body{};
      body.order_id = ndfex_id;
      body.exch_order_id = exchange_id_counter_ - 1;
      body.quantity = event.remaining_quantity;
      body.price = static_cast<ndfex::price_t>(event.price_tick);
      out->response.header =
          make_oe_response_header(ndfex::oe::MSG_TYPE::ACK, out->client_id,
                                  out->req_seq, out->last_seq);
      out->response.body.order_ack_response = body;
      out->body_size = static_cast<std::uint16_t>(sizeof(body));
      if (event.type == matching::EventType::Ack) {
        owner_->emit_order_ack(out->session_id, ndfex_id, body.exch_order_id,
                               event.logical_clock);
      }
      break;
    }
    case matching::EventType::Fill: {
      const bool closed = event.remaining_quantity == 0;
      ndfex::oe::OrderFillResponse body{};
      body.order_id = ndfex_id;
      body.quantity = event.quantity;
      body.price = static_cast<ndfex::price_t>(event.price_tick);
      body.flags = closed ? ndfex::oe::FILL_FLAGS::CLOSED
                          : ndfex::oe::FILL_FLAGS::PARTIAL_FILL;
      out->response.header =
          make_oe_response_header(ndfex::oe::MSG_TYPE::FILL, out->client_id,
                                  out->req_seq, out->last_seq);
      out->response.body.order_fill_response = body;
      out->body_size = static_cast<std::uint16_t>(sizeof(body));
      break;
    }
    case matching::EventType::Cancel: {
      ndfex::oe::OrderClosedResponse body{};
      body.order_id = ndfex_id;
      out->response.header =
          make_oe_response_header(ndfex::oe::MSG_TYPE::CLOSE, out->client_id,
                                  out->req_seq, out->last_seq);
      out->response.body.order_closed_response = body;
      out->body_size = static_cast<std::uint16_t>(sizeof(body));
      break;
    }
    case matching::EventType::Reject: {
      owner_->stats_.total_rejections++;
      ndfex::oe::OrderRejectResponse body{};
      body.order_id = ndfex_id;
      body.reject_reason = matching_reason_to_ndfex(event.reason);
      out->response.header =
          make_oe_response_header(ndfex::oe::MSG_TYPE::REJECT, out->client_id,
                                  out->req_seq, out->last_seq);
      out->response.body.order_reject_response = body;
      out->body_size = static_cast<std::uint16_t>(sizeof(body));
      break;
    }
    }
  }

  void emit_md_reports_async(const matching::ExecutionEvent &event,
                             ndfex::order_id_t ndfex_id, bool is_synthetic) {
    const ndfex::seq_num_t seq =
        md_seq_.fetch_add(1, std::memory_order_relaxed);

    switch (event.type) {
    case matching::EventType::Ack: {
      if (event.remaining_quantity == 0)
        break; // fully matched before resting — no MD NewOrder

      MdOrderState state{};
      state.order_id = ndfex_id;
      state.symbol = static_cast<ndfex::symbol_id_t>(event.instrument_id);
      state.side = matching_side_to_ndfex(event.side);
      state.quantity = event.remaining_quantity;
      state.price = static_cast<ndfex::price_t>(event.price_tick);

      if (is_synthetic && md_orders_.count(ndfex_id) > 0) {
        // Modify ack: update existing entry and publish MODIFY_ORDER
        md_orders_[ndfex_id] = state;
        ndfex::md::ModifyOrder body{};
        body.order_id = ndfex_id;
        body.side = state.side;
        body.quantity = state.quantity;
        body.price = state.price;
        MdReport r{};
        r.message.header = make_md_header(ndfex::md::MSG_TYPE::MODIFY_ORDER,
                                          seq, sizeof(body));
        r.message.body.modify_order = body;
        enqueue_md_report(r);
      } else {
        // Normal new order: publish NEW_ORDER
        md_orders_[ndfex_id] = state;
        ndfex::md::NewOrder body{};
        body.order_id = ndfex_id;
        body.symbol = state.symbol;
        body.side = state.side;
        body.quantity = state.quantity;
        body.price = state.price;
        body.flags = 0;
        MdReport r{};
        r.message.header =
            make_md_header(ndfex::md::MSG_TYPE::NEW_ORDER, seq, sizeof(body));
        r.message.body.new_order = body;
        enqueue_md_report(r);
      }
      break;
    }
    case matching::EventType::Fill: {
      // Trade message for this order
      ndfex::md::Trade trade{};
      trade.order_id = ndfex_id;
      trade.quantity = event.quantity;
      trade.price = static_cast<ndfex::price_t>(event.price_tick);

      MdReport trade_r{};
      trade_r.message.header =
          make_md_header(ndfex::md::MSG_TYPE::TRADE, seq, sizeof(trade));
      trade_r.message.body.trade = trade;
      enqueue_md_report(trade_r);

      // Update MD order state
      const auto seq2 = md_seq_.fetch_add(1, std::memory_order_relaxed);
      auto it = md_orders_.find(ndfex_id);
      if (it != md_orders_.end()) {
        if (event.remaining_quantity == 0) {
          ndfex::md::DeleteOrder del{};
          del.order_id = ndfex_id;
          MdReport del_r{};
          del_r.message.header = make_md_header(
              ndfex::md::MSG_TYPE::DELETE_ORDER, seq2, sizeof(del));
          del_r.message.body.delete_order = del;
          enqueue_md_report(del_r);
          md_orders_.erase(it);
        } else {
          it->second.quantity = event.remaining_quantity;
          ndfex::md::ModifyOrder mod{};
          mod.order_id = ndfex_id;
          mod.side = it->second.side;
          mod.quantity = event.remaining_quantity;
          mod.price = it->second.price;
          MdReport mod_r{};
          mod_r.message.header = make_md_header(
              ndfex::md::MSG_TYPE::MODIFY_ORDER, seq2, sizeof(mod));
          mod_r.message.body.modify_order = mod;
          enqueue_md_report(mod_r);
        }
      }

      // TradeSummary: only emit once (for taker side events, identified by
      // counterparty != 0)
      if (event.counterparty_order_id != 0) {
        const auto seq3 = md_seq_.fetch_add(1, std::memory_order_relaxed);
        const auto sym =
            md_orders_.count(ndfex_id)
                ? md_orders_[ndfex_id].symbol
                : static_cast<ndfex::symbol_id_t>(event.instrument_id);
        ndfex::md::TradeSummary ts{};
        ts.symbol = sym;
        ts.aggressor_side = matching_side_to_ndfex(event.side);
        ts.total_quantity = event.quantity;
        ts.last_price = static_cast<ndfex::price_t>(event.price_tick);
        MdReport ts_r{};
        ts_r.message.header = make_md_header(ndfex::md::MSG_TYPE::TRADE_SUMMARY,
                                             seq3, sizeof(ts));
        ts_r.message.body.trade_summary = ts;
        enqueue_md_report(ts_r);
      }
      break;
    }
    case matching::EventType::Cancel: {
      auto it = md_orders_.find(ndfex_id);
      if (it != md_orders_.end()) {
        ndfex::md::DeleteOrder del{};
        del.order_id = ndfex_id;
        MdReport r{};
        r.message.header =
            make_md_header(ndfex::md::MSG_TYPE::DELETE_ORDER, seq, sizeof(del));
        r.message.body.delete_order = del;
        enqueue_md_report(r);
        md_orders_.erase(it);
      }
      break;
    }
    case matching::EventType::Reject:
      break;
    }
  }

  // ===========================================================
  // Outbound pump (IO thread, async mode)
  // ===========================================================

  void schedule_outbound_pump() {
    outbound_timer_.expires_after(std::chrono::milliseconds(1));
    outbound_timer_.async_wait([this](const error_code &ec) {
      if (ec || !running_.load(std::memory_order_acquire)) {
        return;
      }

      OeReport oe{};
      while (oe_report_queue_.pop(&oe)) {
        handle_oe_outbound(oe);
      }

      MdReport md{};
      while (md_report_queue_.pop(&md)) {
        handle_md_outbound(md);
      }

      schedule_outbound_pump();
    });
  }

  void handle_oe_outbound(const OeReport &r) {
    if (r.session_id != 0) {
      auto it = sessions_by_id_.find(r.session_id);
      if (it != sessions_by_id_.end()) {
        if (auto session = it->second.lock()) {
          auto bytes =
              build_oe_bytes(r.response.header.msg_type, r.client_id, r.req_seq,
                             r.last_seq, &r.response.body, r.body_size);
          queue_write(session, std::move(bytes));
        }
      }
    }

    // Fire shared callbacks
    if (ndfex_response_callback_) {
      ndfex_response_callback_(r.session_id, r.response);
    }
  }

  void handle_md_outbound(const MdReport &r) {
    const bool is_snapshot =
        (r.message.header.magic_number == ndfex::md::SNAPSHOT_MAGIC_NUMBER);
    udp::socket &sock = is_snapshot ? snap_socket_ : md_socket_;
    udp::endpoint &endpt = is_snapshot ? snap_endpoint_ : md_endpoint_;

    if (sock.is_open()) {
      const auto body_size = static_cast<std::size_t>(
          r.message.header.length - sizeof(ndfex::md::Header));
      const std::size_t total = sizeof(ndfex::md::Header) + body_size;
      auto buf = std::make_shared<std::vector<std::uint8_t>>(total);
      std::memcpy(buf->data(), &r.message.header, sizeof(ndfex::md::Header));
      if (body_size > 0) {
        std::memcpy(buf->data() + sizeof(ndfex::md::Header), &r.message.body,
                    body_size);
      }
      sock.async_send_to(asio::buffer(*buf), endpt,
                         [buf](const error_code &, std::size_t) { (void)buf; });
    }
    if (md_callback_) {
      md_callback_(r.message);
    }
  }

  // ===========================================================
  // Snapshot service (engine thread — called when snapshot_due_ fires)
  // ===========================================================

  void flush_snapshot_to_queue() {
    std::map<ndfex::symbol_id_t, std::vector<const MdOrderState *>> by_symbol;
    for (const auto &[id, state] : md_orders_) {
      by_symbol[state.symbol].push_back(&state);
    }

    const ndfex::seq_num_t watermark = md_seq_.load(std::memory_order_relaxed);
    ndfex::seq_num_t seq = watermark;

    for (ndfex::symbol_id_t symbol = 1; symbol <= 13; ++symbol) {
      const auto sym_it = by_symbol.find(symbol);
      const bool has_orders = sym_it != by_symbol.end();

      ndfex::quantity_t bid_count = 0, ask_count = 0;
      if (has_orders) {
        for (const auto *o : sym_it->second) {
          if (o->side == ndfex::SIDE::BUY)
            bid_count++;
          else
            ask_count++;
        }
      }

      ndfex::md::SnapshotInfo info{};
      info.symbol = symbol;
      info.bid_count = bid_count;
      info.ask_count = ask_count;
      info.last_md_seq_num = watermark;

      MdReport info_r{};
      info_r.message.header = make_md_header(ndfex::md::MSG_TYPE::SNAPSHOT_INFO,
                                             seq++, sizeof(info), true);
      info_r.message.body.snapshot_info = info;
      enqueue_md_report(info_r);

      if (has_orders) {
        for (const auto *o : sym_it->second) {
          ndfex::md::NewOrder no{};
          no.order_id = o->order_id;
          no.symbol = o->symbol;
          no.side = o->side;
          no.quantity = o->quantity;
          no.price = o->price;
          no.flags = 0;

          MdReport no_r{};
          no_r.message.header = make_md_header(ndfex::md::MSG_TYPE::NEW_ORDER,
                                               seq++, sizeof(no), true);
          no_r.message.body.new_order = no;
          enqueue_md_report(no_r);
        }
      }
    }
  }

  // ===========================================================
  // Inline processing (test / synchronous path)
  // ===========================================================

  ndfex::oe::client_id_t
  inline_client_id(session_id_t sid, ndfex::oe::client_id_t fallback) const {
    const auto it = inline_session_client_ids_.find(sid);
    return it != inline_session_client_ids_.end() ? it->second : fallback;
  }

  void handle_login_inline(session_id_t session_id,
                           const ndfex::oe::Request &req,
                           ndfex::oe::client_id_t client_id,
                           ndfex::seq_num_t req_seq) {
    // Accept all logins; fire callback so tests can inspect.
    ndfex::oe::Response resp{};
    resp.header = make_oe_response_header(ndfex::oe::MSG_TYPE::LOGIN_RESPONSE,
                                          client_id, req_seq, req_seq);
    resp.body.login_response.session_id =
        static_cast<ndfex::oe::session_id_t>(session_id);
    resp.body.login_response.status = ndfex::oe::LOGIN_STATUS::SUCCESS;
    inline_session_client_ids_[session_id] = client_id;
    inline_emit_oe_response(session_id, resp);
  }

  void handle_new_order_inline(session_id_t sid,
                               const ndfex::oe::NewOrderRequest &req,
                               ndfex::oe::client_id_t cid,
                               ndfex::seq_num_t req_seq) {
    order_owner_session_[req.order_id] = sid;
    order_owner_client_id_[req.order_id] = cid;
    order_req_seq_[req.order_id] = req_seq;

    matching::NewOrderCommand c{
        .client_order_id = req.order_id,
        .instrument_id = req.symbol,
        .side = ndfex_side_to_matching(req.side),
        .order_type = matching::OrderType::Limit,
        .tif = ndfex_flags_to_tif(req.flags),
        .price_tick = static_cast<matching::price_tick_t>(req.price),
        .quantity = req.quantity,
    };
    engine_.submit(c);
    owner_->stats_.total_orders_submitted++;
    inline_flush_events();
  }

  void handle_delete_order_inline(session_id_t sid,
                                  const ndfex::oe::DeleteOrderRequest &req,
                                  ndfex::oe::client_id_t cid,
                                  ndfex::seq_num_t req_seq) {
    order_owner_session_[req.order_id] = sid;
    order_owner_client_id_[req.order_id] = cid;
    order_req_seq_[req.order_id] = req_seq;

    matching::CancelOrderCommand c{
        .client_order_id = req.order_id,
        .target_order_id = req.order_id,
        .cancel_quantity = 0,
    };
    engine_.cancel(c);
    owner_->stats_.total_cancels_submitted++;
    inline_flush_events();
  }

  void handle_modify_order_inline(session_id_t sid,
                                  const ndfex::oe::ModifyOrderRequest &req,
                                  ndfex::oe::client_id_t cid,
                                  ndfex::seq_num_t req_seq) {
    const matching::order_id_t syn = exchange_id_counter_++;
    order_owner_session_[syn] = sid;
    order_owner_client_id_[syn] = cid;
    order_req_seq_[syn] = req_seq;
    synthetic_to_ndfex_id_[syn] = req.order_id;
    pending_modify_original_ids_.insert(req.order_id);

    matching::ModifyOrderCommand c{
        .original_order_id = req.order_id,
        .new_order_id = syn,
        .new_price_tick = static_cast<matching::price_tick_t>(req.price),
        .new_quantity = req.quantity,
    };
    engine_.modify(c);
    owner_->stats_.total_modifies_submitted++;
    inline_flush_events();
  }

  void inline_flush_events() {
    const auto events = engine_.drain_events();
    for (const auto &event : events) {
      process_event_inline(event);
    }
  }

  void process_event_inline(const matching::ExecutionEvent &event) {
    const bool is_synthetic = synthetic_to_ndfex_id_.count(event.order_id) > 0;
    const ndfex::order_id_t ndfex_id =
        is_synthetic ? synthetic_to_ndfex_id_[event.order_id] : event.order_id;

    const auto sit = order_owner_session_.find(event.order_id);
    if (sit == order_owner_session_.end())
      return;
    const session_id_t sid = sit->second;
    const ndfex::oe::client_id_t cid =
        order_owner_client_id_.count(event.order_id)
            ? order_owner_client_id_[event.order_id]
            : 0;
    const ndfex::seq_num_t req_seq = order_req_seq_.count(event.order_id)
                                         ? order_req_seq_[event.order_id]
                                         : 0;

    // Suppress internal Cancel from a pending modify
    if (event.type == matching::EventType::Cancel &&
        pending_modify_original_ids_.count(event.order_id) > 0) {
      pending_modify_original_ids_.erase(event.order_id);
      return;
    }

    // Build and fire OE response
    OeReport r{};
    r.session_id = sid;
    r.client_id = cid;
    r.req_seq = req_seq;
    r.last_seq = req_seq;
    build_oe_response(event, ndfex_id, is_synthetic, &r);
    inline_emit_oe_response(sid, r.response);

    // Build and fire MD
    inline_emit_md(event, ndfex_id, is_synthetic);

    // Cleanup
    if (event.type == matching::EventType::Fill &&
        event.remaining_quantity == 0) {
      order_owner_session_.erase(event.order_id);
      order_owner_client_id_.erase(event.order_id);
      order_req_seq_.erase(event.order_id);
    }
    if (event.type == matching::EventType::Cancel) {
      order_owner_session_.erase(event.order_id);
      order_owner_client_id_.erase(event.order_id);
      order_req_seq_.erase(event.order_id);
    }
    if (is_synthetic) {
      synthetic_to_ndfex_id_.erase(event.order_id);
    }
  }

  void inline_emit_oe_response(session_id_t sid,
                               const ndfex::oe::Response &resp) {
    if (ndfex_response_callback_) {
      ndfex_response_callback_(sid, resp);
    }
  }

  void inline_emit_md(const matching::ExecutionEvent &event,
                      ndfex::order_id_t ndfex_id, bool is_synthetic) {
    const ndfex::seq_num_t seq =
        md_seq_.fetch_add(1, std::memory_order_relaxed);

    switch (event.type) {
    case matching::EventType::Ack: {
      if (event.remaining_quantity == 0)
        break;

      MdOrderState state{};
      state.order_id = ndfex_id;
      state.symbol = static_cast<ndfex::symbol_id_t>(event.instrument_id);
      state.side = matching_side_to_ndfex(event.side);
      state.quantity = event.remaining_quantity;
      state.price = static_cast<ndfex::price_t>(event.price_tick);

      if (is_synthetic && md_orders_.count(ndfex_id) > 0) {
        md_orders_[ndfex_id] = state;
        ndfex::md::Message msg{};
        msg.header = make_md_header(ndfex::md::MSG_TYPE::MODIFY_ORDER, seq,
                                    sizeof(ndfex::md::ModifyOrder));
        msg.body.modify_order.order_id = ndfex_id;
        msg.body.modify_order.side = state.side;
        msg.body.modify_order.quantity = state.quantity;
        msg.body.modify_order.price = state.price;
        if (md_callback_)
          md_callback_(msg);
      } else {
        md_orders_[ndfex_id] = state;
        ndfex::md::Message msg{};
        msg.header = make_md_header(ndfex::md::MSG_TYPE::NEW_ORDER, seq,
                                    sizeof(ndfex::md::NewOrder));
        msg.body.new_order.order_id = ndfex_id;
        msg.body.new_order.symbol = state.symbol;
        msg.body.new_order.side = state.side;
        msg.body.new_order.quantity = state.quantity;
        msg.body.new_order.price = state.price;
        msg.body.new_order.flags = 0;
        if (md_callback_)
          md_callback_(msg);
      }
      break;
    }
    case matching::EventType::Fill: {
      ndfex::md::Message trade_msg{};
      trade_msg.header = make_md_header(ndfex::md::MSG_TYPE::TRADE, seq,
                                        sizeof(ndfex::md::Trade));
      trade_msg.body.trade.order_id = ndfex_id;
      trade_msg.body.trade.quantity = event.quantity;
      trade_msg.body.trade.price =
          static_cast<ndfex::price_t>(event.price_tick);
      if (md_callback_)
        md_callback_(trade_msg);

      const ndfex::seq_num_t seq2 =
          md_seq_.fetch_add(1, std::memory_order_relaxed);
      auto it = md_orders_.find(ndfex_id);
      if (it != md_orders_.end()) {
        if (event.remaining_quantity == 0) {
          ndfex::md::Message del_msg{};
          del_msg.header = make_md_header(ndfex::md::MSG_TYPE::DELETE_ORDER,
                                          seq2, sizeof(ndfex::md::DeleteOrder));
          del_msg.body.delete_order.order_id = ndfex_id;
          if (md_callback_)
            md_callback_(del_msg);
          md_orders_.erase(it);
        } else {
          it->second.quantity = event.remaining_quantity;
          ndfex::md::Message mod_msg{};
          mod_msg.header = make_md_header(ndfex::md::MSG_TYPE::MODIFY_ORDER,
                                          seq2, sizeof(ndfex::md::ModifyOrder));
          mod_msg.body.modify_order.order_id = ndfex_id;
          mod_msg.body.modify_order.side = it->second.side;
          mod_msg.body.modify_order.quantity = event.remaining_quantity;
          mod_msg.body.modify_order.price = it->second.price;
          if (md_callback_)
            md_callback_(mod_msg);
        }
      }

      if (event.counterparty_order_id != 0) {
        const ndfex::seq_num_t seq3 =
            md_seq_.fetch_add(1, std::memory_order_relaxed);
        const auto sym =
            md_orders_.count(ndfex_id)
                ? md_orders_[ndfex_id].symbol
                : static_cast<ndfex::symbol_id_t>(event.instrument_id);
        ndfex::md::Message ts_msg{};
        ts_msg.header = make_md_header(ndfex::md::MSG_TYPE::TRADE_SUMMARY, seq3,
                                       sizeof(ndfex::md::TradeSummary));
        ts_msg.body.trade_summary.symbol = sym;
        ts_msg.body.trade_summary.aggressor_side =
            matching_side_to_ndfex(event.side);
        ts_msg.body.trade_summary.total_quantity = event.quantity;
        ts_msg.body.trade_summary.last_price =
            static_cast<ndfex::price_t>(event.price_tick);
        if (md_callback_)
          md_callback_(ts_msg);
      }
      break;
    }
    case matching::EventType::Cancel: {
      auto it = md_orders_.find(ndfex_id);
      if (it != md_orders_.end()) {
        ndfex::md::Message del_msg{};
        del_msg.header = make_md_header(ndfex::md::MSG_TYPE::DELETE_ORDER, seq,
                                        sizeof(ndfex::md::DeleteOrder));
        del_msg.body.delete_order.order_id = ndfex_id;
        if (md_callback_)
          md_callback_(del_msg);
        md_orders_.erase(it);
      }
      break;
    }
    case matching::EventType::Reject:
      break;
    }
  }
}; // end template class NdfexGateway::Impl<EngineT>

// Explicit instantiations so the template body lives only in this TU.
template class NdfexGateway::Impl<matching::MatchingEngine>;
template class NdfexGateway::Impl<replay::ReplayEngine>;

// =============================================================================
// NdfexGateway public methods
// =============================================================================

NdfexGateway::NdfexGateway(logical_clock_t start_clock)
    : Gateway(start_clock),
      impl_(std::make_unique<Impl<matching::MatchingEngine>>(this,
                                                             start_clock)) {}

NdfexGateway::NdfexGateway(replay::ReplayEngine engine)
    : Gateway(0), impl_(std::make_unique<Impl<replay::ReplayEngine>>(
                      this, std::move(engine))) {}

NdfexGateway::~NdfexGateway() = default;

bool NdfexGateway::start(std::uint16_t oe_port, const std::string &md_host,
                         std::uint16_t md_port, const std::string &snap_host,
                         std::uint16_t snap_port,
                         const std::string &udp_interface) {
  return impl_->start(oe_port, md_host, md_port, snap_host, snap_port,
                      udp_interface);
}

void NdfexGateway::shutdown() { impl_->shutdown(); }

void NdfexGateway::process_frame(session_id_t /*session_id*/,
                                 const protocol::FrameHeader & /*header*/,
                                 const std::vector<std::uint8_t> & /*body*/
) {
  // Not used: NdfexGateway speaks NDFEX OE protocol, not the ghostbook frame
  // protocol.
}

void NdfexGateway::process_ndfex_request(session_id_t session_id,
                                         const ndfex::oe::Request &request) {
  impl_->process_inline(session_id, request);
}

session_id_t
NdfexGateway::create_test_session(ndfex::oe::client_id_t client_id) {
  return impl_->create_test_session(client_id);
}

void NdfexGateway::set_market_data_callback(NdfexMarketDataCallback cb) {
  impl_->md_callback_ = std::move(cb);
}

void NdfexGateway::set_ndfex_response_callback(NdfexResponseCallback cb) {
  impl_->ndfex_response_callback_ = std::move(cb);
}

void NdfexGateway::advance_feed() { impl_->advance_feed(); }

} // namespace ghostbook::gateway
