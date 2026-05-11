#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "ghostbook/ndfex/protocol.h"
#include "ndfex_gateway.h"

using ghostbook::gateway::NdfexGateway;
namespace oe = ghostbook::ndfex::oe;
namespace md = ghostbook::ndfex::md;

// =============================================================================
// Helpers
// =============================================================================

namespace {

bool expect(bool cond, const std::string &label) {
  if (!cond) {
    std::cerr << "  FAIL  " << label << "\n";
    return false;
  }
  std::cout << "  PASS  " << label << "\n";
  return true;
}

template <typename T>
bool expect_eq(T actual, T expected, const std::string &label) {
  if (actual == expected) {
    std::cout << "  PASS  " << label
              << " == " << static_cast<std::uint64_t>(actual) << "\n";
    return true;
  }
  std::cerr << "  FAIL  " << label << ": expected "
            << static_cast<std::uint64_t>(expected) << ", got "
            << static_cast<std::uint64_t>(actual) << "\n";
  return false;
}

// Bind a UDP socket to receive datagrams on the given port (any interface).
int bind_udp_rx(std::uint16_t port) {
  int fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0)
    return -1;
  int yes = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  if (bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
    close(fd);
    return -1;
  }
  return fd;
}

// Receive one MD datagram. Returns false on timeout or short read.
bool recv_md_datagram(int fd, md::Message *out, int timeout_ms) {
  pollfd pfd{fd, POLLIN, 0};
  if (poll(&pfd, 1, timeout_ms) <= 0)
    return false;
  std::uint8_t buf[1500]{};
  const ssize_t n = recv(fd, buf, sizeof(buf), 0);
  if (n < static_cast<ssize_t>(sizeof(md::Header)))
    return false;
  std::memcpy(&out->header, buf, sizeof(md::Header));
  const std::size_t body_sz = out->header.length > sizeof(md::Header)
                                  ? out->header.length - sizeof(md::Header)
                                  : 0;
  if (body_sz > 0 && body_sz <= sizeof(out->body)) {
    std::memcpy(&out->body, buf + sizeof(md::Header), body_sz);
  }
  return true;
}

// Drain all datagrams already buffered in the OS receive buffer.
// window_ms: timeout for the first datagram; inter_ms: timeout between each
// subsequent one. Use a short inter_ms (e.g. 20ms) so the call returns quickly
// once the buffer is empty.
std::vector<md::Message> collect_md_datagrams(int fd, int window_ms,
                                              int inter_ms = 20) {
  std::vector<md::Message> msgs;
  md::Message msg{};
  if (!recv_md_datagram(fd, &msg, window_ms))
    return msgs;
  msgs.push_back(msg);
  while (recv_md_datagram(fd, &msg, inter_ms)) {
    msgs.push_back(msg);
  }
  return msgs;
}

// TCP OE helpers (same pattern as the TCP integration tests)

bool recv_exact(int fd, std::uint8_t *buf, std::size_t len, int timeout_ms) {
  std::size_t total = 0;
  while (total < len) {
    pollfd pfd{fd, POLLIN, 0};
    if (poll(&pfd, 1, timeout_ms) <= 0)
      return false;
    const ssize_t n = recv(fd, buf + total, len - total, 0);
    if (n <= 0)
      return false;
    total += static_cast<std::size_t>(n);
  }
  return true;
}

bool read_oe_response(int fd, oe::Response *out, int timeout_ms = 3000) {
  oe::ResponseHeader hdr{};
  if (!recv_exact(fd, reinterpret_cast<std::uint8_t *>(&hdr), sizeof(hdr),
                  timeout_ms))
    return false;
  const std::size_t body_sz =
      hdr.length > sizeof(hdr) ? hdr.length - sizeof(hdr) : 0;
  out->header = hdr;
  if (body_sz > 0 && body_sz <= sizeof(out->body)) {
    if (!recv_exact(fd, reinterpret_cast<std::uint8_t *>(&out->body), body_sz,
                    timeout_ms))
      return false;
  }
  return true;
}

bool send_oe_request(int fd, oe::Request req, std::size_t body_size) {
  req.header.length =
      static_cast<std::uint16_t>(sizeof(oe::RequestHeader) + body_size);
  std::vector<std::uint8_t> buf(req.header.length);
  std::memcpy(buf.data(), &req.header, sizeof(req.header));
  if (body_size > 0)
    std::memcpy(buf.data() + sizeof(req.header), &req.body, body_size);
  return send(fd, buf.data(), buf.size(), 0) ==
         static_cast<ssize_t>(buf.size());
}

int connect_with_retry(std::uint16_t port, int max_tries = 20) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0)
    return -1;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  for (int i = 0; i < max_tries; ++i) {
    if (connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0)
      return fd;
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  close(fd);
  return -1;
}

struct OeSession {
  int fd;
  oe::session_id_t session_id;
};

OeSession oe_login(std::uint16_t oe_port, oe::client_id_t client_id) {
  int fd = connect_with_retry(oe_port);
  if (fd < 0)
    return {-1, 0};
  oe::Request req{};
  req.header.msg_type = oe::MSG_TYPE::LOGIN;
  req.header.version = oe::OE_PROTOCOL_VERSION;
  req.header.seq_num = 1;
  req.header.client_id = client_id;
  std::memset(&req.body.login_request, 0, sizeof(req.body.login_request));
  if (!send_oe_request(fd, req, sizeof(oe::LoginRequest)))
    return {-1, 0};
  oe::Response resp{};
  if (!read_oe_response(fd, &resp))
    return {-1, 0};
  return {fd, resp.body.login_response.session_id};
}

// Submit a new order and read the first OE response (ACK or REJECT).
bool submit_order(int fd, oe::client_id_t cid, oe::session_id_t sid,
                  ghostbook::ndfex::seq_num_t seq,
                  ghostbook::ndfex::order_id_t order_id,
                  ghostbook::ndfex::symbol_id_t symbol,
                  ghostbook::ndfex::SIDE side, ghostbook::ndfex::quantity_t qty,
                  ghostbook::ndfex::price_t price,
                  oe::ORDER_FLAGS flags = oe::ORDER_FLAGS::NONE) {
  oe::Request req{};
  req.header.msg_type = oe::MSG_TYPE::NEW_ORDER;
  req.header.version = oe::OE_PROTOCOL_VERSION;
  req.header.seq_num = seq;
  req.header.client_id = cid;
  req.header.session_id = sid;
  req.body.new_order_request.order_id = order_id;
  req.body.new_order_request.symbol = symbol;
  req.body.new_order_request.side = side;
  req.body.new_order_request.quantity = qty;
  req.body.new_order_request.price = price;
  req.body.new_order_request.flags = flags;
  if (!send_oe_request(fd, req, sizeof(oe::NewOrderRequest)))
    return false;
  oe::Response resp{};
  return read_oe_response(fd, &resp);
}

bool cancel_order(int fd, oe::client_id_t cid, oe::session_id_t sid,
                  ghostbook::ndfex::seq_num_t seq,
                  ghostbook::ndfex::order_id_t order_id) {
  oe::Request req{};
  req.header.msg_type = oe::MSG_TYPE::DELETE_ORDER;
  req.header.version = oe::OE_PROTOCOL_VERSION;
  req.header.seq_num = seq;
  req.header.client_id = cid;
  req.header.session_id = sid;
  req.body.delete_order_request.order_id = order_id;
  if (!send_oe_request(fd, req, sizeof(oe::DeleteOrderRequest)))
    return false;
  oe::Response resp{};
  return read_oe_response(fd, &resp);
}

// =============================================================================
// Test 1: Resting order publishes NEW_ORDER on live MD stream
// =============================================================================

bool test_md_new_order_on_live_stream() {
  std::cout << "\n[TEST] md_new_order_on_live_stream\n";

  constexpr std::uint16_t kOePort = 24300;
  constexpr std::uint16_t kMdPort = 24301;
  constexpr std::uint16_t kSnapPort = 24302;

  const int md_fd = bind_udp_rx(kMdPort);
  if (!expect(md_fd >= 0, "bound UDP receive socket for live MD"))
    return false;

  NdfexGateway gateway;
  if (!gateway.start(kOePort, "127.0.0.1", kMdPort, "127.0.0.1", kSnapPort)) {
    close(md_fd);
    std::cerr << "  FAIL  gateway start failed\n";
    return false;
  }

  const auto [oe_fd, sid] = oe_login(kOePort, 1);
  if (!expect(oe_fd >= 0, "OE login succeeded")) {
    close(md_fd);
    gateway.shutdown();
    return false;
  }

  bool ok = true;
  ok &= expect(submit_order(oe_fd, 1, sid, 2, 401, 1,
                            ghostbook::ndfex::SIDE::BUY, 5, 100),
               "resting BUY 5 @ 100 ACK received");

  md::Message msg{};
  ok &= expect(recv_md_datagram(md_fd, &msg, 2000),
               "NEW_ORDER datagram arrives on MD stream");
  ok &= expect(msg.header.magic_number == md::MAGIC_NUMBER,
               "live stream magic is GOIRISH!");
  ok &= expect_eq(static_cast<int>(msg.header.msg_type),
                  static_cast<int>(md::MSG_TYPE::NEW_ORDER),
                  "msg_type is NEW_ORDER");
  ok &= expect_eq(msg.body.new_order.order_id,
                  static_cast<ghostbook::ndfex::order_id_t>(401),
                  "order_id == 401");
  ok &= expect_eq(msg.body.new_order.symbol,
                  static_cast<ghostbook::ndfex::symbol_id_t>(1), "symbol == 1");
  ok &= expect_eq(static_cast<int>(msg.body.new_order.side),
                  static_cast<int>(ghostbook::ndfex::SIDE::BUY), "side == BUY");
  ok &=
      expect_eq(msg.body.new_order.quantity,
                static_cast<ghostbook::ndfex::quantity_t>(5), "quantity == 5");
  ok &= expect_eq(msg.body.new_order.price,
                  static_cast<ghostbook::ndfex::price_t>(100), "price == 100");

  close(oe_fd);
  close(md_fd);
  gateway.shutdown();
  return ok;
}

// =============================================================================
// Test 2: Two crossing orders produce TRADE + DELETE_ORDER + TRADE_SUMMARY
// =============================================================================

bool test_md_trade_messages_on_fill() {
  std::cout << "\n[TEST] md_trade_messages_on_fill\n";

  constexpr std::uint16_t kOePort = 24310;
  constexpr std::uint16_t kMdPort = 24311;
  constexpr std::uint16_t kSnapPort = 24312;

  const int md_fd = bind_udp_rx(kMdPort);
  if (!expect(md_fd >= 0, "bound UDP receive socket for live MD"))
    return false;

  NdfexGateway gateway;
  if (!gateway.start(kOePort, "127.0.0.1", kMdPort, "127.0.0.1", kSnapPort)) {
    close(md_fd);
    std::cerr << "  FAIL  gateway start failed\n";
    return false;
  }

  const auto [seller_fd, seller_sid] = oe_login(kOePort, 10);
  const auto [buyer_fd, buyer_sid] = oe_login(kOePort, 20);
  if (!expect(seller_fd >= 0 && buyer_fd >= 0, "seller and buyer logged in")) {
    close(md_fd);
    gateway.shutdown();
    return false;
  }

  bool ok = true;

  // Seller places resting SELL 3 @ 100
  ok &= expect(submit_order(seller_fd, 10, seller_sid, 2, 501, 1,
                            ghostbook::ndfex::SIDE::SELL, 3, 100),
               "seller resting SELL 3 @ 100 ACK received");

  // Drain the seller's NEW_ORDER from MD before the cross
  {
    md::Message m{};
    recv_md_datagram(md_fd, &m, 500);
  }

  // Buyer crosses with aggressive BUY 3 @ 100; engine emits ACK then fills
  ok &= expect(submit_order(buyer_fd, 20, buyer_sid, 2, 502, 1,
                            ghostbook::ndfex::SIDE::BUY, 3, 100),
               "buyer aggressive BUY 3 @ 100 ACK received");

  // Drain buyer FILL and seller FILL from OE so they are fully processed
  {
    oe::Response r{};
    read_oe_response(buyer_fd, &r);
  }
  {
    oe::Response r{};
    read_oe_response(seller_fd, &r);
  }

  // Collect all live MD datagrams produced by the cross
  const auto msgs = collect_md_datagrams(md_fd, 500);

  bool found_trade = false;
  bool found_delete_501 = false;
  bool found_trade_summary = false;

  for (const auto &m : msgs) {
    if (m.header.msg_type == md::MSG_TYPE::TRADE && m.body.trade.quantity == 3)
      found_trade = true;
    if (m.header.msg_type == md::MSG_TYPE::DELETE_ORDER &&
        m.body.delete_order.order_id == 501)
      found_delete_501 = true;
    if (m.header.msg_type == md::MSG_TYPE::TRADE_SUMMARY)
      found_trade_summary = true;
  }

  const bool all_goirish =
      std::all_of(msgs.begin(), msgs.end(), [](const md::Message &m) {
        return m.header.magic_number == md::MAGIC_NUMBER;
      });

  ok &= expect(!msgs.empty(), "MD datagrams received after cross");
  ok &=
      expect(all_goirish, "all fill-related MD datagrams carry GOIRISH! magic");
  ok &= expect(found_trade, "TRADE with quantity == 3 present on MD stream");
  ok &= expect(found_delete_501,
               "DELETE_ORDER for seller order_id == 501 present on MD stream");
  ok &= expect(found_trade_summary, "TRADE_SUMMARY present on MD stream");

  close(seller_fd);
  close(buyer_fd);
  close(md_fd);
  gateway.shutdown();
  return ok;
}

// =============================================================================
// Test 3: Cancelling a resting order produces DELETE_ORDER on live MD stream
// =============================================================================

bool test_md_delete_on_cancel() {
  std::cout << "\n[TEST] md_delete_on_cancel\n";

  constexpr std::uint16_t kOePort = 24320;
  constexpr std::uint16_t kMdPort = 24321;
  constexpr std::uint16_t kSnapPort = 24322;

  const int md_fd = bind_udp_rx(kMdPort);
  if (!expect(md_fd >= 0, "bound UDP receive socket for live MD"))
    return false;

  NdfexGateway gateway;
  if (!gateway.start(kOePort, "127.0.0.1", kMdPort, "127.0.0.1", kSnapPort)) {
    close(md_fd);
    std::cerr << "  FAIL  gateway start failed\n";
    return false;
  }

  const auto [oe_fd, sid] = oe_login(kOePort, 1);
  if (!expect(oe_fd >= 0, "OE login succeeded")) {
    close(md_fd);
    gateway.shutdown();
    return false;
  }

  bool ok = true;

  ok &= expect(submit_order(oe_fd, 1, sid, 2, 601, 1,
                            ghostbook::ndfex::SIDE::BUY, 5, 100),
               "resting BUY 5 @ 100 ACK received");

  // Drain the NEW_ORDER from MD
  {
    md::Message m{};
    recv_md_datagram(md_fd, &m, 500);
  }

  ok &= expect(cancel_order(oe_fd, 1, sid, 3, 601), "cancel CLOSE received");

  md::Message del{};
  ok &= expect(recv_md_datagram(md_fd, &del, 2000),
               "DELETE_ORDER datagram arrives after cancel");
  ok &= expect(del.header.magic_number == md::MAGIC_NUMBER,
               "delete datagram carries GOIRISH! magic");
  ok &= expect_eq(static_cast<int>(del.header.msg_type),
                  static_cast<int>(md::MSG_TYPE::DELETE_ORDER),
                  "msg_type is DELETE_ORDER");
  ok &= expect_eq(del.body.delete_order.order_id,
                  static_cast<ghostbook::ndfex::order_id_t>(601),
                  "deleted order_id == 601");

  close(oe_fd);
  close(md_fd);
  gateway.shutdown();
  return ok;
}

// =============================================================================
// Test 4: Seq nums on the live MD stream are strictly monotonically increasing
// =============================================================================

bool test_md_seq_nums_monotonic() {
  std::cout << "\n[TEST] md_seq_nums_monotonic\n";

  constexpr std::uint16_t kOePort = 24330;
  constexpr std::uint16_t kMdPort = 24331;
  constexpr std::uint16_t kSnapPort = 24332;

  const int md_fd = bind_udp_rx(kMdPort);
  if (!expect(md_fd >= 0, "bound UDP receive socket for live MD"))
    return false;

  NdfexGateway gateway;
  if (!gateway.start(kOePort, "127.0.0.1", kMdPort, "127.0.0.1", kSnapPort)) {
    close(md_fd);
    std::cerr << "  FAIL  gateway start failed\n";
    return false;
  }

  const auto [oe_fd, sid] = oe_login(kOePort, 1);
  if (!expect(oe_fd >= 0, "OE login succeeded")) {
    close(md_fd);
    gateway.shutdown();
    return false;
  }

  // Three resting orders on distinct symbols — no crosses
  submit_order(oe_fd, 1, sid, 2, 701, 1, ghostbook::ndfex::SIDE::BUY, 5, 100);
  submit_order(oe_fd, 1, sid, 3, 702, 2, ghostbook::ndfex::SIDE::SELL, 3, 200);
  submit_order(oe_fd, 1, sid, 4, 703, 3, ghostbook::ndfex::SIDE::BUY, 7, 150);

  // Cancel two of them
  cancel_order(oe_fd, 1, sid, 5, 701);
  cancel_order(oe_fd, 1, sid, 6, 703);

  // All five operations are done; collect the resulting MD datagrams
  // (3 × NEW_ORDER + 2 × DELETE_ORDER = 5 minimum)
  const auto msgs = collect_md_datagrams(md_fd, 500, 100);

  bool ok = true;
  ok &= expect(msgs.size() >= 5, "at least 5 live MD datagrams received");

  const bool all_goirish =
      std::all_of(msgs.begin(), msgs.end(), [](const md::Message &m) {
        return m.header.magic_number == md::MAGIC_NUMBER;
      });
  ok &= expect(all_goirish, "all live MD datagrams carry GOIRISH! magic");

  bool strictly_increasing = true;
  ghostbook::ndfex::seq_num_t prev = 0;
  for (const auto &m : msgs) {
    if (m.header.seq_num <= prev) {
      strictly_increasing = false;
      break;
    }
    prev = m.header.seq_num;
  }
  ok &= expect(strictly_increasing,
               "live MD seq_nums are strictly monotonically increasing");

  close(oe_fd);
  close(md_fd);
  gateway.shutdown();
  return ok;
}

// =============================================================================
// Test 5: Snapshot stream reflects the resting book with SNAPSHOT magic
// =============================================================================

bool test_snapshot_reflects_resting_book() {
  std::cout << "\n[TEST] snapshot_reflects_resting_book\n";

  constexpr std::uint16_t kOePort = 24340;
  constexpr std::uint16_t kMdPort = 24341;
  constexpr std::uint16_t kSnapPort = 24342;

  const int snap_fd = bind_udp_rx(kSnapPort);
  if (!expect(snap_fd >= 0, "bound UDP receive socket for snapshot stream"))
    return false;

  NdfexGateway gateway;
  if (!gateway.start(kOePort, "127.0.0.1", kMdPort, "127.0.0.1", kSnapPort)) {
    close(snap_fd);
    std::cerr << "  FAIL  gateway start failed\n";
    return false;
  }

  const auto [oe_fd, sid] = oe_login(kOePort, 1);
  if (!expect(oe_fd >= 0, "OE login succeeded")) {
    close(snap_fd);
    gateway.shutdown();
    return false;
  }

  bool ok = true;

  // Two resting orders on different symbols
  ok &= expect(submit_order(oe_fd, 1, sid, 2, 801, 1,
                            ghostbook::ndfex::SIDE::BUY, 5, 100),
               "resting BUY on symbol 1 ACK received");
  ok &= expect(submit_order(oe_fd, 1, sid, 3, 802, 2,
                            ghostbook::ndfex::SIDE::SELL, 3, 200),
               "resting SELL on symbol 2 ACK received");

  // Let multiple snapshot cycles fire (timer period = 100ms)
  std::this_thread::sleep_for(std::chrono::milliseconds(350));

  // Drain all snapshot datagrams buffered during the sleep
  const auto snaps = collect_md_datagrams(snap_fd, 50);

  ok &= expect(!snaps.empty(), "snapshot datagrams received on snap stream");

  const bool all_snap_magic =
      std::all_of(snaps.begin(), snaps.end(), [](const md::Message &m) {
        return m.header.magic_number == md::SNAPSHOT_MAGIC_NUMBER;
      });
  ok &= expect(all_snap_magic, "all snapshot datagrams carry SNAPSHOT magic");

  bool found_sym1_info = false;
  bool found_sym2_info = false;
  bool found_order_801 = false;
  bool found_order_802 = false;

  for (const auto &m : snaps) {
    if (m.header.msg_type == md::MSG_TYPE::SNAPSHOT_INFO) {
      if (m.body.snapshot_info.symbol == 1 &&
          m.body.snapshot_info.bid_count == 1 &&
          m.body.snapshot_info.ask_count == 0)
        found_sym1_info = true;
      if (m.body.snapshot_info.symbol == 2 &&
          m.body.snapshot_info.bid_count == 0 &&
          m.body.snapshot_info.ask_count == 1)
        found_sym2_info = true;
    }
    if (m.header.msg_type == md::MSG_TYPE::NEW_ORDER) {
      if (m.body.new_order.order_id == 801)
        found_order_801 = true;
      if (m.body.new_order.order_id == 802)
        found_order_802 = true;
    }
  }

  ok &= expect(found_sym1_info,
               "SnapshotInfo for symbol 1: bid_count=1, ask_count=0");
  ok &= expect(found_sym2_info,
               "SnapshotInfo for symbol 2: bid_count=0, ask_count=1");
  ok &= expect(found_order_801, "snapshot NewOrder for order_id=801 present");
  ok &= expect(found_order_802, "snapshot NewOrder for order_id=802 present");

  close(oe_fd);
  close(snap_fd);
  gateway.shutdown();
  return ok;
}

// =============================================================================
// Test 6: Snapshot always fires for all 13 symbols; no orders appear after cancel
// =============================================================================

bool test_snapshot_no_orders_after_cancel() {
  std::cout << "\n[TEST] snapshot_no_orders_after_cancel\n";

  constexpr std::uint16_t kOePort = 24350;
  constexpr std::uint16_t kMdPort = 24351;
  constexpr std::uint16_t kSnapPort = 24352;

  const int snap_fd = bind_udp_rx(kSnapPort);
  if (!expect(snap_fd >= 0, "bound UDP receive socket for snapshot stream"))
    return false;

  NdfexGateway gateway;
  if (!gateway.start(kOePort, "127.0.0.1", kMdPort, "127.0.0.1", kSnapPort)) {
    close(snap_fd);
    std::cerr << "  FAIL  gateway start failed\n";
    return false;
  }

  const auto [oe_fd, sid] = oe_login(kOePort, 1);
  if (!expect(oe_fd >= 0, "OE login succeeded")) {
    close(snap_fd);
    gateway.shutdown();
    return false;
  }

  bool ok = true;

  ok &= expect(submit_order(oe_fd, 1, sid, 2, 901, 1,
                            ghostbook::ndfex::SIDE::BUY, 5, 100),
               "resting BUY ACK received");

  // Let snapshots accumulate then verify the order is included
  std::this_thread::sleep_for(std::chrono::milliseconds(350));
  const auto snaps_before = collect_md_datagrams(snap_fd, 50);

  bool found_before = false;
  for (const auto &m : snaps_before) {
    if (m.header.msg_type == md::MSG_TYPE::NEW_ORDER &&
        m.body.new_order.order_id == 901)
      found_before = true;
  }
  ok &= expect(found_before, "order 901 present in snapshot before cancel");

  // Cancel the order; receiving CLOSE confirms the engine has removed it from
  // md_orders_
  ok &= expect(cancel_order(oe_fd, 1, sid, 3, 901), "cancel CLOSE received");

  // Flush any snapshot datagrams that were already buffered before the cancel
  collect_md_datagrams(snap_fd, 50);

  // Snapshot still fires for all 13 symbols, but with empty counts and no
  // NEW_ORDER entries
  std::this_thread::sleep_for(std::chrono::milliseconds(350));
  const auto snaps_after = collect_md_datagrams(snap_fd, 50);

  ok &= expect(!snaps_after.empty(),
               "snapshot still fires after all orders cancelled");

  bool found_order_after_cancel = false;
  bool all_infos_empty = true;
  for (const auto &m : snaps_after) {
    if (m.header.msg_type == md::MSG_TYPE::NEW_ORDER)
      found_order_after_cancel = true;
    if (m.header.msg_type == md::MSG_TYPE::SNAPSHOT_INFO &&
        (m.body.snapshot_info.bid_count != 0 ||
         m.body.snapshot_info.ask_count != 0))
      all_infos_empty = false;
  }

  ok &= expect(!found_order_after_cancel,
               "no NEW_ORDER in snapshot after all orders cancelled");
  ok &= expect(all_infos_empty,
               "all SnapshotInfo entries show 0 bids and 0 asks after cancel");

  close(oe_fd);
  close(snap_fd);
  gateway.shutdown();
  return ok;
}

// =============================================================================
// Test 7: Snapshot sequence covers all 13 known symbols every cycle
// =============================================================================

bool test_snapshot_covers_all_symbols() {
  std::cout << "\n[TEST] snapshot_covers_all_symbols\n";

  constexpr std::uint16_t kOePort = 24360;
  constexpr std::uint16_t kMdPort = 24361;
  constexpr std::uint16_t kSnapPort = 24362;

  const int snap_fd = bind_udp_rx(kSnapPort);
  if (!expect(snap_fd >= 0, "bound UDP receive socket for snapshot stream"))
    return false;

  NdfexGateway gateway;
  if (!gateway.start(kOePort, "127.0.0.1", kMdPort, "127.0.0.1", kSnapPort)) {
    close(snap_fd);
    std::cerr << "  FAIL  gateway start failed\n";
    return false;
  }

  const auto [oe_fd, sid] = oe_login(kOePort, 1);
  if (!expect(oe_fd >= 0, "OE login succeeded")) {
    close(snap_fd);
    gateway.shutdown();
    return false;
  }

  bool ok = true;

  // Place resting orders on two symbols only; the remaining 11 should still
  // appear in the snapshot with empty counts
  ok &= expect(submit_order(oe_fd, 1, sid, 2, 1001, 3,
                            ghostbook::ndfex::SIDE::BUY, 4, 100),
               "resting BUY on symbol 3 ACK received");
  ok &= expect(submit_order(oe_fd, 1, sid, 3, 1002, 7,
                            ghostbook::ndfex::SIDE::SELL, 2, 200),
               "resting SELL on symbol 7 ACK received");

  // Wait for at least one snapshot cycle (timer = 100ms)
  std::this_thread::sleep_for(std::chrono::milliseconds(350));
  const auto snaps = collect_md_datagrams(snap_fd, 50);

  ok &= expect(!snaps.empty(), "snapshot datagrams received");

  // Gather all symbol IDs that appear in SNAPSHOT_INFO messages within a single
  // contiguous snapshot cycle (13 consecutive SNAPSHOT_INFO entries)
  std::array<bool, 14> seen_symbol{};  // index 1-13
  seen_symbol.fill(false);

  bool found_sym3_with_bid = false;
  bool found_sym7_with_ask = false;
  bool found_empty_symbol = false;

  for (const auto &m : snaps) {
    if (m.header.msg_type != md::MSG_TYPE::SNAPSHOT_INFO)
      continue;
    const auto sym = m.body.snapshot_info.symbol;
    if (sym >= 1 && sym <= 13)
      seen_symbol[sym] = true;
    if (sym == 3 && m.body.snapshot_info.bid_count == 1 &&
        m.body.snapshot_info.ask_count == 0)
      found_sym3_with_bid = true;
    if (sym == 7 && m.body.snapshot_info.bid_count == 0 &&
        m.body.snapshot_info.ask_count == 1)
      found_sym7_with_ask = true;
    if (m.body.snapshot_info.bid_count == 0 &&
        m.body.snapshot_info.ask_count == 0)
      found_empty_symbol = true;
  }

  for (int sym = 1; sym <= 13; ++sym) {
    ok &= expect(seen_symbol[sym],
                 "SNAPSHOT_INFO present for symbol " + std::to_string(sym));
  }
  ok &= expect(found_sym3_with_bid,
               "SnapshotInfo for symbol 3: bid_count=1, ask_count=0");
  ok &= expect(found_sym7_with_ask,
               "SnapshotInfo for symbol 7: bid_count=0, ask_count=1");
  ok &= expect(found_empty_symbol,
               "at least one symbol has empty SnapshotInfo (no orders)");

  close(oe_fd);
  close(snap_fd);
  gateway.shutdown();
  return ok;
}

} // namespace

// =============================================================================
// Main
// =============================================================================

int main() {
  std::cout << "=== NDFEX Gateway UDP Integration Tests ===\n";

  bool ok = true;
  ok &= test_md_new_order_on_live_stream();
  ok &= test_md_trade_messages_on_fill();
  ok &= test_md_delete_on_cancel();
  ok &= test_md_seq_nums_monotonic();
  ok &= test_snapshot_reflects_resting_book();
  ok &= test_snapshot_no_orders_after_cancel();
  ok &= test_snapshot_covers_all_symbols();

  std::cout << "\n";
  if (!ok) {
    std::cerr << "One or more NDFEX UDP integration tests FAILED\n";
    return 1;
  }
  std::cout << "All NDFEX gateway UDP integration tests passed\n";
  return 0;
}
