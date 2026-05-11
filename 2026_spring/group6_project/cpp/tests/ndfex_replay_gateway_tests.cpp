#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ghostbook/ndfex/protocol.h"
#include "ndfex_gateway.h"
#include "replay_engine.h"

using qty_t   = std::uint32_t;
using oid_t   = std::uint64_t;
using sym_t   = std::uint32_t;
using price_t = std::int32_t;

using ghostbook::gateway::NdfexGateway;
using ghostbook::gateway::NdfexMarketDataCallback;
using ghostbook::gateway::session_id_t;
using ghostbook::ndfex::SIDE;
using ghostbook::ndfex::md::DeleteOrder;
using ghostbook::ndfex::md::Header;
using ghostbook::ndfex::md::MAGIC_NUMBER;
using ghostbook::ndfex::md::MSG_TYPE;
using ghostbook::ndfex::md::Message;
using ghostbook::ndfex::md::ModifyOrder;
using ghostbook::ndfex::md::NewOrder;
using ghostbook::ndfex::md::SNAPSHOT_MAGIC_NUMBER;
using ghostbook::ndfex::md::SnapshotInfo;
using ghostbook::ndfex::md::Trade;
using ghostbook::ndfex::md::TradeSummary;
using ghostbook::ndfex::oe::ORDER_FLAGS;
using ghostbook::ndfex::oe::Request;
using ghostbook::ndfex::oe::RequestHeader;
using ghostbook::ndfex::oe::Response;
using ghostbook::replay::ReplayEngine;

// =============================================================================
// Test infrastructure (identical to ndfex_gateway_tests.cpp conventions)
// =============================================================================

namespace {

int g_pass_count = 0;
int g_fail_count = 0;

void print_section(const char *name) {
    std::cout << "\n[TEST] " << name << "\n";
}

bool check(bool condition, const std::string &desc) {
    if (condition) {
        std::cout << "  PASS  " << desc << "\n";
        ++g_pass_count;
    } else {
        std::cerr << "  FAIL  " << desc << "\n";
        ++g_fail_count;
    }
    return condition;
}

template <typename T>
bool check_eq(T actual, T expected, const std::string &label) {
    if (actual == expected) {
        std::cout << "  PASS  " << label << " == "
                  << static_cast<std::uint64_t>(actual) << "\n";
        ++g_pass_count;
        return true;
    }
    std::cerr << "  FAIL  " << label << ": expected "
              << static_cast<std::uint64_t>(expected) << ", got "
              << static_cast<std::uint64_t>(actual) << "\n";
    ++g_fail_count;
    return false;
}

struct CapturedResponse {
    session_id_t session_id{};
    Response response{};
};

struct CapturedMd {
    Message message{};
};

// =============================================================================
// Binary fixture helpers (duplicated from replay_engine_tests for self-containment)
// =============================================================================

std::vector<std::uint8_t> make_md_msg(MSG_TYPE type, std::uint32_t seq,
                                       const void *body,
                                       std::uint16_t body_size,
                                       bool snapshot = false) {
    Header hdr{};
    hdr.magic_number = snapshot ? SNAPSHOT_MAGIC_NUMBER : MAGIC_NUMBER;
    hdr.length = static_cast<std::uint16_t>(sizeof(Header) + body_size);
    hdr.seq_num = seq;
    hdr.timestamp = 0;
    hdr.msg_type = type;

    std::vector<std::uint8_t> result(sizeof(Header) + body_size);
    std::memcpy(result.data(), &hdr, sizeof(Header));
    if (body_size > 0 && body) {
        std::memcpy(result.data() + sizeof(Header), body, body_size);
    }
    return result;
}

static int s_file_counter = 0;

std::filesystem::path write_temp_file(
    const std::vector<std::vector<std::uint8_t>> &msgs) {
    auto path = std::filesystem::temp_directory_path() /
                ("re_gw_test_" + std::to_string(++s_file_counter));
    std::ofstream out(path, std::ios::binary);
    for (const auto &msg : msgs) {
        out.write(reinterpret_cast<const char *>(msg.data()),
                  static_cast<std::streamsize>(msg.size()));
    }
    return path;
}

// =============================================================================
// OE request builders (matching ndfex_gateway_tests.cpp conventions)
// =============================================================================

RequestHeader make_req_header(ghostbook::ndfex::oe::MSG_TYPE type,
                               ghostbook::ndfex::oe::client_id_t client_id,
                               ghostbook::ndfex::seq_num_t seq,
                               ghostbook::ndfex::oe::session_id_t sid = 0) {
    RequestHeader hdr{};
    hdr.length     = 0;
    hdr.msg_type   = type;
    hdr.version    = ghostbook::ndfex::oe::OE_PROTOCOL_VERSION;
    hdr.seq_num    = seq;
    hdr.client_id  = client_id;
    hdr.session_id = sid;
    return hdr;
}

Request make_new_order(ghostbook::ndfex::oe::client_id_t cid,
                       ghostbook::ndfex::seq_num_t seq,
                       ghostbook::ndfex::order_id_t oid,
                       sym_t symbol, SIDE side, qty_t qty, price_t price,
                       ORDER_FLAGS flags = ORDER_FLAGS::NONE) {
    Request req{};
    req.header = make_req_header(ghostbook::ndfex::oe::MSG_TYPE::NEW_ORDER, cid, seq);
    req.header.length = static_cast<std::uint16_t>(
        sizeof(RequestHeader) + sizeof(ghostbook::ndfex::oe::NewOrderRequest));
    req.body.new_order_request.order_id = oid;
    req.body.new_order_request.symbol   = symbol;
    req.body.new_order_request.side     = side;
    req.body.new_order_request.quantity = qty;
    req.body.new_order_request.price    = price;
    req.body.new_order_request.flags    = flags;
    return req;
}

Request make_delete_order(ghostbook::ndfex::oe::client_id_t cid,
                          ghostbook::ndfex::seq_num_t seq,
                          ghostbook::ndfex::order_id_t oid) {
    Request req{};
    req.header = make_req_header(ghostbook::ndfex::oe::MSG_TYPE::DELETE_ORDER, cid, seq);
    req.header.length = static_cast<std::uint16_t>(
        sizeof(RequestHeader) + sizeof(ghostbook::ndfex::oe::DeleteOrderRequest));
    req.body.delete_order_request.order_id = oid;
    return req;
}

// =============================================================================
// Tests
// =============================================================================

// Test 1: Live order submitted through replay gateway produces ACK.
bool test_replay_gateway_live_order_acked() {
    print_section("replay_gateway_live_order_acked");

    // Empty feed — no historical data, just test the OE path.
    auto feed_path = write_temp_file({});

    ReplayEngine engine;
    engine.open_feed(feed_path);
    std::filesystem::remove(feed_path);

    NdfexGateway gw(std::move(engine));
    const session_id_t sid = gw.create_test_session(1);

    std::vector<CapturedResponse> responses;
    gw.set_ndfex_response_callback(
        [&](session_id_t s, const Response &r) { responses.push_back({s, r}); });

    const auto req = make_new_order(1, 1, 1001, 1, SIDE::BUY, 10, 100);
    gw.process_ndfex_request(sid, req);

    bool ok = true;
    ok &= check(!responses.empty(), "new order produces a response");
    if (!responses.empty()) {
        ok &= check_eq(
            static_cast<int>(responses[0].response.header.msg_type),
            static_cast<int>(ghostbook::ndfex::oe::MSG_TYPE::ACK),
            "response is ACK");
        ok &= check_eq(
            responses[0].response.body.order_ack_response.order_id,
            ghostbook::ndfex::order_id_t(1001),
            "ACK echoes client order_id");
    }
    return ok;
}

// Test 2: Ghost fill from replay feed is delivered as FILL response.
bool test_replay_gateway_ghost_fill_delivered() {
    print_section("replay_gateway_ghost_fill_delivered");

    // Feed: historical order H1 BUY@100, then TRADE(H1, qty=2).
    // Live order LIVE1 BUY@100 is submitted BEFORE H1 is fed → LIVE1 has priority.
    NewOrder no{};
    no.order_id = 501; no.symbol = 1; no.side = SIDE::BUY;
    no.quantity = 5; no.price = 100;
    TradeSummary ts{};
    ts.symbol = 1; ts.aggressor_side = SIDE::SELL;
    ts.total_quantity = 2; ts.last_price = 100;
    Trade tr{};
    tr.order_id = 501; tr.quantity = 2; tr.price = 100;

    auto feed_path = write_temp_file({
        make_md_msg(MSG_TYPE::NEW_ORDER,     1, &no, sizeof(no)),
        make_md_msg(MSG_TYPE::TRADE_SUMMARY, 2, &ts, sizeof(ts)),
        make_md_msg(MSG_TYPE::TRADE,         3, &tr, sizeof(tr)),
    });

    ReplayEngine engine;
    engine.open_feed(feed_path);
    std::filesystem::remove(feed_path);

    NdfexGateway gw(std::move(engine));
    const session_id_t sid = gw.create_test_session(1);

    std::vector<CapturedResponse> responses;
    gw.set_ndfex_response_callback(
        [&](session_id_t s, const Response &r) { responses.push_back({s, r}); });

    // Submit LIVE1 BEFORE advancing feed (before H1 is inserted into shadow LOB)
    const auto req = make_new_order(1, 1, 9001, 1, SIDE::BUY, 3, 100);
    gw.process_ndfex_request(sid, req);
    responses.clear();  // discard ACK

    // Advance feed: NEW_ORDER(H1), TRADE_SUMMARY, TRADE → ghost fill on LIVE1
    gw.advance_feed();  // NEW_ORDER — H1 inserted after LIVE1
    gw.advance_feed();  // TRADE_SUMMARY
    gw.advance_feed();  // TRADE — ghost fill emitted and flushed

    std::vector<CapturedResponse> fills;
    for (const auto &r : responses) {
        if (r.response.header.msg_type == ghostbook::ndfex::oe::MSG_TYPE::FILL) {
            fills.push_back(r);
        }
    }

    bool ok = true;
    ok &= check(!fills.empty(), "ghost fill delivered as FILL response");
    if (!fills.empty()) {
        ok &= check_eq(
            fills[0].response.body.order_fill_response.order_id,
            ghostbook::ndfex::order_id_t(9001),
            "FILL is for live order 9001");
        ok &= check_eq(
            fills[0].response.body.order_fill_response.quantity,
            qty_t(2),
            "FILL qty matches trade qty");
    }
    return ok;
}

// Test 3: Live order submission fires md_callback with NEW_ORDER message.
bool test_replay_gateway_live_order_md_published() {
    print_section("replay_gateway_live_order_md_published");

    auto feed_path = write_temp_file({});

    ReplayEngine engine;
    engine.open_feed(feed_path);
    std::filesystem::remove(feed_path);

    NdfexGateway gw(std::move(engine));
    const session_id_t sid = gw.create_test_session(1);

    std::vector<CapturedMd> md_msgs;
    gw.set_market_data_callback(
        [&](const Message &m) { md_msgs.push_back({m}); });

    const auto req = make_new_order(1, 1, 2001, 1, SIDE::SELL, 5, 200);
    gw.process_ndfex_request(sid, req);

    bool ok = true;
    ok &= check(!md_msgs.empty(), "md_callback fired after live order");
    if (!md_msgs.empty()) {
        ok &= check_eq(
            static_cast<int>(md_msgs[0].message.header.msg_type),
            static_cast<int>(ghostbook::ndfex::md::MSG_TYPE::NEW_ORDER),
            "MD message is NEW_ORDER");
        ok &= check_eq(
            md_msgs[0].message.body.new_order.order_id,
            ghostbook::ndfex::order_id_t(2001),
            "MD NEW_ORDER has correct order_id");
    }
    return ok;
}

// Test 4: Cancel produces CLOSE response and subsequent advance yields no fill.
bool test_replay_gateway_cancel_produces_close() {
    print_section("replay_gateway_cancel_produces_close");

    // Feed: H1, then TRADE on H1.
    NewOrder no{};
    no.order_id = 601; no.symbol = 1; no.side = SIDE::BUY;
    no.quantity = 3; no.price = 100;
    TradeSummary ts{};
    ts.symbol = 1; ts.aggressor_side = SIDE::SELL;
    ts.total_quantity = 3; ts.last_price = 100;
    Trade tr{};
    tr.order_id = 601; tr.quantity = 3; tr.price = 100;

    auto feed_path = write_temp_file({
        make_md_msg(MSG_TYPE::NEW_ORDER,     1, &no, sizeof(no)),
        make_md_msg(MSG_TYPE::TRADE_SUMMARY, 2, &ts, sizeof(ts)),
        make_md_msg(MSG_TYPE::TRADE,         3, &tr, sizeof(tr)),
    });

    ReplayEngine engine;
    engine.open_feed(feed_path);
    std::filesystem::remove(feed_path);

    NdfexGateway gw(std::move(engine));
    const session_id_t sid = gw.create_test_session(1);

    std::vector<CapturedResponse> responses;
    gw.set_ndfex_response_callback(
        [&](session_id_t s, const Response &r) { responses.push_back({s, r}); });

    // Submit live order
    const auto new_req = make_new_order(1, 1, 9001, 1, SIDE::BUY, 5, 100);
    gw.process_ndfex_request(sid, new_req);

    // Cancel it immediately
    const auto del_req = make_delete_order(1, 2, 9001);
    gw.process_ndfex_request(sid, del_req);

    // Find CLOSE response
    bool got_close = false;
    for (const auto &r : responses) {
        if (r.response.header.msg_type == ghostbook::ndfex::oe::MSG_TYPE::CLOSE) {
            got_close = true;
        }
    }

    bool ok = true;
    ok &= check(got_close, "cancel produces CLOSE response");

    // Advance feed past TRADE — should produce no FILL (order was cancelled)
    responses.clear();
    gw.advance_feed();  // NEW_ORDER
    gw.advance_feed();  // TRADE_SUMMARY
    gw.advance_feed();  // TRADE

    bool got_fill = false;
    for (const auto &r : responses) {
        if (r.response.header.msg_type == ghostbook::ndfex::oe::MSG_TYPE::FILL) {
            got_fill = true;
        }
    }
    ok &= check(!got_fill, "no FILL after cancel when TRADE arrives");
    return ok;
}

} // namespace

int main() {
    std::cout << "=== Ndfex Replay Gateway Integration Tests ===\n";

    bool all_ok = true;
    all_ok &= test_replay_gateway_live_order_acked();
    all_ok &= test_replay_gateway_ghost_fill_delivered();
    all_ok &= test_replay_gateway_live_order_md_published();
    all_ok &= test_replay_gateway_cancel_produces_close();

    (void)all_ok;
    std::cout << "\n=== Results ===\n";
    std::cout << "PASS: " << g_pass_count << "\n";
    std::cout << "FAIL: " << g_fail_count << "\n";
    std::cout << "RESULT: " << (g_fail_count == 0 ? "PASS" : "FAIL") << "\n";
    return g_fail_count == 0 ? 0 : 1;
}
