#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ghostbook/ndfex/protocol.h"
#include "replay_engine.h"

// Aliases to avoid namespace ambiguity (ndfex and matching both define quantity_t)
using qty_t   = std::uint32_t;
using oid_t   = std::uint64_t;
using sym_t   = std::uint32_t;
using price_t = std::int32_t;

using ghostbook::ndfex::SIDE;
using ghostbook::ndfex::md::DeleteOrder;
using ghostbook::ndfex::md::Header;
using ghostbook::ndfex::md::MAGIC_NUMBER;
using ghostbook::ndfex::md::MSG_TYPE;
using ghostbook::ndfex::md::ModifyOrder;
using ghostbook::ndfex::md::NewOrder;
using ghostbook::ndfex::md::SNAPSHOT_MAGIC_NUMBER;
using ghostbook::ndfex::md::SnapshotInfo;
using ghostbook::ndfex::md::Trade;
using ghostbook::ndfex::md::TradeSummary;

using ghostbook::matching::CancelOrderCommand;
using ghostbook::matching::EventType;
using ghostbook::matching::ExecutionEvent;
using ghostbook::matching::NewOrderCommand;
using ghostbook::matching::OrderType;
using ghostbook::matching::Side;
using ghostbook::matching::TimeInForce;

using ghostbook::replay::ReplayEngine;

// =============================================================================
// Test infrastructure
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

// =============================================================================
// Binary fixture helpers
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
                ("re_test_" + std::to_string(++s_file_counter));
    std::ofstream out(path, std::ios::binary);
    for (const auto &msg : msgs) {
        out.write(reinterpret_cast<const char *>(msg.data()),
                  static_cast<std::streamsize>(msg.size()));
    }
    return path;
}

NewOrderCommand make_live(oid_t oid, sym_t sym, Side side, price_t price,
                           qty_t qty,
                           TimeInForce tif = TimeInForce::Day) {
    NewOrderCommand c{};
    c.client_order_id = oid;
    c.instrument_id   = sym;
    c.side            = side;
    c.order_type      = OrderType::Limit;
    c.tif             = tif;
    c.price_tick      = price;
    c.quantity        = qty;
    return c;
}

// =============================================================================
// Tests
// =============================================================================

bool test_snapshot_loading_restores_lob() {
    print_section("snapshot_loading_restores_lob");

    SnapshotInfo info{};
    info.symbol = 1; info.bid_count = 2; info.ask_count = 1;
    info.last_md_seq_num = 10;

    NewOrder no1{};
    no1.order_id = 101; no1.symbol = 1; no1.side = SIDE::BUY;
    no1.quantity = 5; no1.price = 100;
    NewOrder no2{};
    no2.order_id = 102; no2.symbol = 1; no2.side = SIDE::BUY;
    no2.quantity = 3; no2.price = 95;
    NewOrder no3{};
    no3.order_id = 201; no3.symbol = 1; no3.side = SIDE::SELL;
    no3.quantity = 7; no3.price = 105;

    auto snap_path = write_temp_file({
        make_md_msg(MSG_TYPE::SNAPSHOT_INFO, 0, &info, sizeof(info), true),
        make_md_msg(MSG_TYPE::NEW_ORDER,     0, &no1,  sizeof(no1),  true),
        make_md_msg(MSG_TYPE::NEW_ORDER,     0, &no2,  sizeof(no2),  true),
        make_md_msg(MSG_TYPE::NEW_ORDER,     0, &no3,  sizeof(no3),  true),
    });

    ReplayEngine engine;
    const bool loaded = engine.load_snapshot(snap_path);
    std::filesystem::remove(snap_path);

    bool ok = true;
    ok &= check(loaded, "snapshot loaded successfully");

    auto bids = engine.bid_levels(1);
    ok &= check_eq(bids.size(), std::size_t(2), "bid level count");
    if (bids.size() >= 2) {
        ok &= check_eq(bids[0].price,     price_t(100), "best bid price");
        ok &= check_eq(bids[0].total_qty, qty_t(5),     "best bid qty");
        ok &= check_eq(bids[1].price,     price_t(95),  "second bid price");
    }

    auto asks = engine.ask_levels(1);
    ok &= check_eq(asks.size(), std::size_t(1), "ask level count");
    if (!asks.empty()) {
        ok &= check_eq(asks[0].price,     price_t(105), "ask price");
        ok &= check_eq(asks[0].total_qty, qty_t(7),     "ask qty");
    }
    return ok;
}

bool test_feed_new_order_inserts_hist() {
    print_section("feed_new_order_inserts_hist");

    NewOrder no{};
    no.order_id = 101; no.symbol = 1; no.side = SIDE::BUY;
    no.quantity = 5; no.price = 100;

    auto feed_path = write_temp_file({
        make_md_msg(MSG_TYPE::NEW_ORDER, 1, &no, sizeof(no)),
    });

    ReplayEngine engine;
    engine.open_feed(feed_path);
    engine.advance();
    std::filesystem::remove(feed_path);

    auto bids = engine.bid_levels(1);
    bool ok = true;
    ok &= check_eq(bids.size(), std::size_t(1), "one bid level after feed new order");
    if (!bids.empty()) {
        ok &= check_eq(bids[0].total_qty,  qty_t(5),        "bid qty");
        ok &= check_eq(bids[0].live_count, std::size_t(0),  "no live entries");
    }
    return ok;
}

bool test_feed_delete_removes_hist() {
    print_section("feed_delete_removes_hist");

    SnapshotInfo info{};
    info.symbol = 1; info.bid_count = 1; info.ask_count = 0;
    info.last_md_seq_num = 0;
    NewOrder no{};
    no.order_id = 101; no.symbol = 1; no.side = SIDE::BUY;
    no.quantity = 5; no.price = 100;

    auto snap_path = write_temp_file({
        make_md_msg(MSG_TYPE::SNAPSHOT_INFO, 0, &info, sizeof(info), true),
        make_md_msg(MSG_TYPE::NEW_ORDER,     0, &no,   sizeof(no),   true),
    });

    DeleteOrder del{};
    del.order_id = 101;
    auto feed_path = write_temp_file({
        make_md_msg(MSG_TYPE::DELETE_ORDER, 1, &del, sizeof(del)),
    });

    ReplayEngine engine;
    engine.load_snapshot(snap_path);
    engine.open_feed(feed_path);
    std::filesystem::remove(snap_path);

    bool ok = true;
    ok &= check_eq(engine.bid_levels(1).size(), std::size_t(1), "one bid before delete");

    engine.advance();
    std::filesystem::remove(feed_path);

    ok &= check_eq(engine.bid_levels(1).size(), std::size_t(0), "zero bids after delete");
    return ok;
}

bool test_feed_modify_updates_qty() {
    print_section("feed_modify_updates_qty");

    SnapshotInfo info{};
    info.symbol = 1; info.bid_count = 0; info.ask_count = 1;
    info.last_md_seq_num = 0;
    NewOrder no{};
    no.order_id = 201; no.symbol = 1; no.side = SIDE::SELL;
    no.quantity = 10; no.price = 200;

    auto snap_path = write_temp_file({
        make_md_msg(MSG_TYPE::SNAPSHOT_INFO, 0, &info, sizeof(info), true),
        make_md_msg(MSG_TYPE::NEW_ORDER,     0, &no,   sizeof(no),   true),
    });

    ModifyOrder mod{};
    mod.order_id = 201; mod.side = SIDE::SELL; mod.quantity = 7; mod.price = 200;
    auto feed_path = write_temp_file({
        make_md_msg(MSG_TYPE::MODIFY_ORDER, 1, &mod, sizeof(mod)),
    });

    ReplayEngine engine;
    engine.load_snapshot(snap_path);
    engine.open_feed(feed_path);
    std::filesystem::remove(snap_path);
    engine.advance();
    std::filesystem::remove(feed_path);

    auto asks = engine.ask_levels(1);
    bool ok = true;
    ok &= check_eq(asks.size(), std::size_t(1), "ask level count");
    if (!asks.empty()) {
        ok &= check_eq(asks[0].total_qty, qty_t(7), "ask qty after modify");
    }
    return ok;
}

bool test_ghost_fill_live_ahead() {
    print_section("ghost_fill_live_ahead");

    ReplayEngine engine;

    // Live order submitted BEFORE hist — it has priority
    engine.submit(make_live(9001, 1, Side::Buy, 100, 3));
    engine.drain_events();

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

    engine.open_feed(feed_path);
    engine.advance();  // NEW_ORDER — hist inserted after live
    engine.advance();  // TRADE_SUMMARY
    engine.advance();  // TRADE — triggers ghost fill
    std::filesystem::remove(feed_path);

    auto events = engine.drain_events();
    std::vector<ExecutionEvent> fills;
    for (const auto &ev : events) {
        if (ev.type == EventType::Fill) fills.push_back(ev);
    }

    bool ok = true;
    ok &= check_eq(fills.size(), std::size_t(1), "one ghost fill");
    if (!fills.empty()) {
        ok &= check_eq(fills[0].order_id,           oid_t(9001), "fill is for live order");
        ok &= check_eq(fills[0].quantity,           qty_t(2),    "fill qty == trade qty");
        ok &= check_eq(fills[0].remaining_quantity, qty_t(1),    "remaining qty after fill");
    }
    return ok;
}

bool test_no_ghost_fill_live_behind() {
    print_section("no_ghost_fill_live_behind");

    // H1 from snapshot — has priority over later live order
    SnapshotInfo info{};
    info.symbol = 1; info.bid_count = 1; info.ask_count = 0;
    info.last_md_seq_num = 0;
    NewOrder no{};
    no.order_id = 501; no.symbol = 1; no.side = SIDE::BUY;
    no.quantity = 5; no.price = 100;

    auto snap_path = write_temp_file({
        make_md_msg(MSG_TYPE::SNAPSHOT_INFO, 0, &info, sizeof(info), true),
        make_md_msg(MSG_TYPE::NEW_ORDER,     0, &no,   sizeof(no),   true),
    });

    ReplayEngine engine;
    engine.load_snapshot(snap_path);
    std::filesystem::remove(snap_path);

    // Live submitted AFTER H1 — behind H1 in queue
    engine.submit(make_live(9001, 1, Side::Buy, 100, 3));
    engine.drain_events();

    TradeSummary ts{};
    ts.symbol = 1; ts.aggressor_side = SIDE::SELL;
    ts.total_quantity = 3; ts.last_price = 100;
    Trade tr{};
    tr.order_id = 501; tr.quantity = 3; tr.price = 100;

    auto feed_path = write_temp_file({
        make_md_msg(MSG_TYPE::TRADE_SUMMARY, 1, &ts, sizeof(ts)),
        make_md_msg(MSG_TYPE::TRADE,         2, &tr, sizeof(tr)),
    });

    engine.open_feed(feed_path);
    engine.advance();
    engine.advance();
    std::filesystem::remove(feed_path);

    auto events = engine.drain_events();
    std::vector<ExecutionEvent> fills;
    for (const auto &ev : events) {
        if (ev.type == EventType::Fill) fills.push_back(ev);
    }

    return check_eq(fills.size(), std::size_t(0), "no ghost fill when live is behind hist");
}

bool test_multiple_live_budget_sharing() {
    print_section("multiple_live_orders_ahead_budget_sharing");

    ReplayEngine engine;

    engine.submit(make_live(9001, 1, Side::Buy, 100, 2));
    engine.submit(make_live(9002, 1, Side::Buy, 100, 3));
    engine.drain_events();

    // Hist arrives after both live orders — behind both in queue
    NewOrder no{};
    no.order_id = 501; no.symbol = 1; no.side = SIDE::BUY;
    no.quantity = 5; no.price = 100;
    TradeSummary ts{};
    ts.symbol = 1; ts.aggressor_side = SIDE::SELL;
    ts.total_quantity = 4; ts.last_price = 100;
    Trade tr{};
    tr.order_id = 501; tr.quantity = 4; tr.price = 100;

    auto feed_path = write_temp_file({
        make_md_msg(MSG_TYPE::NEW_ORDER,     1, &no, sizeof(no)),
        make_md_msg(MSG_TYPE::TRADE_SUMMARY, 2, &ts, sizeof(ts)),
        make_md_msg(MSG_TYPE::TRADE,         3, &tr, sizeof(tr)),
    });

    engine.open_feed(feed_path);
    engine.advance();
    engine.advance();
    engine.advance();
    std::filesystem::remove(feed_path);

    auto events = engine.drain_events();
    std::vector<ExecutionEvent> fills;
    for (const auto &ev : events) {
        if (ev.type == EventType::Fill) fills.push_back(ev);
    }

    bool ok = true;
    ok &= check_eq(fills.size(), std::size_t(2), "two ghost fills");

    const ExecutionEvent *f1 = nullptr, *f2 = nullptr;
    for (const auto &f : fills) {
        if (f.order_id == 9001) f1 = &f;
        else if (f.order_id == 9002) f2 = &f;
    }
    ok &= check(f1 != nullptr, "fill for LIVE1 exists");
    ok &= check(f2 != nullptr, "fill for LIVE2 exists");
    if (f1) {
        ok &= check_eq(f1->quantity,           qty_t(2), "LIVE1 fill qty");
        ok &= check_eq(f1->remaining_quantity, qty_t(0), "LIVE1 fully filled");
    }
    if (f2) {
        // Budget = 4; LIVE1 took 2, budget = 2; LIVE2 has qty=3, fills min(3,2)=2
        ok &= check_eq(f2->quantity,           qty_t(2), "LIVE2 fill qty (budget remainder)");
        ok &= check_eq(f2->remaining_quantity, qty_t(1), "LIVE2 partial fill");
    }
    return ok;
}

bool test_cancel_before_trade_no_fill() {
    print_section("cancel_before_trade_no_fill");

    ReplayEngine engine;

    engine.submit(make_live(9001, 1, Side::Buy, 100, 5));
    engine.drain_events();

    CancelOrderCommand cancel{};
    cancel.client_order_id = 1;
    cancel.target_order_id = 9001;
    engine.cancel(cancel);
    engine.drain_events();

    NewOrder no{};
    no.order_id = 501; no.symbol = 1; no.side = SIDE::BUY;
    no.quantity = 3; no.price = 100;
    TradeSummary ts{};
    ts.symbol = 1; ts.aggressor_side = SIDE::SELL;
    ts.total_quantity = 3; ts.last_price = 100;
    Trade tr{};
    tr.order_id = 501; tr.quantity = 3; tr.price = 100;

    auto feed_path = write_temp_file({
        make_md_msg(MSG_TYPE::NEW_ORDER,     1, &no, sizeof(no)),
        make_md_msg(MSG_TYPE::TRADE_SUMMARY, 2, &ts, sizeof(ts)),
        make_md_msg(MSG_TYPE::TRADE,         3, &tr, sizeof(tr)),
    });

    engine.open_feed(feed_path);
    engine.advance();
    engine.advance();
    engine.advance();
    std::filesystem::remove(feed_path);

    auto events = engine.drain_events();
    std::vector<ExecutionEvent> fills;
    for (const auto &ev : events) {
        if (ev.type == EventType::Fill) fills.push_back(ev);
    }

    return check_eq(fills.size(), std::size_t(0), "no fill after cancel");
}

bool test_live_taker_gets_fill() {
    print_section("live_taker_gets_fill");

    SnapshotInfo info{};
    info.symbol = 1; info.bid_count = 0; info.ask_count = 1;
    info.last_md_seq_num = 0;
    NewOrder no{};
    no.order_id = 201; no.symbol = 1; no.side = SIDE::SELL;
    no.quantity = 5; no.price = 100;

    auto snap_path = write_temp_file({
        make_md_msg(MSG_TYPE::SNAPSHOT_INFO, 0, &info, sizeof(info), true),
        make_md_msg(MSG_TYPE::NEW_ORDER,     0, &no,   sizeof(no),   true),
    });

    ReplayEngine engine;
    engine.load_snapshot(snap_path);
    std::filesystem::remove(snap_path);

    auto asks_before = engine.ask_levels(1);

    // IOC BUY@105 qty=3 — crosses hist SELL@100
    engine.submit(make_live(9001, 1, Side::Buy, 105, 3, TimeInForce::IOC));

    auto events = engine.drain_events();
    std::vector<ExecutionEvent> fills;
    for (const auto &ev : events) {
        if (ev.type == EventType::Fill) fills.push_back(ev);
    }

    bool ok = true;
    ok &= check_eq(asks_before.size(), std::size_t(1), "hist ask present before cross");
    ok &= check(fills.size() >= 1, "taker fill received");
    if (!fills.empty()) {
        ok &= check_eq(fills[0].order_id, oid_t(9001), "fill is for live taker");
        ok &= check_eq(fills[0].quantity, qty_t(3),    "taker fill qty");
    }

    // Historical order NOT consumed (v1 simplification)
    auto asks_after = engine.ask_levels(1);
    ok &= check_eq(asks_after.size(), std::size_t(1), "hist ask still present (v1: not consumed)");
    if (!asks_after.empty()) {
        ok &= check_eq(asks_after[0].total_qty, qty_t(5), "hist qty unchanged");
    }
    return ok;
}

bool test_feed_exhaustion() {
    print_section("feed_exhaustion");

    NewOrder no1{};
    no1.order_id = 101; no1.symbol = 1; no1.side = SIDE::BUY;
    no1.quantity = 1; no1.price = 100;
    NewOrder no2{};
    no2.order_id = 102; no2.symbol = 1; no2.side = SIDE::BUY;
    no2.quantity = 1; no2.price = 101;

    auto feed_path = write_temp_file({
        make_md_msg(MSG_TYPE::NEW_ORDER, 1, &no1, sizeof(no1)),
        make_md_msg(MSG_TYPE::NEW_ORDER, 2, &no2, sizeof(no2)),
    });

    ReplayEngine engine;
    engine.open_feed(feed_path);

    bool ok = true;
    ok &= check(!engine.feed_exhausted(), "not exhausted initially");
    ok &= check(engine.advance(),         "first advance returns true");
    ok &= check(!engine.feed_exhausted(), "not exhausted after first advance");
    ok &= check(engine.advance(),         "second advance returns true");
    ok &= check(!engine.advance(),        "third advance returns false (EOF)");
    ok &= check(engine.feed_exhausted(),  "feed_exhausted() == true after EOF");

    std::filesystem::remove(feed_path);
    return ok;
}

bool test_snapshot_watermark_skips_stale() {
    print_section("snapshot_watermark_skips_stale_feed_messages");

    SnapshotInfo info{};
    info.symbol = 1; info.bid_count = 0; info.ask_count = 0;
    info.last_md_seq_num = 50;

    auto snap_path = write_temp_file({
        make_md_msg(MSG_TYPE::SNAPSHOT_INFO, 0, &info, sizeof(info), true),
    });

    ReplayEngine engine;
    engine.load_snapshot(snap_path);
    std::filesystem::remove(snap_path);

    NewOrder no1{};  // stale: seq=48
    no1.order_id = 101; no1.symbol = 1; no1.side = SIDE::BUY;
    no1.quantity = 5; no1.price = 100;
    NewOrder no2{};  // stale: seq=50
    no2.order_id = 102; no2.symbol = 1; no2.side = SIDE::BUY;
    no2.quantity = 3; no2.price = 95;
    NewOrder no3{};  // new: seq=51
    no3.order_id = 103; no3.symbol = 1; no3.side = SIDE::BUY;
    no3.quantity = 2; no3.price = 102;

    auto feed_path = write_temp_file({
        make_md_msg(MSG_TYPE::NEW_ORDER, 48, &no1, sizeof(no1)),
        make_md_msg(MSG_TYPE::NEW_ORDER, 50, &no2, sizeof(no2)),
        make_md_msg(MSG_TYPE::NEW_ORDER, 51, &no3, sizeof(no3)),
    });

    engine.open_feed(feed_path);
    engine.advance();  // should process only seq=51
    std::filesystem::remove(feed_path);

    auto bids = engine.bid_levels(1);
    bool ok = true;
    ok &= check_eq(bids.size(), std::size_t(1), "only one bid level (seq=51 processed)");
    if (!bids.empty()) {
        ok &= check_eq(bids[0].price, price_t(102), "price matches seq=51 order");
    }
    return ok;
}

bool test_live_order_survives_adjacent_hist_trade() {
    print_section("live_order_survives_adjacent_hist_trade");

    ReplayEngine engine;

    // LIVE1 at BUY@100
    engine.submit(make_live(9001, 1, Side::Buy, 100, 5));
    engine.drain_events();

    // H1 at BUY@100, H2 at BUY@105 — different price level from LIVE1
    NewOrder no1{};
    no1.order_id = 501; no1.symbol = 1; no1.side = SIDE::BUY;
    no1.quantity = 3; no1.price = 100;
    NewOrder no2{};
    no2.order_id = 502; no2.symbol = 1; no2.side = SIDE::BUY;
    no2.quantity = 4; no2.price = 105;

    // TRADE on H2 at @105 — LIVE1 is at @100, not in that queue
    TradeSummary ts{};
    ts.symbol = 1; ts.aggressor_side = SIDE::SELL;
    ts.total_quantity = 4; ts.last_price = 105;
    Trade tr{};
    tr.order_id = 502; tr.quantity = 4; tr.price = 105;

    auto feed_path = write_temp_file({
        make_md_msg(MSG_TYPE::NEW_ORDER,     1, &no1, sizeof(no1)),
        make_md_msg(MSG_TYPE::NEW_ORDER,     2, &no2, sizeof(no2)),
        make_md_msg(MSG_TYPE::TRADE_SUMMARY, 3, &ts,  sizeof(ts)),
        make_md_msg(MSG_TYPE::TRADE,         4, &tr,  sizeof(tr)),
    });

    engine.open_feed(feed_path);
    engine.advance();
    engine.advance();
    engine.advance();
    engine.advance();
    std::filesystem::remove(feed_path);

    auto events = engine.drain_events();
    std::vector<ExecutionEvent> fills;
    for (const auto &ev : events) {
        if (ev.type == EventType::Fill) fills.push_back(ev);
    }

    bool ok = true;
    ok &= check_eq(fills.size(), std::size_t(0),
                   "no ghost fill (LIVE1 at @100, TRADE at @105)");

    // LIVE1 still resting at @100
    bool found_live = false;
    for (const auto &lv : engine.bid_levels(1)) {
        if (lv.price == 100 && lv.live_count > 0) found_live = true;
    }
    ok &= check(found_live, "LIVE1 still resting at @100");
    return ok;
}

} // namespace

int main() {
    std::cout << "=== Replay Engine Unit Tests ===\n";

    bool all_ok = true;
    all_ok &= test_snapshot_loading_restores_lob();
    all_ok &= test_feed_new_order_inserts_hist();
    all_ok &= test_feed_delete_removes_hist();
    all_ok &= test_feed_modify_updates_qty();
    all_ok &= test_ghost_fill_live_ahead();
    all_ok &= test_no_ghost_fill_live_behind();
    all_ok &= test_multiple_live_budget_sharing();
    all_ok &= test_cancel_before_trade_no_fill();
    all_ok &= test_live_taker_gets_fill();
    all_ok &= test_feed_exhaustion();
    all_ok &= test_snapshot_watermark_skips_stale();
    all_ok &= test_live_order_survives_adjacent_hist_trade();

    (void)all_ok;
    std::cout << "\n=== Results ===\n";
    std::cout << "PASS: " << g_pass_count << "\n";
    std::cout << "FAIL: " << g_fail_count << "\n";
    std::cout << "RESULT: " << (g_fail_count == 0 ? "PASS" : "FAIL") << "\n";
    return g_fail_count == 0 ? 0 : 1;
}
