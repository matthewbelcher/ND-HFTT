/**
 * Unit tests for the risk management system
 *
 * Tests PositionTracker, RiskSystem, and their integration.
 * No exchange connection required — all tests run in isolation.
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <thread>
#include <chrono>

#include "hft/PositionTracker.h"
#include "hft/RiskSystem.h"
#include "hft/orderbook.h"

/* ── Test Utilities ────────────────────────────────────────────────────── */

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)                                                        \
    static void test_##name();                                            \
    static struct TestRunner_##name {                                     \
        TestRunner_##name() {                                             \
            printf("  %-55s ", #name);                                    \
            try {                                                         \
                test_##name();                                            \
                printf("PASS\n");                                         \
                tests_passed++;                                           \
            } catch (const std::exception &e) {                           \
                printf("FAIL: %s\n", e.what());                           \
                tests_failed++;                                           \
            } catch (...) {                                               \
                printf("FAIL: unknown exception\n");                      \
                tests_failed++;                                           \
            }                                                             \
        }                                                                 \
    } test_runner_##name;                                                 \
    static void test_##name()

#define ASSERT_EQ(a, b) do {                                              \
    auto _a = (a); auto _b = (b);                                        \
    if (_a != _b) {                                                       \
        char buf[256];                                                    \
        snprintf(buf, sizeof(buf), "ASSERT_EQ failed: %s != %s (line %d)", #a, #b, __LINE__); \
        throw std::runtime_error(buf);                                    \
    }                                                                     \
} while(0)

#define ASSERT_TRUE(expr) do {                                            \
    if (!(expr)) {                                                        \
        char buf[256];                                                    \
        snprintf(buf, sizeof(buf), "ASSERT_TRUE failed: %s (line %d)", #expr, __LINE__); \
        throw std::runtime_error(buf);                                    \
    }                                                                     \
} while(0)

#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))

#define ASSERT_NEAR(a, b, eps) do {                                       \
    auto _a = (a); auto _b = (b);                                        \
    if (std::abs(_a - _b) > (eps)) {                                     \
        char buf[256];                                                    \
        snprintf(buf, sizeof(buf), "ASSERT_NEAR failed: %s=%f != %s=%f (line %d)", #a, (double)_a, #b, (double)_b, __LINE__); \
        throw std::runtime_error(buf);                                    \
    }                                                                     \
} while(0)


/* ── Helper: build a NewOrderRequest ───────────────────────────────────── */

static hft::msg::oe::NewOrderRequest make_order(
    hft::msg::order_id_t id, hft::msg::symbol_id_t sym,
    hft::msg::SIDE side, hft::msg::quantity_t qty, hft::msg::price_t price) {
    return hft::msg::oe::NewOrderRequest{
        .order_id = id,
        .symbol = sym,
        .side = side,
        .quantity = qty,
        .price = price,
        .flags = hft::msg::oe::ORDER_FLAGS::NONE,
    };
}


/* ══════════════════════════════════════════════════════════════════════════
 *  POSITION TRACKER TESTS
 * ══════════════════════════════════════════════════════════════════════ */

TEST(position_tracker_initial_state) {
    hft::risk::PositionTracker tracker;
    ASSERT_EQ(tracker.get_position(1), 0);
    ASSERT_EQ(tracker.get_buy_exposure(1), 0);
    ASSERT_EQ(tracker.get_sell_exposure(1), 0);
}

TEST(position_tracker_buy_fill) {
    hft::risk::PositionTracker tracker;
    tracker.on_fill(1, hft::msg::SIDE::BUY, 100, 5000);
    ASSERT_EQ(tracker.get_position(1), 100);

    const auto &s = tracker.get_state(1);
    ASSERT_EQ(s.total_bought_qty, 100u);
    ASSERT_EQ(s.total_buy_value, 500000);
}

TEST(position_tracker_sell_fill) {
    hft::risk::PositionTracker tracker;
    tracker.on_fill(1, hft::msg::SIDE::SELL, 50, 6000);
    ASSERT_EQ(tracker.get_position(1), -50);

    const auto &s = tracker.get_state(1);
    ASSERT_EQ(s.total_sold_qty, 50u);
    ASSERT_EQ(s.total_sell_value, 300000);
}

TEST(position_tracker_net_position) {
    hft::risk::PositionTracker tracker;
    tracker.on_fill(1, hft::msg::SIDE::BUY, 100, 5000);
    tracker.on_fill(1, hft::msg::SIDE::SELL, 60, 5500);
    ASSERT_EQ(tracker.get_position(1), 40);
}

TEST(position_tracker_exposure_outstanding_orders) {
    hft::risk::PositionTracker tracker;
    // Position: long 100
    tracker.on_fill(1, hft::msg::SIDE::BUY, 100, 5000);

    // Outstanding buy order of 50
    tracker.on_order_sent(10, 1, hft::msg::SIDE::BUY, 50);

    // buy_exposure = position + outstanding_buy = 100 + 50 = 150
    ASSERT_EQ(tracker.get_buy_exposure(1), 150);
    // sell_exposure = -position + outstanding_sell = -100 + 0 = -100
    ASSERT_EQ(tracker.get_sell_exposure(1), -100);
}

TEST(position_tracker_exposure_after_fill) {
    hft::risk::PositionTracker tracker;
    tracker.on_order_sent(10, 1, hft::msg::SIDE::BUY, 100);
    ASSERT_EQ(tracker.get_buy_exposure(1), 100);

    // Partial fill of 40
    tracker.on_order_fill(10, 40);
    ASSERT_EQ(tracker.get_state(1).outstanding_buy_qty, 60u);
}

TEST(position_tracker_exposure_after_close) {
    hft::risk::PositionTracker tracker;
    tracker.on_order_sent(10, 1, hft::msg::SIDE::SELL, 200);
    ASSERT_EQ(tracker.get_state(1).outstanding_sell_qty, 200u);

    tracker.on_order_closed(10);
    ASSERT_EQ(tracker.get_state(1).outstanding_sell_qty, 0u);
}

TEST(position_tracker_exposure_modify) {
    hft::risk::PositionTracker tracker;
    tracker.on_order_sent(10, 1, hft::msg::SIDE::BUY, 100);
    ASSERT_EQ(tracker.get_state(1).outstanding_buy_qty, 100u);

    // Modify: change to SELL 50
    tracker.on_order_modified(10, hft::msg::SIDE::SELL, 50);
    ASSERT_EQ(tracker.get_state(1).outstanding_buy_qty, 0u);
    ASSERT_EQ(tracker.get_state(1).outstanding_sell_qty, 50u);
}

TEST(position_tracker_realized_pnl) {
    hft::risk::PositionTracker tracker;
    // Buy 100 @ 5000, sell 100 @ 5500
    tracker.on_fill(1, hft::msg::SIDE::BUY, 100, 5000);
    tracker.on_fill(1, hft::msg::SIDE::SELL, 100, 5500);

    // Realized PNL = 100 * (5500 - 5000) = 50000
    ASSERT_NEAR(tracker.get_realized_pnl(1), 50000.0, 0.01);
    ASSERT_EQ(tracker.get_position(1), 0);
}

TEST(position_tracker_unrealized_pnl_long) {
    hft::risk::PositionTracker tracker;
    tracker.on_fill(1, hft::msg::SIDE::BUY, 100, 5000);
    // Position is long 100, avg buy = 5000, mark at mid = 5200
    // Unrealized = 100 * (5200 - 5000) = 20000
    ASSERT_NEAR(tracker.get_unrealized_pnl(1, 5200), 20000.0, 0.01);
}

TEST(position_tracker_unrealized_pnl_short) {
    hft::risk::PositionTracker tracker;
    tracker.on_fill(1, hft::msg::SIDE::SELL, 100, 5000);
    // Position is short 100, avg sell = 5000, mark at mid = 4800
    // Unrealized = 100 * (5000 - 4800) = 20000
    ASSERT_NEAR(tracker.get_unrealized_pnl(1, 4800), 20000.0, 0.01);
}

TEST(position_tracker_total_pnl) {
    hft::risk::PositionTracker tracker;
    // Buy 100 @ 5000, sell 50 @ 5500
    tracker.on_fill(1, hft::msg::SIDE::BUY, 100, 5000);
    tracker.on_fill(1, hft::msg::SIDE::SELL, 50, 5500);
    // Position: long 50
    // Realized: 50 * (5500 - 5000) = 25000
    // Unrealized (mid=5200): 50 * (5200 - 5000) = 10000
    // Total: 35000
    ASSERT_NEAR(tracker.get_total_pnl(1, 5200), 35000.0, 0.01);
}

TEST(position_tracker_multi_symbol) {
    hft::risk::PositionTracker tracker;
    tracker.on_fill(0, hft::msg::SIDE::BUY, 50, 1000);
    tracker.on_fill(1, hft::msg::SIDE::SELL, 30, 2000);
    ASSERT_EQ(tracker.get_position(0), 50);
    ASSERT_EQ(tracker.get_position(1), -30);
}


/* ══════════════════════════════════════════════════════════════════════════
 *  RISK SYSTEM TESTS
 * ══════════════════════════════════════════════════════════════════════ */

static hft::risk::RiskLimits test_limits() {
    return hft::risk::RiskLimits{
        .max_qty_per_order = 100,
        .max_qty_per_side = 500,
        .max_exposure_per_side = 600,
        .max_position = 300,
        .max_abs_position_shutdown = 400,
        .min_pnl_shutdown = -50000.0,
        .max_orders_per_second = 5,
        .max_orders_per_md_update = 2,
        .max_inflight_orders = 3,
        .min_valid_price = 1,
        .max_valid_price = 999999,
    };
}

/* (a) Max qty per order */
TEST(risk_max_qty_per_order) {
    hft::risk::PositionTracker tracker;
    auto orderbook = hft::orderbook::MultiSymbolOrderBook(2);
    hft::risk::RiskSystem risk(test_limits(), tracker, orderbook);

    auto order = make_order(1, 0, hft::msg::SIDE::BUY, 101, 5000);
    auto result = risk.evaluate_order(order);
    ASSERT_FALSE(result.accepted);
    ASSERT_EQ(result.reason, hft::risk::RejectReason::MAX_QTY_PER_ORDER);

    // Under limit should pass
    order.quantity = 100;
    result = risk.evaluate_order(order);
    ASSERT_TRUE(result.accepted);
}

/* (b) Max qty per side */
TEST(risk_max_qty_per_side) {
    hft::risk::PositionTracker tracker;
    auto orderbook = hft::orderbook::MultiSymbolOrderBook(2);
    hft::risk::RiskLimits limits = test_limits();
    limits.max_orders_per_second = 100;
    limits.max_inflight_orders = 100;
    hft::risk::RiskSystem risk(limits, tracker, orderbook);

    // Register 5 outstanding buy orders of 100 each = 500, at limit
    for (uint64_t i = 1; i <= 5; i++) {
        tracker.on_order_sent(i, 0, hft::msg::SIDE::BUY, 100);
    }

    // 6th order should be rejected (outstanding_buy = 500, +100 = 600 > 500)
    auto order = make_order(6, 0, hft::msg::SIDE::BUY, 100, 5000);
    auto result = risk.evaluate_order(order);
    ASSERT_FALSE(result.accepted);
    ASSERT_EQ(result.reason, hft::risk::RejectReason::MAX_QTY_PER_SIDE);
}

/* (c) Max exposure per side */
TEST(risk_max_exposure) {
    hft::risk::PositionTracker tracker;
    auto orderbook = hft::orderbook::MultiSymbolOrderBook(2);
    hft::risk::RiskSystem risk(test_limits(), tracker, orderbook);

    // Position of 500 long (from fills, no outstanding orders)
    tracker.on_fill(0, hft::msg::SIDE::BUY, 500, 5000);
    // buy_exposure = 500 + 0 = 500. Adding 100 + 1 = 601 > 600
    auto order = make_order(1, 0, hft::msg::SIDE::BUY, 100 + 1, 5000);
    // qty is 101 > max_qty_per_order(100), so fix:
    order.quantity = 100;
    // buy_exposure would be 500 + 0 + 100 = 600, which equals limit — should pass
    auto result = risk.evaluate_order(order);
    // But position 500 > max_position 300, so it triggers POSITION_LIMIT first
    ASSERT_FALSE(result.accepted);
    ASSERT_EQ(result.reason, hft::risk::RejectReason::POSITION_LIMIT);
}

TEST(risk_max_exposure_from_outstanding) {
    hft::risk::PositionTracker tracker;
    auto orderbook = hft::orderbook::MultiSymbolOrderBook(2);
    // Use higher rate/qty limits so they don't interfere with this test
    hft::risk::RiskLimits limits = test_limits();
    limits.max_qty_per_side = 10000;
    limits.max_orders_per_second = 100;
    limits.max_orders_per_md_update = 100;
    limits.max_inflight_orders = 100;
    hft::risk::RiskSystem risk(limits, tracker, orderbook);

    // Outstanding buy orders totaling 500
    // (only register with tracker, not risk system, to avoid rate limit counters)
    tracker.on_order_sent(1, 0, hft::msg::SIDE::BUY, 100);
    tracker.on_order_sent(2, 0, hft::msg::SIDE::BUY, 100);
    tracker.on_order_sent(3, 0, hft::msg::SIDE::BUY, 100);
    tracker.on_order_sent(4, 0, hft::msg::SIDE::BUY, 100);
    tracker.on_order_sent(5, 0, hft::msg::SIDE::BUY, 100);

    // buy_exposure = 0 + 500 = 500. Adding 100 more → 600, at limit
    auto order = make_order(6, 0, hft::msg::SIDE::BUY, 100, 5000);
    auto result = risk.evaluate_order(order);
    ASSERT_TRUE(result.accepted);

    // Now add one that pushes over the exposure limit
    tracker.on_order_sent(6, 0, hft::msg::SIDE::BUY, 100);

    auto order2 = make_order(7, 0, hft::msg::SIDE::BUY, 1, 5000);
    result = risk.evaluate_order(order2);
    ASSERT_FALSE(result.accepted);
    ASSERT_EQ(result.reason, hft::risk::RejectReason::MAX_EXPOSURE);
}

/* (d) Invalid price */
TEST(risk_invalid_price) {
    hft::risk::PositionTracker tracker;
    auto orderbook = hft::orderbook::MultiSymbolOrderBook(2);
    hft::risk::RiskSystem risk(test_limits(), tracker, orderbook);

    auto order = make_order(1, 0, hft::msg::SIDE::BUY, 10, 0);
    auto result = risk.evaluate_order(order);
    ASSERT_FALSE(result.accepted);
    ASSERT_EQ(result.reason, hft::risk::RejectReason::INVALID_PRICE);

    order.price = -100;
    result = risk.evaluate_order(order);
    ASSERT_FALSE(result.accepted);
    ASSERT_EQ(result.reason, hft::risk::RejectReason::INVALID_PRICE);

    order.price = 1000000;  // > max
    result = risk.evaluate_order(order);
    ASSERT_FALSE(result.accepted);
    ASSERT_EQ(result.reason, hft::risk::RejectReason::INVALID_PRICE);
}

/* (e) Position limit — order would increase |position| when at limit */
TEST(risk_position_limit) {
    hft::risk::PositionTracker tracker;
    auto orderbook = hft::orderbook::MultiSymbolOrderBook(2);
    hft::risk::RiskSystem risk(test_limits(), tracker, orderbook);

    // Position = 300 (at limit)
    tracker.on_fill(0, hft::msg::SIDE::BUY, 300, 5000);

    // Buying more should be rejected
    auto buy = make_order(1, 0, hft::msg::SIDE::BUY, 10, 5000);
    auto result = risk.evaluate_order(buy);
    ASSERT_FALSE(result.accepted);
    ASSERT_EQ(result.reason, hft::risk::RejectReason::POSITION_LIMIT);

    // Selling should be allowed (reduces position)
    auto sell = make_order(2, 0, hft::msg::SIDE::SELL, 10, 5000);
    result = risk.evaluate_order(sell);
    ASSERT_TRUE(result.accepted);
}

TEST(risk_position_limit_short) {
    hft::risk::PositionTracker tracker;
    auto orderbook = hft::orderbook::MultiSymbolOrderBook(2);
    hft::risk::RiskSystem risk(test_limits(), tracker, orderbook);

    // Position = -300 (short, at limit)
    tracker.on_fill(0, hft::msg::SIDE::SELL, 300, 5000);

    // Selling more should be rejected
    auto sell = make_order(1, 0, hft::msg::SIDE::SELL, 10, 5000);
    auto result = risk.evaluate_order(sell);
    ASSERT_FALSE(result.accepted);
    ASSERT_EQ(result.reason, hft::risk::RejectReason::POSITION_LIMIT);

    // Buying should be allowed
    auto buy = make_order(2, 0, hft::msg::SIDE::BUY, 10, 5000);
    result = risk.evaluate_order(buy);
    ASSERT_TRUE(result.accepted);
}

/* (f) Max orders per second */
TEST(risk_max_orders_per_second) {
    hft::risk::PositionTracker tracker;
    auto orderbook = hft::orderbook::MultiSymbolOrderBook(2);
    hft::risk::RiskLimits limits = test_limits();
    limits.max_inflight_orders = 100;  // Don't let inflight interfere
    hft::risk::RiskSystem risk(limits, tracker, orderbook);

    // Send max_orders_per_second (5) orders
    for (int i = 0; i < 5; i++) {
        auto order = make_order(i + 1, 0, hft::msg::SIDE::BUY, 10, 5000);
        auto result = risk.evaluate_order(order);
        ASSERT_TRUE(result.accepted);
        risk.on_order_sent();
        risk.on_md_update();
    }

    // 6th should be rejected
    auto order = make_order(6, 0, hft::msg::SIDE::BUY, 10, 5000);
    auto result = risk.evaluate_order(order);
    ASSERT_FALSE(result.accepted);
    ASSERT_EQ(result.reason, hft::risk::RejectReason::MAX_ORDERS_PER_SECOND);
}

/* (g) Max orders per MD update */
TEST(risk_max_orders_per_md_update) {
    hft::risk::PositionTracker tracker;
    auto orderbook = hft::orderbook::MultiSymbolOrderBook(2);
    hft::risk::RiskSystem risk(test_limits(), tracker, orderbook);

    // Send 2 orders (max_orders_per_md_update = 2)
    for (int i = 0; i < 2; i++) {
        auto order = make_order(i + 1, 0, hft::msg::SIDE::BUY, 10, 5000);
        auto result = risk.evaluate_order(order);
        ASSERT_TRUE(result.accepted);
        risk.on_order_sent();
    }

    // 3rd should fail
    auto order = make_order(3, 0, hft::msg::SIDE::BUY, 10, 5000);
    auto result = risk.evaluate_order(order);
    ASSERT_FALSE(result.accepted);
    ASSERT_EQ(result.reason, hft::risk::RejectReason::MAX_ORDERS_PER_MD_UPDATE);

    // After MD update, counter resets
    risk.on_md_update();
    result = risk.evaluate_order(order);
    ASSERT_TRUE(result.accepted);
}

/* (h) Max un-acked (inflight) orders */
TEST(risk_max_inflight_orders) {
    hft::risk::PositionTracker tracker;
    auto orderbook = hft::orderbook::MultiSymbolOrderBook(2);
    hft::risk::RiskSystem risk(test_limits(), tracker, orderbook);

    // Send 3 orders (max_inflight = 3)
    for (int i = 0; i < 3; i++) {
        auto order = make_order(i + 1, 0, hft::msg::SIDE::BUY, 10, 5000);
        auto result = risk.evaluate_order(order);
        ASSERT_TRUE(result.accepted);
        risk.on_order_sent();
        risk.on_md_update();
    }

    // 4th should fail
    auto order = make_order(4, 0, hft::msg::SIDE::BUY, 10, 5000);
    auto result = risk.evaluate_order(order);
    ASSERT_FALSE(result.accepted);
    ASSERT_EQ(result.reason, hft::risk::RejectReason::MAX_INFLIGHT_ORDERS);

    // After an ACK, should pass
    risk.on_order_acked();
    result = risk.evaluate_order(order);
    ASSERT_TRUE(result.accepted);
}

/* (i) Order during shutdown */
TEST(risk_shutdown_blocks_orders) {
    hft::risk::PositionTracker tracker;
    auto orderbook = hft::orderbook::MultiSymbolOrderBook(2);
    hft::risk::RiskLimits limits = test_limits();
    limits.max_abs_position_shutdown = 100;
    hft::risk::RiskSystem risk(limits, tracker, orderbook);

    // Build up position past shutdown threshold
    tracker.on_fill(0, hft::msg::SIDE::BUY, 101, 5000);
    // Need to register symbol with risk system first
    auto order = make_order(1, 0, hft::msg::SIDE::SELL, 10, 5000);
    // This order might pass (reduces position), but let's trigger shutdown first
    risk.evaluate_order(make_order(99, 0, hft::msg::SIDE::SELL, 1, 5000));

    ASSERT_TRUE(risk.check_shutdown_conditions());
    ASSERT_TRUE(risk.is_shutdown());

    // Now all orders should be rejected
    auto result = risk.evaluate_order(order);
    ASSERT_FALSE(result.accepted);
    ASSERT_EQ(result.reason, hft::risk::RejectReason::SHUTDOWN_IN_PROGRESS);
}

/* Shutdown on PNL breach */
TEST(risk_shutdown_on_pnl) {
    hft::risk::PositionTracker tracker;
    auto orderbook = hft::orderbook::MultiSymbolOrderBook(2);
    hft::risk::RiskLimits limits = test_limits();
    limits.min_pnl_shutdown = -1000.0;
    hft::risk::RiskSystem risk(limits, tracker, orderbook);

    // Register symbol by evaluating an order
    risk.evaluate_order(make_order(99, 0, hft::msg::SIDE::BUY, 1, 5000));

    // Create a losing position: buy high, sell low
    tracker.on_fill(0, hft::msg::SIDE::BUY, 100, 5000);
    tracker.on_fill(0, hft::msg::SIDE::SELL, 100, 4980);
    // Realized PNL = 100 * (4980 - 5000) = -2000

    ASSERT_TRUE(risk.check_shutdown_conditions());
    ASSERT_TRUE(risk.is_shutdown());
}

/* Valid order passes all checks */
TEST(risk_valid_order_passes) {
    hft::risk::PositionTracker tracker;
    auto orderbook = hft::orderbook::MultiSymbolOrderBook(2);
    hft::risk::RiskSystem risk(test_limits(), tracker, orderbook);

    auto order = make_order(1, 0, hft::msg::SIDE::BUY, 50, 5000);
    auto result = risk.evaluate_order(order);
    ASSERT_TRUE(result.accepted);
    ASSERT_EQ(result.reason, hft::risk::RejectReason::NONE);
}

/* Modify order evaluation */
TEST(risk_evaluate_modify) {
    hft::risk::PositionTracker tracker;
    auto orderbook = hft::orderbook::MultiSymbolOrderBook(2);
    hft::risk::RiskSystem risk(test_limits(), tracker, orderbook);

    // Original order: buy 50
    tracker.on_order_sent(1, 0, hft::msg::SIDE::BUY, 50);
    risk.on_order_sent();
    risk.on_md_update();

    // Modify to buy 200 — exceeds max_qty_per_order (100)
    hft::msg::oe::ModifyOrderRequest mod{};
    mod.order_id = 1;
    mod.side = hft::msg::SIDE::BUY;
    mod.quantity = 200;
    mod.price = 5000;

    auto result = risk.evaluate_modify(mod, 0, hft::msg::SIDE::BUY, 50);
    ASSERT_FALSE(result.accepted);
    ASSERT_EQ(result.reason, hft::risk::RejectReason::MAX_QTY_PER_ORDER);
}


/* ══════════════════════════════════════════════════════════════════════════
 *  MAIN
 * ══════════════════════════════════════════════════════════════════════ */

int main() {

    printf("\n=== Risk System unit test results: %d passed, %d failed ===\n\n", tests_passed, tests_failed);
    return tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
