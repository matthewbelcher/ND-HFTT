#include "matching_main.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using ghostbook::matching::EventType;
using ghostbook::matching::MatchingEngine;
using ghostbook::matching::instrument_id_t;
using ghostbook::matching::NewOrderCommand;
using ghostbook::matching::OrderType;
using ghostbook::matching::order_id_t;
using ghostbook::matching::price_tick_t;
using ghostbook::matching::Side;
using ghostbook::matching::quantity_t;
using ghostbook::matching::TimeInForce;

namespace {

bool expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        return false;
    }
    return true;
}

bool test_fifo_within_price_level() {
    MatchingEngine engine;

    NewOrderCommand maker_one{
        .client_order_id = 100,
        .instrument_id = 1,
        .side = Side::Sell,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::Day,
        .price_tick = 100,
        .quantity = 5,
    };
    NewOrderCommand maker_two{
        .client_order_id = 101,
        .instrument_id = 1,
        .side = Side::Sell,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::Day,
        .price_tick = 100,
        .quantity = 7,
    };

    engine.submit(maker_one);
    engine.submit(maker_two);
    (void)engine.drain_events();

    NewOrderCommand taker{
        .client_order_id = 200,
        .instrument_id = 1,
        .side = Side::Buy,
        .order_type = OrderType::Market,
        .tif = TimeInForce::IOC,
        .price_tick = 0,
        .quantity = 6,
    };

    engine.submit(taker);
    const auto events = engine.drain_events();

    if (!expect(events.size() == 5, "fifo test expected 5 events")) {
        return false;
    }

    bool ok = true;
    ok &= expect(events[0].type == EventType::Ack, "fifo event 0 should be ack");
    ok &= expect(events[1].type == EventType::Fill, "fifo event 1 should be fill");
    ok &= expect(events[1].order_id == 200 && events[1].counterparty_order_id == 100,
                 "fifo taker first fill should hit maker 100");
    ok &= expect(events[1].quantity == 5 && events[1].remaining_quantity == 1,
                 "fifo taker first fill qty/leaves mismatch");
    ok &= expect(events[2].type == EventType::Fill && events[2].order_id == 100,
                 "fifo event 2 should be maker 100 fill");
    ok &= expect(events[3].type == EventType::Fill, "fifo event 3 should be fill");
    ok &= expect(events[3].order_id == 200 && events[3].counterparty_order_id == 101,
                 "fifo taker second fill should hit maker 101");
    ok &= expect(events[3].quantity == 1 && events[3].remaining_quantity == 0,
                 "fifo taker second fill qty/leaves mismatch");
    ok &= expect(events[4].type == EventType::Fill && events[4].order_id == 101,
                 "fifo event 4 should be maker 101 fill");

    return ok;
}

bool test_ioc_cancels_unfilled_remainder() {
    MatchingEngine engine;

    NewOrderCommand maker{
        .client_order_id = 300,
        .instrument_id = 2,
        .side = Side::Sell,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::Day,
        .price_tick = 100,
        .quantity = 3,
    };
    engine.submit(maker);
    (void)engine.drain_events();

    NewOrderCommand taker{
        .client_order_id = 301,
        .instrument_id = 2,
        .side = Side::Buy,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::IOC,
        .price_tick = 100,
        .quantity = 5,
    };

    engine.submit(taker);
    const auto events = engine.drain_events();

    if (!expect(events.size() == 4, "ioc test expected 4 events")) {
        return false;
    }

    bool ok = true;
    ok &= expect(events[0].type == EventType::Ack, "ioc event 0 should be ack");
    ok &= expect(events[1].type == EventType::Fill && events[1].order_id == 301,
                 "ioc event 1 should be taker fill");
    ok &= expect(events[1].quantity == 3 && events[1].remaining_quantity == 2,
                 "ioc taker fill qty/leaves mismatch");
    ok &= expect(events[3].type == EventType::Cancel && events[3].order_id == 301,
                 "ioc event 3 should be taker cancel remainder");
    ok &= expect(events[3].quantity == 2 && events[3].remaining_quantity == 0,
                 "ioc cancel remainder qty/leaves mismatch");

    return ok;
}

bool test_post_only_rejects_crossing_order() {
    MatchingEngine engine;

    NewOrderCommand maker{
        .client_order_id = 400,
        .instrument_id = 3,
        .side = Side::Sell,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::Day,
        .price_tick = 100,
        .quantity = 2,
    };
    engine.submit(maker);
    (void)engine.drain_events();

    NewOrderCommand post_only_cross{
        .client_order_id = 401,
        .instrument_id = 3,
        .side = Side::Buy,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::PostOnly,
        .price_tick = 100,
        .quantity = 1,
    };

    engine.submit(post_only_cross);
    const auto events = engine.drain_events();

    if (!expect(events.size() == 1, "post-only test expected 1 event")) {
        return false;
    }

    bool ok = true;
    ok &= expect(events[0].type == EventType::Reject, "post-only should reject crossing order");
    ok &= expect(events[0].reason == "post-only would cross", "post-only reject reason mismatch");
    return ok;
}

bool test_fok_rejects_when_not_fully_executable() {
    MatchingEngine engine;

    NewOrderCommand maker{
        .client_order_id = 500,
        .instrument_id = 4,
        .side = Side::Sell,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::Day,
        .price_tick = 100,
        .quantity = 3,
    };
    engine.submit(maker);
    (void)engine.drain_events();

    NewOrderCommand fok_buy{
        .client_order_id = 501,
        .instrument_id = 4,
        .side = Side::Buy,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::FOK,
        .price_tick = 100,
        .quantity = 5,
    };

    engine.submit(fok_buy);
    const auto events = engine.drain_events();

    if (!expect(events.size() == 1, "fok test expected 1 event")) {
        return false;
    }

    bool ok = true;
    ok &= expect(events[0].type == EventType::Reject, "fok should reject when not fully executable");
    ok &= expect(events[0].reason == "fok not fully executable", "fok reject reason mismatch");
    return ok;
}

bool test_cancel_partial_and_full() {
    MatchingEngine engine;

    NewOrderCommand maker{
        .client_order_id = 600,
        .instrument_id = 5,
        .side = Side::Buy,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::Day,
        .price_tick = 100,
        .quantity = 10,
    };
    engine.submit(maker);
    (void)engine.drain_events();

    auto partial_cancel = engine.cancel({.client_order_id = 601, .target_order_id = 600, .cancel_quantity = 4});
    const auto partial_events = engine.drain_events();

    bool ok = true;
    ok &= expect(partial_cancel.has_value(), "partial cancel should return event");
    ok &= expect(partial_events.size() == 1, "partial cancel should emit one event");
    ok &= expect(partial_events[0].type == EventType::Cancel, "partial cancel event type mismatch");
    ok &= expect(partial_events[0].quantity == 4 && partial_events[0].remaining_quantity == 6,
                 "partial cancel quantity/leaves mismatch");

    auto full_cancel = engine.cancel({.client_order_id = 602, .target_order_id = 600, .cancel_quantity = 0});
    const auto full_events = engine.drain_events();
    ok &= expect(full_cancel.has_value(), "full cancel should return event");
    ok &= expect(full_events.size() == 1, "full cancel should emit one event");
    ok &= expect(full_events[0].type == EventType::Cancel, "full cancel event type mismatch");
    ok &= expect(full_events[0].quantity == 6 && full_events[0].remaining_quantity == 0,
                 "full cancel quantity/leaves mismatch");

    auto reject_cancel = engine.cancel({.client_order_id = 603, .target_order_id = 600, .cancel_quantity = 0});
    const auto reject_events = engine.drain_events();
    ok &= expect(reject_cancel.has_value(), "cancel-not-found should return reject event");
    ok &= expect(reject_events.size() == 1, "cancel-not-found should emit one event");
    ok &= expect(reject_events[0].type == EventType::Reject, "cancel-not-found should reject");

    return ok;
}

bool test_modify_rejects_duplicate_new_id() {
    MatchingEngine engine;

    NewOrderCommand existing{
        .client_order_id = 700,
        .instrument_id = 6,
        .side = Side::Buy,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::Day,
        .price_tick = 100,
        .quantity = 5,
    };
    NewOrderCommand other{
        .client_order_id = 701,
        .instrument_id = 6,
        .side = Side::Buy,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::Day,
        .price_tick = 99,
        .quantity = 5,
    };

    engine.submit(existing);
    engine.submit(other);
    (void)engine.drain_events();

    auto modify_event =
        engine.modify({.original_order_id = 700, .new_order_id = 701, .new_price_tick = 101, .new_quantity = 5});
    const auto events = engine.drain_events();

    bool ok = true;
    ok &= expect(modify_event.has_value(), "modify duplicate-id should return event");
    ok &= expect(events.size() == 1, "modify duplicate-id should emit one event");
    ok &= expect(events[0].type == EventType::Reject, "modify duplicate-id should reject");
    ok &= expect(events[0].reason == "new order id already exists", "modify duplicate-id reason mismatch");
    return ok;
}

bool test_modify_replaces_and_matches() {
    MatchingEngine engine;

    NewOrderCommand passive_sell{
        .client_order_id = 800,
        .instrument_id = 7,
        .side = Side::Sell,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::Day,
        .price_tick = 105,
        .quantity = 4,
    };
    NewOrderCommand buy_resting{
        .client_order_id = 801,
        .instrument_id = 7,
        .side = Side::Buy,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::Day,
        .price_tick = 100,
        .quantity = 4,
    };

    engine.submit(passive_sell);
    engine.submit(buy_resting);
    (void)engine.drain_events();

    auto modify_event =
        engine.modify({.original_order_id = 801, .new_order_id = 802, .new_price_tick = 105, .new_quantity = 4});
    const auto events = engine.drain_events();

    if (!expect(modify_event.has_value(), "modify replace should return event")) {
        return false;
    }

    bool seen_ack = false;
    bool seen_taker_fill = false;
    bool seen_maker_fill = false;
    for (const auto& event : events) {
        if (event.type == EventType::Ack && event.order_id == 802) {
            seen_ack = true;
        }
        if (event.type == EventType::Fill && event.order_id == 802 && event.quantity == 4) {
            seen_taker_fill = true;
        }
        if (event.type == EventType::Fill && event.order_id == 800 && event.quantity == 4) {
            seen_maker_fill = true;
        }
    }

    bool ok = true;
    ok &= expect(seen_ack, "modify replace should ack new id");
    ok &= expect(seen_taker_fill, "modify replace should produce taker fill");
    ok &= expect(seen_maker_fill, "modify replace should produce maker fill");
    return ok;
}

bool test_market_post_only_rejected() {
    MatchingEngine engine;

    NewOrderCommand invalid_order{
        .client_order_id = 900,
        .instrument_id = 8,
        .side = Side::Buy,
        .order_type = OrderType::Market,
        .tif = TimeInForce::PostOnly,
        .price_tick = 0,
        .quantity = 1,
    };

    engine.submit(invalid_order);
    const auto events = engine.drain_events();

    if (!expect(events.size() == 1, "market post-only test expected 1 event")) {
        return false;
    }

    bool ok = true;
    ok &= expect(events[0].type == EventType::Reject, "market post-only should reject");
    ok &= expect(events[0].reason == "post-only requires limit order",
                 "market post-only reject reason mismatch");
    return ok;
}

bool test_modify_reject_keeps_original_order() {
    MatchingEngine engine;

    NewOrderCommand resting_buy{
        .client_order_id = 910,
        .instrument_id = 9,
        .side = Side::Buy,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::Day,
        .price_tick = 100,
        .quantity = 5,
    };
    engine.submit(resting_buy);
    (void)engine.drain_events();

    auto modify_event =
        engine.modify({.original_order_id = 910, .new_order_id = 911, .new_price_tick = 0, .new_quantity = 5});
    const auto modify_events = engine.drain_events();

    bool ok = true;
    ok &= expect(modify_event.has_value(), "modify invalid-price should return event");
    ok &= expect(modify_events.size() == 1, "modify invalid-price should emit one event");
    ok &= expect(modify_events[0].type == EventType::Reject, "modify invalid-price should reject");
    ok &= expect(modify_events[0].reason == "limit price must be > 0",
                 "modify invalid-price reason mismatch");

    NewOrderCommand aggressive_sell{
        .client_order_id = 912,
        .instrument_id = 9,
        .side = Side::Sell,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::IOC,
        .price_tick = 100,
        .quantity = 5,
    };
    engine.submit(aggressive_sell);
    const auto events = engine.drain_events();

    bool filled_against_original = false;
    bool unexpected_new_id_fill = false;
    for (const auto& event : events) {
        if (event.type == EventType::Fill && event.order_id == 912 && event.counterparty_order_id == 910 &&
            event.quantity == 5) {
            filled_against_original = true;
        }
        if (event.type == EventType::Fill && event.counterparty_order_id == 911) {
            unexpected_new_id_fill = true;
        }
    }

    ok &= expect(filled_against_original, "failed modify should keep original order resting");
    ok &= expect(!unexpected_new_id_fill, "failed modify should not leave replacement order");
    return ok;
}

bool test_bulk_match_stress() {
    MatchingEngine engine;

    constexpr std::size_t maker_count = 2048;
    constexpr std::uint64_t maker_id_base = 1'000;
    constexpr std::uint64_t taker_id = 9'000;
    constexpr price_tick_t base_price = 100;
    constexpr std::size_t levels = 8;
    constexpr std::size_t per_level = maker_count / levels;

    for (std::size_t i = 0; i < maker_count; ++i) {
        engine.submit(NewOrderCommand{
            .client_order_id = maker_id_base + static_cast<std::uint64_t>(i),
            .instrument_id = 10,
            .side = Side::Sell,
            .order_type = OrderType::Limit,
            .tif = TimeInForce::Day,
            .price_tick = base_price + static_cast<price_tick_t>(i / per_level),
            .quantity = 1,
        });
    }
    (void)engine.drain_events();

    engine.submit(NewOrderCommand{
        .client_order_id = taker_id,
        .instrument_id = 10,
        .side = Side::Buy,
        .order_type = OrderType::Market,
        .tif = TimeInForce::IOC,
        .price_tick = 0,
        .quantity = static_cast<quantity_t>(maker_count),
    });

    const auto events = engine.drain_events();
    if (!expect(events.size() == 1 + maker_count * 2, "bulk match should emit ack plus paired fills")) {
        return false;
    }

    bool ok = true;
    ok &= expect(events[0].type == EventType::Ack && events[0].order_id == taker_id,
                 "bulk match should begin with taker ack");

    std::uint64_t taker_filled = 0;
    for (std::size_t i = 1; i < events.size(); i += 2) {
        const auto& taker_fill = events[i];
        const auto& maker_fill = events[i + 1];

        ok &= expect(taker_fill.type == EventType::Fill, "bulk match taker event type mismatch");
        ok &= expect(maker_fill.type == EventType::Fill, "bulk match maker event type mismatch");
        ok &= expect(taker_fill.order_id == taker_id, "bulk match taker order id mismatch");
        ok &= expect(maker_fill.order_id >= maker_id_base &&
                         maker_fill.order_id < maker_id_base + static_cast<std::uint64_t>(maker_count),
                     "bulk match maker order id out of range");
        ok &= expect(taker_fill.counterparty_order_id == maker_fill.order_id,
                     "bulk match taker/maker pairing mismatch");
        ok &= expect(taker_fill.quantity == 1 && maker_fill.quantity == 1,
                     "bulk match quantities should remain unit size");
        ok &= expect(maker_fill.remaining_quantity == 0, "bulk match maker should fully exhaust each order");

        taker_filled += taker_fill.quantity;
    }

    ok &= expect(taker_filled == maker_count, "bulk match should fill the full taker quantity");
    ok &= expect(events.back().remaining_quantity == 0, "bulk match should end fully filled");
    return ok;
}

bool test_randomized_invariants() {
    MatchingEngine engine;

    struct RestingOrderRef {
        order_id_t order_id{};
        instrument_id_t instrument_id{};
        price_tick_t price_tick{};
        Side side{Side::Buy};
    };

    std::unordered_map<order_id_t, RestingOrderRef> active_orders;
    std::vector<order_id_t> order_ids;

    std::mt19937_64 rng(0xC0FFEEu);
    std::uniform_int_distribution<int> action_dist(0, 4);
    std::uniform_int_distribution<int> instrument_dist(0, 1);
    std::uniform_int_distribution<int> price_dist(100, 110);

    order_id_t next_order_id = 50'000;

    auto erase_order_from_vector = [&order_ids](order_id_t order_id) {
        const auto it = std::find(order_ids.begin(), order_ids.end(), order_id);
        if (it != order_ids.end()) {
            order_ids.erase(it);
        }
    };

    for (std::size_t step = 0; step < 300; ++step) {
        const int action = action_dist(rng);

        if (action <= 1 || active_orders.empty()) {
            const order_id_t order_id = next_order_id++;
            const instrument_id_t instrument_id = static_cast<instrument_id_t>(11 + instrument_dist(rng));
            const Side side = instrument_id == 11 ? Side::Buy : Side::Sell;
            const price_tick_t price_tick = static_cast<price_tick_t>(price_dist(rng));
            const NewOrderCommand command{
                .client_order_id = order_id,
                .instrument_id = instrument_id,
                .side = side,
                .order_type = OrderType::Limit,
                .tif = TimeInForce::Day,
                .price_tick = price_tick,
                .quantity = 1,
            };

            engine.submit(command);
            const auto events = engine.drain_events();

            if (!expect(!events.empty(), "randomized submit should emit at least one event")) {
                return false;
            }

            if (!expect(events[0].type == EventType::Ack && events[0].order_id == order_id,
                        "randomized submit should ack the submitted order")) {
                return false;
            }

            if (!expect(events.size() == 1, "randomized submit should not match in the non-crossing scenario")) {
                return false;
            }

            active_orders.emplace(order_id, RestingOrderRef{order_id, instrument_id, price_tick, side});
            order_ids.push_back(order_id);

            continue;
        }

        auto active_it = active_orders.begin();
        std::advance(active_it, static_cast<std::ptrdiff_t>(rng() % active_orders.size()));
        const order_id_t target_id = active_it->first;

        if (action == 2) {
            engine.cancel({.client_order_id = next_order_id++, .target_order_id = target_id, .cancel_quantity = 0});
            const auto events = engine.drain_events();
            if (!expect(events.size() == 1, "randomized cancel should emit one event")) {
                return false;
            }
            if (!expect(events[0].type == EventType::Cancel && events[0].order_id == target_id,
                        "randomized cancel should target the selected order")) {
                return false;
            }
            active_orders.erase(target_id);
            erase_order_from_vector(target_id);
            continue;
        }

        if (action == 3) {
            const price_tick_t new_price = static_cast<price_tick_t>(price_dist(rng));
            const order_id_t new_id = next_order_id++;
            const auto target_order = active_it->second;
            engine.modify({
                .original_order_id = target_id,
                .new_order_id = new_id,
                .new_price_tick = new_price,
                .new_quantity = 1,
            });
            const auto events = engine.drain_events();
            if (!expect(!events.empty(), "randomized modify should emit at least one event")) {
                return false;
            }

            if (events[0].type == EventType::Reject) {
                if (!expect(active_orders.contains(target_id), "rejected modify should keep original order active")) {
                    return false;
                }
                continue;
            }

            if (!expect(events[0].type == EventType::Ack && events[0].order_id == new_id,
                        "randomized modify should acknowledge the replacement order")) {
                return false;
            }

            active_orders.erase(target_id);
            erase_order_from_vector(target_id);
            active_orders.emplace(new_id, RestingOrderRef{new_id, target_order.instrument_id, new_price, target_order.side});
            order_ids.push_back(new_id);
            continue;
        }

        engine.cancel({.client_order_id = next_order_id++, .target_order_id = target_id, .cancel_quantity = 1});
        const auto events = engine.drain_events();
        if (!expect(!events.empty(), "randomized partial cancel should emit an event")) {
            return false;
        }
        if (events[0].type == EventType::Cancel) {
            active_orders.erase(target_id);
            erase_order_from_vector(target_id);
        }
    }

    bool ok = true;
    ok &= expect(active_orders.size() == order_ids.size(), "randomized active order tracking should stay in sync");
    for (const auto& [order_id, order] : active_orders) {
        ok &= expect(std::find(order_ids.begin(), order_ids.end(), order_id) != order_ids.end(),
                     "randomized order id list should contain every active order");
        ok &= expect(order.price_tick >= 100 && order.price_tick <= 110,
                     "randomized order prices should stay within the configured range");
        ok &= expect(order.instrument_id == 11 || order.instrument_id == 12,
                     "randomized order instruments should stay within the configured range");
        ok &= expect((order.instrument_id == 11 && order.side == Side::Buy) ||
                         (order.instrument_id == 12 && order.side == Side::Sell),
                     "randomized orders should stay on their instrument side");
    }
    return ok;
}

bool test_long_mixed_stream_stress() {
    MatchingEngine engine;

    struct RestingOrderRef {
        order_id_t order_id{};
        instrument_id_t instrument_id{};
        price_tick_t price_tick{};
        Side side{Side::Buy};
    };

    std::unordered_map<order_id_t, RestingOrderRef> active_orders;
    std::vector<order_id_t> order_ids;

    std::mt19937_64 rng(0xBADC0FFEu);
    std::uniform_int_distribution<int> action_dist(0, 9);
    std::uniform_int_distribution<int> instrument_dist(0, 1);
    std::uniform_int_distribution<int> price_dist(95, 120);

    order_id_t next_order_id = 100'000;

    auto erase_order_from_vector = [&order_ids](order_id_t order_id) {
        const auto it = std::find(order_ids.begin(), order_ids.end(), order_id);
        if (it != order_ids.end()) {
            order_ids.erase(it);
        }
    };

    auto assert_periodic_invariants = [&](std::size_t step) {
        bool ok = true;
        ok &= expect(active_orders.size() == order_ids.size(),
                     "mixed stress map/vector active size mismatch");

        std::unordered_set<order_id_t> unique_ids;
        for (order_id_t order_id : order_ids) {
            unique_ids.insert(order_id);
        }
        ok &= expect(unique_ids.size() == order_ids.size(),
                     "mixed stress order id list contains duplicates");

        for (const auto& [order_id, order] : active_orders) {
            ok &= expect(unique_ids.contains(order_id),
                         "mixed stress active order missing from id list");
            ok &= expect(order.instrument_id == 12 || order.instrument_id == 13,
                         "mixed stress order instrument out of expected set");
            ok &= expect((order.instrument_id == 12 && order.side == Side::Buy) ||
                             (order.instrument_id == 13 && order.side == Side::Sell),
                         "mixed stress order side does not match instrument partition");
            ok &= expect(order.price_tick >= 95 && order.price_tick <= 120,
                         "mixed stress price out of expected range");
        }

        if (!ok) {
            std::cerr << "Invariant failure at step " << step << std::endl;
            return false;
        }
        return true;
    };

    const std::size_t total_steps = 5'000;
    const std::size_t invariant_interval = 250;
    for (std::size_t step = 1; step <= total_steps; ++step) {
        const int action = action_dist(rng);

        if (action <= 5 || active_orders.empty()) {
            const order_id_t order_id = next_order_id++;
            const instrument_id_t instrument_id = static_cast<instrument_id_t>(12 + instrument_dist(rng));
            const Side side = instrument_id == 12 ? Side::Buy : Side::Sell;
            const price_tick_t price_tick = static_cast<price_tick_t>(price_dist(rng));

            engine.submit(NewOrderCommand{
                .client_order_id = order_id,
                .instrument_id = instrument_id,
                .side = side,
                .order_type = OrderType::Limit,
                .tif = TimeInForce::Day,
                .price_tick = price_tick,
                .quantity = 1,
            });
            const auto events = engine.drain_events();
            if (!expect(events.size() == 1, "mixed stress submit should emit one ack event")) {
                return false;
            }
            if (!expect(events[0].type == EventType::Ack && events[0].order_id == order_id,
                        "mixed stress submit ack mismatch")) {
                return false;
            }

            active_orders.emplace(order_id, RestingOrderRef{order_id, instrument_id, price_tick, side});
            order_ids.push_back(order_id);
        } else {
            auto active_it = active_orders.begin();
            std::advance(active_it, static_cast<std::ptrdiff_t>(rng() % active_orders.size()));
            const order_id_t target_id = active_it->first;

            if (action <= 7) {
                engine.cancel({
                    .client_order_id = next_order_id++,
                    .target_order_id = target_id,
                    .cancel_quantity = 0,
                });
                const auto events = engine.drain_events();
                if (!expect(events.size() == 1, "mixed stress cancel should emit one event")) {
                    return false;
                }
                if (!expect(events[0].type == EventType::Cancel && events[0].order_id == target_id,
                            "mixed stress cancel event mismatch")) {
                    return false;
                }

                active_orders.erase(target_id);
                erase_order_from_vector(target_id);
            } else {
                const price_tick_t new_price = static_cast<price_tick_t>(price_dist(rng));
                const order_id_t new_id = next_order_id++;
                const RestingOrderRef target_order = active_it->second;

                engine.modify({
                    .original_order_id = target_id,
                    .new_order_id = new_id,
                    .new_price_tick = new_price,
                    .new_quantity = 1,
                });
                const auto events = engine.drain_events();
                if (!expect(!events.empty(), "mixed stress modify should emit event(s)")) {
                    return false;
                }
                if (!expect(events[0].type == EventType::Ack && events[0].order_id == new_id,
                            "mixed stress modify ack mismatch")) {
                    return false;
                }

                active_orders.erase(target_id);
                erase_order_from_vector(target_id);
                active_orders.emplace(
                    new_id,
                    RestingOrderRef{new_id, target_order.instrument_id, new_price, target_order.side});
                order_ids.push_back(new_id);
            }
        }

        if (step % invariant_interval == 0) {
            if (!assert_periodic_invariants(step)) {
                return false;
            }
        }
    }

    return assert_periodic_invariants(total_steps);
}

bool test_unknown_instrument_rejected() {
    MatchingEngine engine;

    NewOrderCommand invalid_instrument{
        .client_order_id = 2000,
        .instrument_id = 14,
        .side = Side::Buy,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::Day,
        .price_tick = 100,
        .quantity = 1,
    };

    engine.submit(invalid_instrument);
    const auto events = engine.drain_events();

    if (!expect(events.size() == 1, "unknown instrument should emit 1 event")) {
        return false;
    }

    bool ok = true;
    ok &= expect(events[0].type == EventType::Reject, "unknown instrument should reject");
    ok &= expect(events[0].reason == "unknown instrument", "unknown instrument reject reason mismatch");
    ok &= expect(events[0].order_id == 2000, "unknown instrument reject should carry the order id");
    return ok;
}

bool test_zero_instrument_rejected() {
    MatchingEngine engine;

    NewOrderCommand zero_instrument{
        .client_order_id = 2001,
        .instrument_id = 0,
        .side = Side::Sell,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::Day,
        .price_tick = 100,
        .quantity = 1,
    };

    engine.submit(zero_instrument);
    const auto events = engine.drain_events();

    if (!expect(events.size() == 1, "zero instrument should emit 1 event")) {
        return false;
    }

    bool ok = true;
    ok &= expect(events[0].type == EventType::Reject, "zero instrument should reject");
    ok &= expect(events[0].reason == "unknown instrument", "zero instrument reject reason mismatch");
    return ok;
}

bool test_boundary_instruments_accepted() {
    MatchingEngine engine;

    NewOrderCommand min_instrument{
        .client_order_id = 2002,
        .instrument_id = 1,
        .side = Side::Buy,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::Day,
        .price_tick = 100,
        .quantity = 1,
    };
    NewOrderCommand max_instrument{
        .client_order_id = 2003,
        .instrument_id = 13,
        .side = Side::Buy,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::Day,
        .price_tick = 100,
        .quantity = 1,
    };

    engine.submit(min_instrument);
    engine.submit(max_instrument);
    const auto events = engine.drain_events();

    if (!expect(events.size() == 2, "boundary instruments should each emit 1 ack")) {
        return false;
    }

    bool ok = true;
    ok &= expect(events[0].type == EventType::Ack && events[0].order_id == 2002,
                 "instrument 1 should be accepted");
    ok &= expect(events[1].type == EventType::Ack && events[1].order_id == 2003,
                 "instrument 13 should be accepted");
    return ok;
}

}  // namespace

int main() {
    bool ok = true;

    ok &= test_fifo_within_price_level();
    ok &= test_ioc_cancels_unfilled_remainder();
    ok &= test_post_only_rejects_crossing_order();
    ok &= test_fok_rejects_when_not_fully_executable();
    ok &= test_cancel_partial_and_full();
    ok &= test_modify_rejects_duplicate_new_id();
    ok &= test_modify_replaces_and_matches();
    ok &= test_market_post_only_rejected();
    ok &= test_modify_reject_keeps_original_order();
    ok &= test_bulk_match_stress();
    ok &= test_randomized_invariants();
    ok &= test_long_mixed_stream_stress();
    ok &= test_unknown_instrument_rejected();
    ok &= test_zero_instrument_rejected();
    ok &= test_boundary_instruments_accepted();

    if (!ok) {
        return 1;
    }

    std::cout << "All matching engine tests passed" << std::endl;
    return 0;
}
