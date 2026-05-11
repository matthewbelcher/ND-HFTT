#include "matching_main.h"

#include <iostream>

int main() {
    using namespace ghostbook::matching;

    MatchingEngine engine;
    engine.run_once();

    NewOrderCommand bid{
        .client_order_id = 1,
        .instrument_id = 1001,
        .side = Side::Buy,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::Day,
        .price_tick = 1012300,
        .quantity = 10,
    };

    NewOrderCommand ask{
        .client_order_id = 2,
        .instrument_id = 1001,
        .side = Side::Sell,
        .order_type = OrderType::Limit,
        .tif = TimeInForce::Day,
        .price_tick = 1012300,
        .quantity = 4,
    };

    engine.submit(bid);
    engine.submit(ask);

    for (const auto& event : engine.drain_events()) {
        std::cout << "event type=" << static_cast<int>(event.type) << " order=" << event.order_id
                  << " qty=" << event.quantity << " leaves=" << event.remaining_quantity
                  << " cum=" << event.cumulative_quantity << " clock=" << event.logical_clock
                  << " reason=" << event.reason << std::endl;
    }

    return 0;
}
