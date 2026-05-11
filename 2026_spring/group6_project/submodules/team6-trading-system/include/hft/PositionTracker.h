/*
 * Position, exposure, and PNL tracking per symbol
 */

#ifndef HFT_POSITION_TRACKER_H
#define HFT_POSITION_TRACKER_H

#include <cstdint>
#include <unordered_map>

#include "msg/common.h"

namespace hft::risk {

    /* Data Structures */

    struct SymbolState {
        // Position tracking
        int64_t net_position = 0;           // total bought - total sold (signed)
        uint64_t total_bought_qty = 0;
        uint64_t total_sold_qty = 0;
        int64_t total_buy_value = 0;        // Σ(qty * price) for all buys
        int64_t total_sell_value = 0;       // Σ(qty * price) for all sells

        // Exposure tracking (outstanding order quantities)
        uint64_t outstanding_buy_qty = 0;
        uint64_t outstanding_sell_qty = 0;
    };

    struct TrackedOrder {
        msg::symbol_id_t symbol;
        msg::SIDE side;
        msg::quantity_t remaining_qty;
    };


    /* Class Definitions */

    /**
     * Tracks per-symbol position, exposure, and PNL based on exchange fill
     * notifications and outstanding order lifecycle events.
     */
    class PositionTracker {
    public:

        /* Fill notification — updates position */

        void on_fill(msg::symbol_id_t symbol, msg::SIDE side,
                     msg::quantity_t qty, msg::price_t price);

        /* Exposure tracking — updates outstanding order state */

        void on_order_sent(msg::order_id_t id, msg::symbol_id_t symbol,
                           msg::SIDE side, msg::quantity_t qty);
        void on_order_fill(msg::order_id_t id, msg::quantity_t filled_qty);
        void on_order_closed(msg::order_id_t id);
        void on_order_modified(msg::order_id_t id, msg::SIDE new_side,
                               msg::quantity_t new_qty);

        /* Position getters */

        [[nodiscard]] int64_t get_position(msg::symbol_id_t symbol) const;
        [[nodiscard]] const SymbolState& get_state(msg::symbol_id_t symbol) const;

        /* Exposure getters */

        [[nodiscard]] int64_t get_buy_exposure(msg::symbol_id_t symbol) const;
        [[nodiscard]] int64_t get_sell_exposure(msg::symbol_id_t symbol) const;

        /* PNL getters */

        [[nodiscard]] double get_realized_pnl(msg::symbol_id_t symbol) const;
        [[nodiscard]] double get_unrealized_pnl(msg::symbol_id_t symbol,
                                                 msg::price_t mid_price) const;
        [[nodiscard]] double get_total_pnl(msg::symbol_id_t symbol,
                                            msg::price_t mid_price) const;

    private:
        std::unordered_map<msg::symbol_id_t, SymbolState> states_;
        std::unordered_map<msg::order_id_t, TrackedOrder> tracked_orders_;

        static const SymbolState empty_state_;
    };

}

#endif //HFT_POSITION_TRACKER_H
