#include "hft/PositionTracker.h"

#include <cstdio>
#include <cinttypes>

namespace hft::risk {

    const SymbolState PositionTracker::empty_state_{};

    /* ── Fill notification ─────────────────────────────────────────────── */

    void PositionTracker::on_fill(const msg::symbol_id_t symbol, const msg::SIDE side,
                                   const msg::quantity_t qty, const msg::price_t price) {
        auto &s = states_[symbol];
        const int64_t value = static_cast<int64_t>(qty) * price;

        if (side == msg::SIDE::BUY) {
            s.total_bought_qty += qty;
            s.total_buy_value += value;
            s.net_position += qty;
        } else {
            s.total_sold_qty += qty;
            s.total_sell_value += value;
            s.net_position -= qty;
        }
    }

    /* ── Exposure tracking ─────────────────────────────────────────────── */

    void PositionTracker::on_order_sent(const msg::order_id_t id, const msg::symbol_id_t symbol,
                                         const msg::SIDE side, const msg::quantity_t qty) {
        auto &s = states_[symbol];

        if (side == msg::SIDE::BUY) {
            s.outstanding_buy_qty += qty;
        } else {
            s.outstanding_sell_qty += qty;
        }

        tracked_orders_[id] = TrackedOrder{symbol, side, qty};
    }

    void PositionTracker::on_order_fill(const msg::order_id_t id, const msg::quantity_t filled_qty) {
        auto it = tracked_orders_.find(id);
        if (it == tracked_orders_.end()) return;

        auto &[symbol, side, remaining] = it->second;
        auto &s = states_[symbol];

        if (side == msg::SIDE::BUY) {
            s.outstanding_buy_qty -= std::min(static_cast<uint64_t>(filled_qty), s.outstanding_buy_qty);
        } else {
            s.outstanding_sell_qty -= std::min(static_cast<uint64_t>(filled_qty), s.outstanding_sell_qty);
        }

        if (filled_qty >= remaining) {
            tracked_orders_.erase(it);
        } else {
            remaining -= filled_qty;
        }
    }

    void PositionTracker::on_order_closed(const msg::order_id_t id) {
        auto it = tracked_orders_.find(id);
        if (it == tracked_orders_.end()) return;

        auto &[symbol, side, remaining] = it->second;
        auto &s = states_[symbol];

        if (side == msg::SIDE::BUY) {
            s.outstanding_buy_qty -= std::min(static_cast<uint64_t>(remaining), s.outstanding_buy_qty);
        } else {
            s.outstanding_sell_qty -= std::min(static_cast<uint64_t>(remaining), s.outstanding_sell_qty);
        }

        tracked_orders_.erase(it);
    }

    void PositionTracker::on_order_modified(const msg::order_id_t id, const msg::SIDE new_side,
                                             const msg::quantity_t new_qty) {
        auto it = tracked_orders_.find(id);
        if (it == tracked_orders_.end()) return;

        auto &[symbol, old_side, old_remaining] = it->second;
        auto &s = states_[symbol];

        // Remove old exposure
        if (old_side == msg::SIDE::BUY) {
            s.outstanding_buy_qty -= std::min(static_cast<uint64_t>(old_remaining), s.outstanding_buy_qty);
        } else {
            s.outstanding_sell_qty -= std::min(static_cast<uint64_t>(old_remaining), s.outstanding_sell_qty);
        }

        // Add new exposure
        if (new_side == msg::SIDE::BUY) {
            s.outstanding_buy_qty += new_qty;
        } else {
            s.outstanding_sell_qty += new_qty;
        }

        it->second.side = new_side;
        it->second.remaining_qty = new_qty;
    }

    /* ── Position getters ──────────────────────────────────────────────── */

    int64_t PositionTracker::get_position(const msg::symbol_id_t symbol) const {
        auto it = states_.find(symbol);
        return (it != states_.end()) ? it->second.net_position : 0;
    }

    const SymbolState& PositionTracker::get_state(const msg::symbol_id_t symbol) const {
        auto it = states_.find(symbol);
        return (it != states_.end()) ? it->second : empty_state_;
    }

    /* ── Exposure getters ──────────────────────────────────────────────── */

    int64_t PositionTracker::get_buy_exposure(const msg::symbol_id_t symbol) const {
        auto it = states_.find(symbol);
        if (it == states_.end()) return 0;
        return it->second.net_position + static_cast<int64_t>(it->second.outstanding_buy_qty);
    }

    int64_t PositionTracker::get_sell_exposure(const msg::symbol_id_t symbol) const {
        auto it = states_.find(symbol);
        if (it == states_.end()) return 0;
        return -it->second.net_position + static_cast<int64_t>(it->second.outstanding_sell_qty);
    }

    /* ── PNL getters ───────────────────────────────────────────────────── */

    double PositionTracker::get_realized_pnl(const msg::symbol_id_t symbol) const {
        auto it = states_.find(symbol);
        if (it == states_.end()) return 0.0;

        const auto &s = it->second;

        // Realized PNL: the profit from the "closed" portion of trades.
        // If we've bought B shares and sold S shares, the closed portion
        // is min(B, S) shares on each side.
        const uint64_t closed_qty = std::min(s.total_bought_qty, s.total_sold_qty);
        if (closed_qty == 0) return 0.0;

        const double avg_buy = static_cast<double>(s.total_buy_value) / static_cast<double>(s.total_bought_qty);
        const double avg_sell = static_cast<double>(s.total_sell_value) / static_cast<double>(s.total_sold_qty);

        return static_cast<double>(closed_qty) * (avg_sell - avg_buy);
    }

    double PositionTracker::get_unrealized_pnl(const msg::symbol_id_t symbol,
                                                const msg::price_t mid_price) const {
        auto it = states_.find(symbol);
        if (it == states_.end()) return 0.0;

        const auto &s = it->second;
        if (s.net_position == 0) return 0.0;

        // Mark-to-market: value the open position at mid-price
        if (s.net_position > 0) {
            // Long position - cost basis is avg buy price
            const double avg_buy = static_cast<double>(s.total_buy_value) / static_cast<double>(s.total_bought_qty);
            return static_cast<double>(s.net_position) * (static_cast<double>(mid_price) - avg_buy);
        } else {
            // Short position - cost basis is avg sell price
            const double avg_sell = static_cast<double>(s.total_sell_value) / static_cast<double>(s.total_sold_qty);
            return static_cast<double>(-s.net_position) * (avg_sell - static_cast<double>(mid_price));
        }
    }

    double PositionTracker::get_total_pnl(const msg::symbol_id_t symbol,
                                           const msg::price_t mid_price) const {
        return get_realized_pnl(symbol) + get_unrealized_pnl(symbol, mid_price);
    }

}
