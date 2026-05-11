#pragma once

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <vector>

#include <hft/ExchangeOrder.h>
#include <hft/Strategy.h>

namespace hft::strategy::impl {

/**
 * Liquidate all open positions across a given set of symbols.
 *
 * For each symbol with a non-zero net position, a single limit order is posted
 * at the current best bid (for longs) or best ask (for shorts).  The order is
 * repriced on every tick when the BBO moves.  Once an order closes the
 * strategy checks remaining position and, if non-zero, immediately posts a new
 * order for the remainder.  When every symbol reaches a flat position the
 * strategy reports is_done() == true.
 *
 * Orders are chunked to chunk_size to stay within per-order risk limits.
 */
class Liquidate final : public strategy::Strategy {
public:
    Liquidate(client::ExchangeClient &client,
              orderbook::MultiSymbolOrderBook &orderbook,
              risk::PositionTracker &tracker,
              std::vector<msg::symbol_id_t> symbols,
              msg::quantity_t chunk_size = 8)
        : Strategy(client, orderbook, tracker),
          symbols_(std::move(symbols)),
          chunk_size_(chunk_size) {
        for (const auto sym : symbols_) subscribe_md(sym);
    }

    // Process only the symbol whose BBO just changed.
    void on_market_data(msg::symbol_id_t symbol,
                        const msg::md::MdMessage &) override {
        process_symbol(symbol);
    }

    [[nodiscard]] bool is_done() const {
        return std::all_of(symbols_.begin(), symbols_.end(), [this](auto sym) {
            const auto it = states_.find(sym);
            return it != states_.end() && it->second.done;
        });
    }

private:
    struct SymbolState {
        order::ExchangeOrder *active_order = nullptr;
        bool done = false;
    };

    void process_symbol(const msg::symbol_id_t symbol) {
        auto &state = states_[symbol];
        if (state.done) return;

        const int64_t position = tracker_.get_position(symbol);

        if (position == 0) {
            // Cancel any lingering open order (e.g. position closed by a fill
            // that arrived between our last tick and this one).
            if (state.active_order != nullptr &&
                state.active_order->status == order::ORDER_STATUS::OPEN) {
                state.active_order->cancel();
            }
            const bool order_settled =
                state.active_order == nullptr ||
                state.active_order->status == order::ORDER_STATUS::CLOSED ||
                state.active_order->status == order::ORDER_STATUS::CANCEL_IN_FLIGHT;
            if (order_settled) {
                state.done = true;
            }
            return;
        }

        const msg::SIDE side =
            (position > 0) ? msg::SIDE::SELL : msg::SIDE::BUY;
        const auto remaining =
            static_cast<msg::quantity_t>(std::abs(position));
        const msg::quantity_t qty = std::min(remaining, chunk_size_);

        const auto bbo = client_.get_bbo(symbol);
        const msg::price_t target_price =
            (side == msg::SIDE::SELL) ? bbo.bid_price : bbo.ask_price;

        if (target_price == 0) return; // No market data yet

        if (state.active_order == nullptr) {
            send_order(state, symbol, side, qty, target_price);
        } else {
            manage_order(state, side, qty, target_price);
        }
    }

    // Place a fresh order and track it.  The order object is kept alive in
    // orders_ for the lifetime of this strategy so that the ExchangeClient's
    // raw-pointer map never dangles.
    void send_order(SymbolState &state, const msg::symbol_id_t symbol,
                    const msg::SIDE side, const msg::quantity_t qty,
                    const msg::price_t price) {
        orders_.push_back(std::make_unique<order::ExchangeOrder>(
            client_, 0, symbol, side, qty, price));
        auto *ord = orders_.back().get();
        if (ord->commit()) {
            state.active_order = ord;
        }
        // If commit() was risk-rejected the order stays LOCAL_ONLY in orders_
        // (keeping the client pointer valid) and active_order remains nullptr
        // so we retry on the next tick.
    }

    // Manage an existing active order: reprice when the BBO moves, or clear
    // the active-order slot once the order is closed so a follow-up order can
    // be issued for any remaining position.
    void manage_order(SymbolState &state, const msg::SIDE side,
                      const msg::quantity_t qty, const msg::price_t price) {
        auto *ord = state.active_order;

        switch (ord->status) {
        case order::ORDER_STATUS::CLOSED:
            // Fully filled or cancelled — next tick will create a new order
            // if position is still non-zero.
            state.active_order = nullptr;
            break;

        case order::ORDER_STATUS::OPEN:
            if (ord->local_state.price != price) {
                ord->modify(side, qty, price);
            }
            break;

        default:
            // IN_FLIGHT / UPDATE_IN_FLIGHT / CANCEL_IN_FLIGHT: wait for ACK.
            break;
        }
    }

    const std::vector<msg::symbol_id_t> symbols_;
    const msg::quantity_t chunk_size_;

    std::unordered_map<msg::symbol_id_t, SymbolState> states_;
    // Owns all ExchangeOrder objects; never shrinks so client pointers stay valid.
    std::vector<std::unique_ptr<order::ExchangeOrder>> orders_;
};

} // namespace hft::strategy::impl
