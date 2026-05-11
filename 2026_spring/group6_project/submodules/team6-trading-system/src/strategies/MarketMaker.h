#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <hft/ExchangeClient.h>
#include <hft/ExchangeOrder.h>
#include <hft/Strategy.h>
#include <hft/msg/common.h>

#include "Logger.h"

namespace hft::strategy::impl {

/**
 * Two-sided market maker.
 *
 * Continuously posts a bid and an ask around the mid-price with a configurable
 * half-spread.  Quotes are repriced on every MD update when the BBO moves.
 *
 * Inventory skew: when |position| > max_skew, the quote on the accumulating
 * side is pushed away by (|position| - max_skew) ticks, scaling aggressively
 * as inventory grows.
 *
 * Stop-loss: when total PnL drops below the threshold both quotes are cancelled
 * and the strategy enters PAUSED.  It resumes automatically once PnL recovers
 * to half the threshold, allowing it to work off any residual position rather
 * than holding it indefinitely.
 */
class MarketMaker final : public Strategy {
public:
    MarketMaker(client::ExchangeClient &client,
                msg::symbol_id_t symbol,
                msg::quantity_t quote_qty,
                msg::price_t half_spread,
                int64_t max_skew,
                double stop_loss_threshold)
        : Strategy(client, client.get_orderbook(), client.get_position_tracker()),
          symbol_(symbol),
          quote_qty_(quote_qty),
          half_spread_(half_spread),
          max_skew_(max_skew),
          stop_loss_(stop_loss_threshold) {
        subscribe_md(symbol_);
    }

    void on_market_data(msg::symbol_id_t, const msg::md::MdMessage &) override {
        const auto bbo = client_.get_bbo(symbol_);
        if (bbo.bid_price == 0 || bbo.ask_price == 0) return;

        const msg::price_t mid = log::round_dn((bbo.bid_price + bbo.ask_price) / 2);
        const double pnl = tracker_.get_total_pnl(symbol_, mid);

        if (phase_ == PHASE::PAUSED) {
            // Resume once PnL recovers to half the loss threshold.
            if (pnl >= stop_loss_ * 0.5) {
                phase_ = PHASE::QUOTING;
                log::log_action(symbol_, "MARKET MAKER RESUMED");
            } else {
                return;
            }
        }

        // Stop-loss: pause and cancel quotes until PnL recovers.
        if (pnl < stop_loss_) {
            cancel_side(bid_order_);
            cancel_side(ask_order_);
            phase_ = PHASE::PAUSED;
            log::log_action(symbol_, "MARKET MAKER PAUSED: stop-loss triggered");
            return;
        }

        const int64_t position = tracker_.get_position(symbol_);

        // Compute desired quotes with proportional inventory skew.
        // Each tick beyond max_skew pushes the accumulating side further away,
        // making it progressively harder to grow an already-large position.
        msg::price_t desired_bid = mid - half_spread_;
        msg::price_t desired_ask = mid + half_spread_;

        const auto skew_steps = static_cast<msg::price_t>(
            std::max<int64_t>(0, std::abs(position) - max_skew_));
        const msg::price_t skew = skew_steps * 5;
        if (position >  max_skew_) desired_bid -= skew;
        if (position < -max_skew_) desired_ask += skew;

        // Clamp to not be more aggressive than the best BBO.
        desired_bid = std::min(desired_bid, bbo.bid_price);
        desired_ask = std::max(desired_ask, bbo.ask_price);

        // Enforce a minimum 1-tick separation so quotes never cross.
        if (desired_bid >= desired_ask) {
            desired_bid = desired_ask - 5;
        }

        // Tick-align both prices.
        desired_bid = log::round_dn(desired_bid);
        desired_ask = log::round_up(desired_ask);

        manage_side(bid_order_, msg::SIDE::BUY,  desired_bid, position, +1);
        manage_side(ask_order_, msg::SIDE::SELL, desired_ask, position, -1);
    }

private:
    enum class PHASE { QUOTING, PAUSED };

    const msg::symbol_id_t symbol_;
    const msg::quantity_t  quote_qty_;
    const msg::price_t     half_spread_;
    const int64_t          max_skew_;
    const double           stop_loss_;

    PHASE phase_ = PHASE::QUOTING;

    order::ExchangeOrder *bid_order_ = nullptr;
    order::ExchangeOrder *ask_order_ = nullptr;
    std::vector<std::unique_ptr<order::ExchangeOrder>> orders_;

    /* ------------------------------------------------------------------ */

    // direction: +1 for BUY side (position must not exceed +10),
    //            -1 for SELL side (position must not go below -10).
    void manage_side(order::ExchangeOrder *&slot,
                     msg::SIDE side,
                     msg::price_t desired_price,
                     int64_t position,
                     int direction) {
        // Position-limit guard: skip placing if the new order would push us
        // further past the hard limit in that direction.
        const int64_t projected = position + direction * static_cast<int64_t>(quote_qty_);
        if (projected > 10 || projected < -10) {
            // Cancel any existing resting order on that side.
            if (slot && slot->status == order::ORDER_STATUS::OPEN) {
                slot->cancel();
            }
            return;
        }

        if (slot == nullptr) {
            place_order(slot, side, quote_qty_, desired_price);
            return;
        }

        switch (slot->status) {
        case order::ORDER_STATUS::CLOSED:
            slot = nullptr;
            break;
        case order::ORDER_STATUS::OPEN:
            if (slot->local_state.price != desired_price) {
                slot->modify(side, quote_qty_, desired_price);
            }
            break;
        default:
            // IN_FLIGHT / UPDATE_IN_FLIGHT / CANCEL_IN_FLIGHT: wait for ACK.
            break;
        }
    }

    void place_order(order::ExchangeOrder *&slot,
                     msg::SIDE side,
                     msg::quantity_t qty,
                     msg::price_t price) {
        orders_.push_back(
            std::make_unique<order::ExchangeOrder>(client_, 0, symbol_, side, qty, price));
        auto *raw = orders_.back().get();
        raw->commit();
        slot = raw;
    }

    void cancel_side(order::ExchangeOrder *&slot) {
        if (!slot) return;
        if (slot->status == order::ORDER_STATUS::OPEN ||
            slot->status == order::ORDER_STATUS::IN_FLIGHT) {
            slot->cancel();
        }
        slot = nullptr;
    }
};

} // namespace hft::strategy::impl
