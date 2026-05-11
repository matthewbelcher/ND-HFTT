#pragma once

#include <algorithm>
#include <deque>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <hft/ExchangeClient.h>
#include <hft/ExchangeOrder.h>
#include <hft/Strategy.h>
#include <hft/msg/common.h>

#include "Logger.h"

namespace hft::strategy::impl {

/**
 * Rolling-window momentum scalper.
 *
 * Computes a price-momentum signal from the recent vs. older average mid-price
 * over a configurable history window.  When the signal exceeds a threshold the
 * strategy places an aggressive limit order (at the current best ask for longs,
 * best bid for shorts) to enter a position quickly.  The trade is exited at a
 * take-profit target; a stop-loss forces the exit at the best available price
 * if the market moves against the position.  A cooldown period prevents
 * over-trading after each round trip.
 */
class MomentumScalper final : public Strategy {
public:
    MomentumScalper(client::ExchangeClient &client,
                    msg::symbol_id_t symbol,
                    msg::quantity_t trade_qty,
                    size_t window,
                    double threshold,
                    msg::price_t take_profit,
                    msg::price_t stop_loss,
                    uint32_t cooldown_ticks)
        : Strategy(client, client.get_orderbook(), client.get_position_tracker()),
          symbol_(symbol),
          trade_qty_(trade_qty),
          window_(window),
          threshold_(threshold),
          take_profit_(take_profit),
          stop_loss_(stop_loss),
          cooldown_ticks_(cooldown_ticks) {
        subscribe_md(symbol_);
    }

    void on_market_data(msg::symbol_id_t, const msg::md::MdMessage &) override {
        const auto bbo = client_.get_bbo(symbol_);
        if (bbo.bid_price == 0 || bbo.ask_price == 0) return;

        const msg::price_t mid = (bbo.bid_price + bbo.ask_price) / 2;

        switch (phase_) {
        case PHASE::IDLE:      tick_idle(bbo, mid);      break;
        case PHASE::IN_TRADE:  tick_in_trade(bbo);       break;
        case PHASE::EXITING:   tick_exiting(bbo);        break;
        }
    }

private:
    // Ticks to wait for an aggressive entry to fill before cancelling.
    static constexpr uint32_t ENTRY_TIMEOUT = 10;

    enum class PHASE { IDLE, IN_TRADE, EXITING };
    enum class DIR   { LONG, SHORT };

    const msg::symbol_id_t symbol_;
    const msg::quantity_t  trade_qty_;
    const size_t           window_;
    const double           threshold_;
    const msg::price_t     take_profit_;
    const msg::price_t     stop_loss_;
    const uint32_t         cooldown_ticks_;

    PHASE phase_     = PHASE::IDLE;
    DIR   direction_ = DIR::LONG;

    msg::price_t entry_price_      = 0;
    uint32_t     cooldown_rem_     = 0;
    uint32_t     entry_wait_ticks_ = 0;

    std::deque<msg::price_t> history_;

    order::ExchangeOrder *active_order_ = nullptr;
    std::vector<std::unique_ptr<order::ExchangeOrder>> orders_;

    /* ------------------------------------------------------------------ */

    void tick_idle(const client::BidAndOffer &bbo, msg::price_t mid) {
        if (cooldown_rem_ > 0) { --cooldown_rem_; return; }

        // Maintain rolling mid-price history.
        history_.push_back(mid);
        if (history_.size() > window_) history_.pop_front();
        if (history_.size() < window_)  return;

        const double signal = compute_signal();
        const int64_t pos   = tracker_.get_position(symbol_);

        if (signal > threshold_ && pos + static_cast<int64_t>(trade_qty_) <= 10) {
            // Bullish: buy at best ask (aggressive limit).
            entry_price_      = bbo.ask_price;
            direction_        = DIR::LONG;
            entry_wait_ticks_ = 0;
            place_order(msg::SIDE::BUY, trade_qty_, bbo.ask_price);
            phase_ = PHASE::IN_TRADE;
        } else if (signal < -threshold_ && pos - static_cast<int64_t>(trade_qty_) >= -10) {
            // Bearish: sell at best bid (aggressive limit).
            entry_price_      = bbo.bid_price;
            direction_        = DIR::SHORT;
            entry_wait_ticks_ = 0;
            place_order(msg::SIDE::SELL, trade_qty_, bbo.bid_price);
            phase_ = PHASE::IN_TRADE;
        }
    }

    void tick_in_trade(const client::BidAndOffer &bbo) {
        if (!active_order_) { phase_ = PHASE::IDLE; return; }

        if (active_order_->status == order::ORDER_STATUS::LOCAL_ONLY) {
            active_order_ = nullptr;
            phase_        = PHASE::IDLE;
            return;
        }

        // Full fill → place exit order.
        if (active_order_->fill_quantity >= trade_qty_ ||
            active_order_->status == order::ORDER_STATUS::CLOSED) {
            if (active_order_->fill_quantity > 0) {
                entry_price_  = weighted_fill();
                active_order_ = nullptr;
                place_exit(bbo);
                phase_ = PHASE::EXITING;
            } else {
                active_order_ = nullptr;
                cooldown_rem_ = cooldown_ticks_ / 2;
                phase_        = PHASE::IDLE;
            }
            return;
        }

        // Entry timeout: market moved away, abort.
        if (++entry_wait_ticks_ >= ENTRY_TIMEOUT &&
            active_order_->status == order::ORDER_STATUS::OPEN) {
            active_order_->cancel();
            active_order_ = nullptr;
            cooldown_rem_ = cooldown_ticks_ / 2;
            phase_        = PHASE::IDLE;
        }
    }

    void tick_exiting(const client::BidAndOffer &bbo) {
        if (!active_order_) { phase_ = PHASE::IDLE; return; }

        if (active_order_->fill_quantity >= trade_qty_ ||
            active_order_->status == order::ORDER_STATUS::CLOSED) {
            const msg::price_t exit_px = weighted_fill();
            const int32_t pnl =
                (direction_ == DIR::LONG ? exit_px - entry_price_
                                         : entry_price_ - exit_px) *
                static_cast<int32_t>(active_order_->fill_quantity);
            log::log_action(symbol_, "MOMENTUM EXIT pnl=" + std::to_string(pnl));
            active_order_ = nullptr;
            cooldown_rem_ = cooldown_ticks_;
            phase_        = PHASE::IDLE;
            return;
        }

        if (active_order_->status != order::ORDER_STATUS::OPEN) return;

        if (direction_ == DIR::LONG) {
            // Stop-loss: bid has fallen too far below entry.
            if (bbo.bid_price > 0 && bbo.bid_price < entry_price_ - stop_loss_) {
                active_order_->cancel();
                active_order_ = nullptr;
                place_order(msg::SIDE::SELL, trade_qty_, bbo.bid_price);
                // Stay in EXITING — the new order will close us out.
            }
        } else {
            // Stop-loss for short: ask has risen too far above entry.
            if (bbo.ask_price > 0 && bbo.ask_price > entry_price_ + stop_loss_) {
                active_order_->cancel();
                active_order_ = nullptr;
                place_order(msg::SIDE::BUY, trade_qty_, bbo.ask_price);
            }
        }
    }

    /* ------------------------------------------------------------------ */

    void place_exit(const client::BidAndOffer &bbo) {
        if (direction_ == DIR::LONG) {
            const msg::price_t target =
                log::round_dn(std::max(bbo.bid_price, entry_price_ + take_profit_));
            place_order(msg::SIDE::SELL, trade_qty_, target);
        } else {
            const msg::price_t target =
                log::round_up(std::min(bbo.ask_price, entry_price_ - take_profit_));
            place_order(msg::SIDE::BUY, trade_qty_, target);
        }
    }

    void place_order(msg::SIDE side, msg::quantity_t qty, msg::price_t price) {
        orders_.push_back(
            std::make_unique<order::ExchangeOrder>(client_, 0, symbol_, side, qty, price));
        auto *raw = orders_.back().get();
        raw->commit();
        active_order_ = raw;
    }

    double compute_signal() const {
        const size_t quarter = std::max<size_t>(1, window_ / 4);
        const double recent_avg =
            std::accumulate(history_.end() - static_cast<ptrdiff_t>(quarter),
                            history_.end(), 0.0) /
            static_cast<double>(quarter);
        const double older_avg =
            std::accumulate(history_.begin(),
                            history_.begin() + static_cast<ptrdiff_t>(quarter), 0.0) /
            static_cast<double>(quarter);
        if (older_avg == 0.0) return 0.0;
        return (recent_avg - older_avg) / older_avg;
    }

    msg::price_t weighted_fill() const {
        if (!active_order_ || active_order_->fills.empty()) {
            return active_order_ ? active_order_->exchange_state.price : entry_price_;
        }
        int64_t value = 0, qty = 0;
        for (const auto &[px, q] : active_order_->fills) {
            value += static_cast<int64_t>(px) * q;
            qty   += q;
        }
        return qty > 0 ? static_cast<msg::price_t>(value / qty) : entry_price_;
    }
};

} // namespace hft::strategy::impl
