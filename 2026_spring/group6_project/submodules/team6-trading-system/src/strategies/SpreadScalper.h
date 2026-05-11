#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include <hft/ExchangeClient.h>
#include <hft/ExchangeOrder.h>
#include <hft/Strategy.h>
#include <hft/msg/common.h>

#include "Logger.h"

namespace hft::strategy::impl {

/**
 * Spread-aware scalper.
 *
 * Only enters when the quoted spread is wide enough to leave room for a profit
 * after crossing.  The buy limit is placed one tick inside the spread
 * (bid + tick) rather than far below ask.  The entry is repriced on every tick
 * to track the moving BBO, and cancelled if the opportunity disappears or the
 * market does not fill us within MAX_WAIT ticks.  The exit targets a
 * configurable profit; a stop-loss forces an exit at the current bid if price
 * moves against us.
 */
class SpreadScalper final : public Strategy {
public:
  SpreadScalper(client::ExchangeClient &client, msg::symbol_id_t symbol,
                msg::quantity_t scalp_qty, msg::price_t min_spread,
                msg::price_t target_profit, msg::price_t stop_loss_per_trade)
      : Strategy(client, client.get_orderbook(), client.get_position_tracker()),
        symbol_(symbol), scalp_qty_(scalp_qty), min_spread_(min_spread),
        target_profit_(target_profit), stop_loss_(stop_loss_per_trade) {
    subscribe_md(symbol_);
  }

  void on_market_data(msg::symbol_id_t, const msg::md::MdMessage &) override {
    switch (phase_) {
    case PHASE::IDLE:
      tick_idle();
      break;
    case PHASE::WAIT_BUY:
      tick_wait_buy();
      break;
    case PHASE::SELL:
      tick_sell();
      break;
    case PHASE::WAIT_SELL:
      tick_wait_sell();
      break;
    }
  }

private:
  static constexpr uint32_t MAX_WAIT = 500;
  static constexpr uint32_t REPRICE_INTERVAL = 100;

  enum class PHASE { IDLE, WAIT_BUY, SELL, WAIT_SELL };

  const msg::symbol_id_t symbol_;
  const msg::quantity_t scalp_qty_;
  const msg::price_t min_spread_;
  const msg::price_t target_profit_;
  const msg::price_t stop_loss_;

  PHASE phase_ = PHASE::IDLE;

  msg::price_t entry_price_ = 0;
  uint32_t wait_ticks_ = 0;
  uint32_t ticks_since_reprice_ = 0;

  order::ExchangeOrder *order_ = nullptr;
  // Graveyard: ExchangeClient holds raw pointers into this vector.
  std::vector<std::unique_ptr<order::ExchangeOrder>> orders_;

  /* ------------------------------------------------------------------ */

  void tick_idle() {
    const auto bbo = client_.get_bbo(symbol_);
    if (bbo.bid_price == 0 || bbo.ask_price == 0)
      return;

    const msg::price_t spread = bbo.ask_price - bbo.bid_price;
    if (spread < min_spread_)
      return;

    // Guard position limit before buying.
    if (tracker_.get_position(symbol_) + static_cast<int64_t>(scalp_qty_) > 10)
      return;

    const msg::price_t entry = log::round_dn(bbo.bid_price + 5);
    place_order(msg::SIDE::BUY, scalp_qty_, entry);
    wait_ticks_ = 0;
    phase_ = PHASE::WAIT_BUY;
  }

  void tick_wait_buy() {
    if (!order_) {
      phase_ = PHASE::IDLE;
      return;
    }

    if (order_->status == order::ORDER_STATUS::LOCAL_ONLY) {
      phase_ = PHASE::IDLE;
      return;
    }

    // Full fill → move to exit leg.
    if (order_->fill_quantity >= scalp_qty_ ||
        order_->status == order::ORDER_STATUS::CLOSED) {
      if (order_->fill_quantity > 0) {
        entry_price_ = weighted_fill_price();
        order_ = nullptr;
        phase_ = PHASE::SELL;
      } else {
        order_ = nullptr;
        phase_ = PHASE::IDLE;
      }
      return;
    }

    ++wait_ticks_;
    const auto bbo = client_.get_bbo(symbol_);
    const msg::price_t spread = bbo.ask_price - bbo.bid_price;

    // Abandon if market no longer offers enough spread.
    if (spread < min_spread_ && order_->status == order::ORDER_STATUS::OPEN) {
      order_->cancel();
      order_ = nullptr;
      phase_ = PHASE::IDLE;
      return;
    }

    // Timeout — cancel stale entry.
    if (wait_ticks_ >= MAX_WAIT &&
        order_->status == order::ORDER_STATUS::OPEN) {
      order_->cancel();
      order_ = nullptr;
      phase_ = PHASE::IDLE;
      return;
    }

    // Reprice to track the BBO, but throttle to once every REPRICE_INTERVAL
    // ticks.
    ++ticks_since_reprice_;
    if (order_->status == order::ORDER_STATUS::OPEN && bbo.bid_price > 0 &&
        ticks_since_reprice_ >= REPRICE_INTERVAL) {
      const msg::price_t desired = log::round_dn(bbo.bid_price + 5);
      if (desired != order_->local_state.price) {
        order_->modify(msg::SIDE::BUY, scalp_qty_, desired);
        ticks_since_reprice_ = 0;
      }
    }
  }

  void tick_sell() {
    const auto bbo = client_.get_bbo(symbol_);
    if (bbo.bid_price == 0)
      return;

    const msg::price_t target =
        log::round_dn(std::max(bbo.bid_price, entry_price_ + target_profit_));
    place_order(msg::SIDE::SELL, scalp_qty_, target);
    phase_ = PHASE::WAIT_SELL;
  }

  void tick_wait_sell() {
    if (!order_) {
      phase_ = PHASE::SELL;
      return;
    }

    if (order_->status == order::ORDER_STATUS::LOCAL_ONLY) {
      phase_ = PHASE::SELL;
      return;
    }

    if (order_->fill_quantity >= scalp_qty_ ||
        order_->status == order::ORDER_STATUS::CLOSED) {
      if (order_->fill_quantity > 0) {
        const msg::price_t sold_at = weighted_fill_price();
        log::log_action(symbol_,
                        "ROUND TRIP pnl=" +
                            std::to_string((sold_at - entry_price_) *
                                           static_cast<int32_t>(scalp_qty_)));
      }
      order_ = nullptr;
      phase_ = PHASE::IDLE;
      return;
    }

    // Stop-loss: market moved against us — take the current bid.
    if (order_->status == order::ORDER_STATUS::OPEN) {
      const auto bbo = client_.get_bbo(symbol_);
      if (bbo.bid_price > 0 && bbo.bid_price < entry_price_ - stop_loss_) {
        order_->cancel();
        // Reset to SELL so next tick re-places at current bid.
        order_ = nullptr;
        phase_ = PHASE::SELL;
      }
    }
  }

  /* ------------------------------------------------------------------ */

  void place_order(msg::SIDE side, msg::quantity_t qty, msg::price_t price) {
    orders_.push_back(std::make_unique<order::ExchangeOrder>(
        client_, 0, symbol_, side, qty, price));
    auto *raw = orders_.back().get();
    raw->commit();
    order_ = raw;
    ticks_since_reprice_ = 0;
  }

  msg::price_t weighted_fill_price() const {
    if (!order_ || order_->fills.empty()) {
      return order_ ? order_->exchange_state.price : entry_price_;
    }
    int64_t value = 0;
    int64_t qty = 0;
    for (const auto &[price, q] : order_->fills) {
      value += static_cast<int64_t>(price) * q;
      qty += q;
    }
    return qty > 0 ? static_cast<msg::price_t>(value / qty) : entry_price_;
  }
};

} // namespace hft::strategy::impl
