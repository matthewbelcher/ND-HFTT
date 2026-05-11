#pragma once

#include <array>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <hft/ExchangeClient.h>
#include <hft/ExchangeOrder.h>
#include <hft/Strategy.h>
#include <hft/msg/common.h>

#include "Logger.h"

namespace hft::strategy::impl {

/**
 * ETF / basket arbitrage.
 *
 * Symbol 13 is an ETF equivalent to one lot of each of the 10 stocks
 * (symbols 3–12).  This strategy monitors the ETF mid-price against the
 * average mid-price of the basket.  When the dislocation exceeds a threshold
 * it enters a position in the ETF expecting reversion toward fair value.
 *
 * Only one active trade is held at a time.  Positions are closed once the
 * ETF mid reverts to within revert_threshold of the basket mid.  If the ETF
 * order does not fill at the posted price, it is repriced to track the
 * current BBO.
 */
class ETFArbitrage final : public Strategy {
public:
  ETFArbitrage(client::ExchangeClient &client, msg::price_t threshold,
               msg::quantity_t trade_qty, msg::price_t revert_threshold)
      : Strategy(client, client.get_orderbook(), client.get_position_tracker()),
        threshold_(threshold), trade_qty_(trade_qty),
        revert_threshold_(revert_threshold) {
    subscribe_md(ETF_SYMBOL);
    for (const auto sym : BASKET) subscribe_md(sym);
  }

  void on_market_data(msg::symbol_id_t, const msg::md::MdMessage &) override {
    const msg::price_t basket_mid = compute_basket_mid();
    if (basket_mid == 0)
      return;

    const auto etf = client_.get_bbo(ETF_SYMBOL);
    if (etf.bid_price == 0 || etf.ask_price == 0)
      return;

    switch (phase_) {
    case PHASE::FLAT:
      tick_flat(basket_mid, etf);
      break;
    case PHASE::LONG_ETF:
      tick_long(basket_mid, etf);
      break;
    case PHASE::SHORT_ETF:
      tick_short(basket_mid, etf);
      break;
    case PHASE::EXITING:
      tick_exiting(basket_mid, etf);
      break;
    }
  }

private:
  static constexpr msg::symbol_id_t ETF_SYMBOL = 13;
  static constexpr std::array<msg::symbol_id_t, 10> BASKET = {3, 4, 5,  6,  7,
                                                              8, 9, 10, 11, 12};

  enum class PHASE { FLAT, LONG_ETF, SHORT_ETF, EXITING };

  const msg::price_t threshold_;
  const msg::quantity_t trade_qty_;
  const msg::price_t revert_threshold_;

  PHASE phase_ = PHASE::FLAT;

  msg::price_t entry_price_ = 0;
  PHASE entry_dir_ = PHASE::FLAT; // LONG_ETF or SHORT_ETF

  order::ExchangeOrder *active_order_ = nullptr;
  std::vector<std::unique_ptr<order::ExchangeOrder>> orders_;

  /* ------------------------------------------------------------------ */

  void tick_flat(msg::price_t basket_mid, const client::BidAndOffer &etf) {
    const int64_t pos = tracker_.get_position(ETF_SYMBOL);

    // ETF is cheap vs basket → buy ETF.
    if (etf.ask_price < basket_mid - threshold_ &&
        pos + static_cast<int64_t>(trade_qty_) <= 10) {
      entry_price_ = etf.ask_price;
      entry_dir_ = PHASE::LONG_ETF;
      place_order(msg::SIDE::BUY, trade_qty_, etf.ask_price);
      phase_ = PHASE::LONG_ETF;
      log::log_action(ETF_SYMBOL, "ARB ENTRY LONG basket_mid=" +
                                      std::to_string(basket_mid) + " etf_ask=" +
                                      std::to_string(etf.ask_price));
      // ETF is expensive vs basket → sell ETF.
    } else if (etf.bid_price > basket_mid + threshold_ &&
               pos - static_cast<int64_t>(trade_qty_) >= -10) {
      entry_price_ = etf.bid_price;
      entry_dir_ = PHASE::SHORT_ETF;
      place_order(msg::SIDE::SELL, trade_qty_, etf.bid_price);
      phase_ = PHASE::SHORT_ETF;
      log::log_action(ETF_SYMBOL, "ARB ENTRY SHORT basket_mid=" +
                                      std::to_string(basket_mid) + " etf_bid=" +
                                      std::to_string(etf.bid_price));
    }
  }

  void tick_long(msg::price_t basket_mid, const client::BidAndOffer &etf) {
    if (!active_order_) {
      phase_ = PHASE::FLAT;
      return;
    }

    if (active_order_->status == order::ORDER_STATUS::LOCAL_ONLY) {
      active_order_ = nullptr;
      phase_ = PHASE::FLAT;
      return;
    }

    // Entry filled → wait for reversion then exit.
    if (active_order_->fill_quantity > 0 &&
        (active_order_->fill_quantity >= trade_qty_ ||
         active_order_->status == order::ORDER_STATUS::CLOSED)) {
      entry_price_ = weighted_fill();
      active_order_ = nullptr;
      // Check reversion immediately; otherwise wait in this phase.
      const msg::price_t etf_mid = (etf.bid_price + etf.ask_price) / 2;
      if (etf_mid >= basket_mid - revert_threshold_) {
        place_order(msg::SIDE::SELL, trade_qty_, etf.bid_price);
        phase_ = PHASE::EXITING;
      }
      // else stay LONG_ETF with no active_order_ — next tick will try exit.
      return;
    }

    if (active_order_->status == order::ORDER_STATUS::CLOSED) {
      // Closed without fill (e.g. cancelled).
      active_order_ = nullptr;
      phase_ = PHASE::FLAT;
      return;
    }

    // No fill yet: check for reversion while entry is still live.
    const msg::price_t etf_mid = (etf.bid_price + etf.ask_price) / 2;
    if (etf_mid >= basket_mid - revert_threshold_ &&
        active_order_->status == order::ORDER_STATUS::OPEN) {
      // Opportunity closed — cancel entry.
      active_order_->cancel();
      return;
    }

    // Reprice entry to track current ask.
    if (active_order_->status == order::ORDER_STATUS::OPEN &&
        active_order_->local_state.price != etf.ask_price) {
      active_order_->modify(msg::SIDE::BUY, trade_qty_, etf.ask_price);
    }
  }

  void tick_short(msg::price_t basket_mid, const client::BidAndOffer &etf) {
    if (!active_order_) {
      phase_ = PHASE::FLAT;
      return;
    }

    if (active_order_->status == order::ORDER_STATUS::LOCAL_ONLY) {
      active_order_ = nullptr;
      phase_ = PHASE::FLAT;
      return;
    }

    if (active_order_->fill_quantity > 0 &&
        (active_order_->fill_quantity >= trade_qty_ ||
         active_order_->status == order::ORDER_STATUS::CLOSED)) {
      entry_price_ = weighted_fill();
      active_order_ = nullptr;
      const msg::price_t etf_mid = (etf.bid_price + etf.ask_price) / 2;
      if (etf_mid <= basket_mid + revert_threshold_) {
        place_order(msg::SIDE::BUY, trade_qty_, etf.ask_price);
        phase_ = PHASE::EXITING;
      }
      return;
    }

    if (active_order_->status == order::ORDER_STATUS::CLOSED) {
      active_order_ = nullptr;
      phase_ = PHASE::FLAT;
      return;
    }

    const msg::price_t etf_mid = (etf.bid_price + etf.ask_price) / 2;
    if (etf_mid <= basket_mid + revert_threshold_ &&
        active_order_->status == order::ORDER_STATUS::OPEN) {
      active_order_->cancel();
      return;
    }

    if (active_order_->status == order::ORDER_STATUS::OPEN &&
        active_order_->local_state.price != etf.bid_price) {
      active_order_->modify(msg::SIDE::SELL, trade_qty_, etf.bid_price);
    }
  }

  void tick_exiting(msg::price_t, const client::BidAndOffer &etf) {
    if (!active_order_) {
      // May need to re-place if it was cleared without a fill.
      phase_ = PHASE::FLAT;
      return;
    }

    if (active_order_->fill_quantity >= trade_qty_ ||
        active_order_->status == order::ORDER_STATUS::CLOSED) {
      const msg::price_t exit_px = weighted_fill();
      const int32_t pnl =
          (entry_dir_ == PHASE::LONG_ETF ? exit_px - entry_price_
                                         : entry_price_ - exit_px) *
          static_cast<int32_t>(active_order_->fill_quantity);
      log::log_action(ETF_SYMBOL, "ARB EXIT pnl=" + std::to_string(pnl));
      active_order_ = nullptr;
      phase_ = PHASE::FLAT;
      return;
    }

    if (active_order_->status != order::ORDER_STATUS::OPEN)
      return;

    // Reprice exit to current best price.
    const msg::price_t desired =
        (entry_dir_ == PHASE::LONG_ETF) ? etf.bid_price : etf.ask_price;
    if (desired > 0 && active_order_->local_state.price != desired) {
      const msg::SIDE exit_side =
          (entry_dir_ == PHASE::LONG_ETF) ? msg::SIDE::SELL : msg::SIDE::BUY;
      active_order_->modify(exit_side, trade_qty_, desired);
    }
  }

  /* ------------------------------------------------------------------ */

  msg::price_t compute_basket_mid() const {
    int64_t sum = 0;
    for (const auto sym : BASKET) {
      const auto bbo = client_.get_bbo(sym);
      if (bbo.bid_price == 0 || bbo.ask_price == 0)
        return 0;
      sum += bbo.bid_price + bbo.ask_price;
    }
    // sum = Σ(bid_i + ask_i); divide by 2*10 = 20 to get average mid.
    return static_cast<msg::price_t>(sum / 20);
  }

  void place_order(msg::SIDE side, msg::quantity_t qty, msg::price_t price) {
    orders_.push_back(std::make_unique<order::ExchangeOrder>(
        client_, 0, ETF_SYMBOL, side, qty, price));
    auto *raw = orders_.back().get();
    raw->commit();
    active_order_ = raw;
  }

  msg::price_t weighted_fill() const {
    if (!active_order_ || active_order_->fills.empty()) {
      return active_order_ ? active_order_->exchange_state.price : entry_price_;
    }
    int64_t value = 0, qty = 0;
    for (const auto &[px, q] : active_order_->fills) {
      value += static_cast<int64_t>(px) * q;
      qty += q;
    }
    return qty > 0 ? static_cast<msg::price_t>(value / qty) : entry_price_;
  }
};

} // namespace hft::strategy::impl
