#pragma once

#include <iostream>
#include <memory>

#include <hft/ExchangeClient.h>
#include <hft/ExchangeOrder.h>
#include <hft/Strategy.h>
#include <hft/msg/common.h>
#include <hft/msg/market_data.h>

namespace hft::strategy::impl {

class Simple final : public Strategy {
private:
  enum class PHASE { BUY, WAIT_BUY_FILL, SELL, WAIT_SELL_FILL };

  client::ExchangeClient &client_;
  const msg::symbol_id_t target_symbol_;
  const msg::quantity_t scalp_qty_;

  std::vector<std::pair<msg::symbol_id_t, int16_t>> positions_{};
  std::vector<std::unique_ptr<order::ExchangeOrder>> orders_{};

  std::unique_ptr<order::ExchangeOrder> order_{};

  PHASE phase_ = PHASE::BUY;
  msg::price_t bought_at{};

public:
  /* Constructors */
  Simple(client::ExchangeClient &client, msg::symbol_id_t symbol,
         msg::quantity_t scalp_qty)
      : Strategy(client, client.get_orderbook(), client.get_position_tracker()),
        client_(client), target_symbol_(symbol), scalp_qty_(scalp_qty) {};
  ~Simple() = default;

  /* Strategy callbacks */
  void on_market_data(const msg::md::MdMessage &) override {}

  void on_tick() override {

    if (phase_ == PHASE::BUY) {
      if (!order_)
        delete order_.release();

      // Send initial order
      const auto bbo = client_.get_bbo(target_symbol_);
      const auto buy_price = bbo.ask_price - 10;
      order_ = std::make_unique<order::ExchangeOrder>(
          client_, 0, target_symbol_, msg::SIDE::BUY, scalp_qty_, buy_price);
      order_->commit();
      phase_ = PHASE::WAIT_BUY_FILL;

    } else if (phase_ == PHASE::WAIT_BUY_FILL) {
      // Wait for quantity to fill
      if (order_->fill_quantity == scalp_qty_) {
        bought_at = order_->exchange_state.price;
        phase_ = PHASE::SELL;
      } else if (order_->status == order::ORDER_STATUS::LOCAL_ONLY) {
        phase_ = PHASE::BUY;
      }

    } else if (phase_ == PHASE::SELL) {
      // Send sell order
      const auto bbo = client_.get_bbo(target_symbol_);
      const auto sell_price = std::max(bbo.bid_price, bought_at + 10);
      delete order_.release();

      order_ = std::make_unique<order::ExchangeOrder>(
          client_, 0, target_symbol_, msg::SIDE::SELL, scalp_qty_, sell_price);
      order_->commit();
      phase_ = PHASE::WAIT_SELL_FILL;

    } else if (phase_ == PHASE::WAIT_SELL_FILL) {
      // Wait for sell order to fill
      if (order_->fill_quantity == scalp_qty_) {
        const auto sold_at = order_->exchange_state.price;
        const auto profit = (sold_at - bought_at) * scalp_qty_;
        std::cout << "[STRAT: SIMPLE] Generated $" << profit << std::endl;

        phase_ = PHASE::BUY;
      } else if (order_->status == order::ORDER_STATUS::LOCAL_ONLY) {
        phase_ = PHASE::SELL;
      }
    }
  };
};

} // namespace hft::strategy::impl
