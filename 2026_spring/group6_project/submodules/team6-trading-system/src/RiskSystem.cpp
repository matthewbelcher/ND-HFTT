#include "hft/RiskSystem.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>

namespace hft::risk {

const char *reject_reason_string(const RejectReason reason) {
  switch (reason) {
  case RejectReason::NONE:
    return "NONE";
  case RejectReason::MAX_QTY_PER_ORDER:
    return "MAX_QTY_PER_ORDER";
  case RejectReason::MAX_QTY_PER_SIDE:
    return "MAX_QTY_PER_SIDE";
  case RejectReason::MAX_EXPOSURE:
    return "MAX_EXPOSURE";
  case RejectReason::INVALID_PRICE:
    return "INVALID_PRICE";
  case RejectReason::POSITION_LIMIT:
    return "POSITION_LIMIT";
  case RejectReason::MAX_ORDERS_PER_SECOND:
    return "MAX_ORDERS_PER_SECOND";
  case RejectReason::MAX_ORDERS_PER_MD_UPDATE:
    return "MAX_ORDERS_PER_MD_UPDATE";
  case RejectReason::MAX_INFLIGHT_ORDERS:
    return "MAX_INFLIGHT_ORDERS";
  case RejectReason::SHUTDOWN_IN_PROGRESS:
    return "SHUTDOWN_IN_PROGRESS";
  default:
    return "UNKNOWN";
  }
}

/* ── Constructor ───────────────────────────────────────────────────── */

RiskSystem::RiskSystem(RiskLimits limits, PositionTracker &tracker,
                       orderbook::MultiSymbolOrderBook &orderbook)
    : limits_(limits), tracker_(tracker), orderbook_(orderbook),
      last_second_reset_(std::chrono::steady_clock::now()) {}

/* ── Pre-trade evaluation: new order ───────────────────────────────── */

RiskCheckResult
RiskSystem::evaluate_order(const msg::oe::NewOrderRequest &req) {

  // Debug: dump order being evaluated
  msg::oe::Request debug_req{};
  debug_req.header.msg_type = msg::oe::MSG_TYPE::NEW_ORDER;
  debug_req.body.new_order_request = req;

  // Shutdown check
  if (shutdown_) {
    return {false, RejectReason::SHUTDOWN_IN_PROGRESS};
  }

  // Max qty per order
  if (req.quantity > limits_.max_qty_per_order) {
    return {false, RejectReason::MAX_QTY_PER_ORDER};
  }

  // Valid price
  if (req.price < limits_.min_valid_price ||
      req.price > limits_.max_valid_price ||
      (limits_.tick_size > 1 && req.price % limits_.tick_size != 0)) {
    return {false, RejectReason::INVALID_PRICE};
  }

  // Max qty per side - outstanding qty on the order's side
  const auto &state = tracker_.get_state(req.symbol);
  if (req.side == msg::SIDE::BUY) {
    if (state.outstanding_buy_qty + req.quantity > limits_.max_qty_per_side) {
      return {false, RejectReason::MAX_QTY_PER_SIDE};
    }
  } else {
    if (state.outstanding_sell_qty + req.quantity > limits_.max_qty_per_side) {
      return {false, RejectReason::MAX_QTY_PER_SIDE};
    }
  }

  // Max exposure per side
  if (req.side == msg::SIDE::BUY) {
    const int64_t new_exposure =
        tracker_.get_buy_exposure(req.symbol) + req.quantity;
    if (new_exposure > limits_.max_exposure_per_side) {
      return {false, RejectReason::MAX_EXPOSURE};
    }
  } else {
    const int64_t new_exposure =
        tracker_.get_sell_exposure(req.symbol) + req.quantity;
    if (new_exposure > limits_.max_exposure_per_side) {
      return {false, RejectReason::MAX_EXPOSURE};
    }
  }

  // Position limit — reject if filling this order could push the projected
  // position (fills + all outstanding orders + this order) past the limit.
  // Using exposure getters ensures we account for orders already in flight.
  if (req.side == msg::SIDE::BUY) {
    if (tracker_.get_buy_exposure(req.symbol) + static_cast<int64_t>(req.quantity) >
        limits_.max_position) {
      return {false, RejectReason::POSITION_LIMIT};
    }
  } else {
    if (tracker_.get_sell_exposure(req.symbol) + static_cast<int64_t>(req.quantity) >
        limits_.max_position) {
      return {false, RejectReason::POSITION_LIMIT};
    }
  }

  // Max orders per second
  maybe_reset_per_second_counter();
  if (orders_this_second_ >= limits_.max_orders_per_second) {
    return {false, RejectReason::MAX_ORDERS_PER_SECOND};
  }

  // Max orders per market data update
  if (orders_this_md_update_ >= limits_.max_orders_per_md_update) {
    return {false, RejectReason::MAX_ORDERS_PER_MD_UPDATE};
  }

  // Max un-acked (in-flight) orders
  if (inflight_count_ >= limits_.max_inflight_orders) {
    return {false, RejectReason::MAX_INFLIGHT_ORDERS};
  }

  // Track the symbol for shutdown monitoring
  known_symbols_.insert(req.symbol);

  return {true, RejectReason::NONE};
}

/* ── Pre-trade evaluation: modify order ────────────────────────────── */

RiskCheckResult RiskSystem::evaluate_modify(
    const msg::oe::ModifyOrderRequest &req, const msg::symbol_id_t symbol,
    const msg::SIDE current_side, const msg::quantity_t current_qty) {

  // Debug: dump modify order being evaluated
  if (shutdown_) {
    return {false, RejectReason::SHUTDOWN_IN_PROGRESS};
  }

  // Max qty per order
  if (req.quantity > limits_.max_qty_per_order) {
    return {false, RejectReason::MAX_QTY_PER_ORDER};
  }

  // Valid price
  if (req.price < limits_.min_valid_price ||
      req.price > limits_.max_valid_price ||
      (limits_.tick_size > 1 && req.price % limits_.tick_size != 0)) {
    return {false, RejectReason::INVALID_PRICE};
  }

  // For side/qty checks, compute the delta from current outstanding
  // The modify replaces (current_side, current_qty) with (req.side,
  // req.quantity)

  // Max qty per side — compute what outstanding would be after modify
  const auto &state = tracker_.get_state(symbol);
  uint64_t new_buy_outstanding = state.outstanding_buy_qty;
  uint64_t new_sell_outstanding = state.outstanding_sell_qty;

  // Remove old
  if (current_side == msg::SIDE::BUY) {
    new_buy_outstanding -=
        std::min(static_cast<uint64_t>(current_qty), new_buy_outstanding);
  } else {
    new_sell_outstanding -=
        std::min(static_cast<uint64_t>(current_qty), new_sell_outstanding);
  }

  // Add new
  if (req.side == msg::SIDE::BUY) {
    new_buy_outstanding += req.quantity;
  } else {
    new_sell_outstanding += req.quantity;
  }

  if (req.side == msg::SIDE::BUY &&
      new_buy_outstanding > limits_.max_qty_per_side) {
    return {false, RejectReason::MAX_QTY_PER_SIDE};
  }
  if (req.side == msg::SIDE::SELL &&
      new_sell_outstanding > limits_.max_qty_per_side) {
    return {false, RejectReason::MAX_QTY_PER_SIDE};
  }

  // Max exposure per side
  const int64_t position = tracker_.get_position(symbol);
  if (req.side == msg::SIDE::BUY) {
    const int64_t new_exposure =
        position + static_cast<int64_t>(new_buy_outstanding);
    if (new_exposure > limits_.max_exposure_per_side) {
      return {false, RejectReason::MAX_EXPOSURE};
    }
  } else {
    const int64_t new_exposure =
        -position + static_cast<int64_t>(new_sell_outstanding);
    if (new_exposure > limits_.max_exposure_per_side) {
      return {false, RejectReason::MAX_EXPOSURE};
    }
  }

  // Position limit — compute projected exposure after the modify is applied.
  // The old exposure was already removed from new_buy/sell_outstanding above.
  if (req.side == msg::SIDE::BUY) {
    const int64_t projected = position + static_cast<int64_t>(new_buy_outstanding);
    if (projected > limits_.max_position) {
      return {false, RejectReason::POSITION_LIMIT};
    }
  } else {
    const int64_t projected = -position + static_cast<int64_t>(new_sell_outstanding);
    if (projected > limits_.max_position) {
      return {false, RejectReason::POSITION_LIMIT};
    }
  }

  // Max orders per second
  maybe_reset_per_second_counter();
  if (orders_this_second_ >= limits_.max_orders_per_second) {
    return {false, RejectReason::MAX_ORDERS_PER_SECOND};
  }

  // Max orders per MD update
  if (orders_this_md_update_ >= limits_.max_orders_per_md_update) {
    return {false, RejectReason::MAX_ORDERS_PER_MD_UPDATE};
  }

  // Max inflight
  if (inflight_count_ >= limits_.max_inflight_orders) {
    return {false, RejectReason::MAX_INFLIGHT_ORDERS};
  }

  return {true, RejectReason::NONE};
}

/* ── Rate limiting / lifecycle ─────────────────────────────────────── */

void RiskSystem::on_md_update() { orders_this_md_update_ = 0; }

void RiskSystem::on_order_sent() {
  orders_this_second_++;
  orders_this_md_update_++;
  inflight_count_++;
}

void RiskSystem::on_order_acked() {
  if (inflight_count_ > 0)
    inflight_count_--;
}

void RiskSystem::on_order_closed() {
  if (inflight_count_ > 0)
    inflight_count_--;
}

/* ── Shutdown ──────────────────────────────────────────────────────── */

bool RiskSystem::check_shutdown_conditions() {
  if (shutdown_)
    return true;

  // Check PNL across all known symbols
  const double total_pnl = get_total_pnl();
  if (total_pnl < limits_.min_pnl_shutdown) {
    fprintf(stderr, "[RISK] SHUTDOWN: PNL %.2f < min %.2f\n", total_pnl,
            limits_.min_pnl_shutdown);
    shutdown_ = true;
    return true;
  }

  // Check absolute position for each known symbol
  for (const auto symbol : known_symbols_) {
    const int64_t pos = tracker_.get_position(symbol);
    if (std::abs(pos) > limits_.max_abs_position_shutdown) {
      fprintf(stderr,
              "[RISK] SHUTDOWN: |position| %" PRId64 " > max %" PRId64
              " for symbol %u\n",
              std::abs(pos), limits_.max_abs_position_shutdown, symbol);
      shutdown_ = true;
      return true;
    }
  }

  return false;
}

bool RiskSystem::is_shutdown() const { return shutdown_; }

/* ── Private helpers ───────────────────────────────────────────────── */

std::optional<msg::price_t>
RiskSystem::get_mid_price(const msg::symbol_id_t symbol) const {
  try {
    const auto best_bid =
        orderbook_.get_best_price_level(symbol, msg::SIDE::BUY);
    const auto best_ask =
        orderbook_.get_best_price_level(symbol, msg::SIDE::SELL);

    if (!best_bid.has_value() || !best_ask.has_value())
      return std::nullopt;

    return (best_bid->price + best_ask->price) / 2;
  } catch (const std::out_of_range &) {
    return std::nullopt;
  }
}

double RiskSystem::get_total_pnl() const {
  double total = 0.0;
  for (const auto symbol : known_symbols_) {
    const auto mid = get_mid_price(symbol);
    if (mid.has_value()) {
      total += tracker_.get_total_pnl(symbol, mid.value());
    } else {
      // No mid available — use realized PNL only (conservative)
      total += tracker_.get_realized_pnl(symbol);
    }
  }
  return total;
}

void RiskSystem::maybe_reset_per_second_counter() {
  const auto now = std::chrono::steady_clock::now();
  if (now - last_second_reset_ >= std::chrono::seconds(1)) {
    orders_this_second_ = 0;
    last_second_reset_ = now;
  }
}

} // namespace hft::risk
