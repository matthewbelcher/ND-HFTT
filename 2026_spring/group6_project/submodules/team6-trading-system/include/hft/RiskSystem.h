/*
 * Pre-trade risk management system
 */

#ifndef HFT_RISK_SYSTEM_H
#define HFT_RISK_SYSTEM_H

#include <chrono>
#include <cstdint>
#include <optional>
#include <unordered_set>
#include <vector>

#include "PositionTracker.h"
#include "msg/order_entry.h"
#include "orderbook.h"

namespace hft::risk {

/* Configuration */

struct RiskLimits {
  msg::quantity_t max_qty_per_order = 1000;
  msg::quantity_t max_qty_per_side = 5000;
  int64_t max_exposure_per_side = 10000;
  int64_t max_position = 5000;
  int64_t max_abs_position_shutdown = 5000;
  double min_pnl_shutdown = -100000.0;
  uint32_t max_orders_per_second = 50;
  uint32_t max_orders_per_md_update = 5;
  uint32_t max_inflight_orders = 10;
  msg::price_t min_valid_price = 1;
  msg::price_t max_valid_price = 999999;
  msg::price_t tick_size = 5;
};

/* Result types */

enum class RejectReason : uint8_t {
  NONE = 0,
  MAX_QTY_PER_ORDER,
  MAX_QTY_PER_SIDE,
  MAX_EXPOSURE,
  INVALID_PRICE,
  POSITION_LIMIT,
  MAX_ORDERS_PER_SECOND,
  MAX_ORDERS_PER_MD_UPDATE,
  MAX_INFLIGHT_ORDERS,
  SHUTDOWN_IN_PROGRESS,
};

const char *reject_reason_string(RejectReason reason);

struct RiskCheckResult {
  bool accepted;
  RejectReason reason = RejectReason::NONE;
};

/* Class Definition */

/**
 * Evaluates all outgoing orders against configurable risk limits.
 * Hooks into PositionTracker for position/exposure data and into
 * the orderbook for mark-to-market PNL.
 */
class RiskSystem {
public:
  RiskSystem(RiskLimits limits, PositionTracker &tracker,
             orderbook::MultiSymbolOrderBook &orderbook);

  /* Pre-trade evaluation — called by ExchangeClient before sending */

  RiskCheckResult evaluate_order(const msg::oe::NewOrderRequest &req);
  RiskCheckResult evaluate_modify(const msg::oe::ModifyOrderRequest &req,
                                  msg::symbol_id_t symbol,
                                  msg::SIDE current_side,
                                  msg::quantity_t current_qty);

  /* Rate limiting / lifecycle notifications */

  void on_md_update();
  void on_order_sent();
  void on_order_acked();
  void on_order_closed();

  /* Shutdown logic */

  bool check_shutdown_conditions();
  [[nodiscard]] bool is_shutdown() const;

  /* Accessors */

  [[nodiscard]] const RiskLimits &get_limits() const { return limits_; }
  PositionTracker &get_tracker() { return tracker_; }

private:
  RiskLimits limits_;
  PositionTracker &tracker_;
  orderbook::MultiSymbolOrderBook &orderbook_;

  uint32_t orders_this_second_ = 0;
  uint32_t orders_this_md_update_ = 0;
  uint32_t inflight_count_ = 0;
  bool shutdown_ = false;

  std::chrono::steady_clock::time_point last_second_reset_;

  // Symbols we've seen (for shutdown position checks across all symbols)
  std::unordered_set<msg::symbol_id_t> known_symbols_;

  std::optional<msg::price_t> get_mid_price(msg::symbol_id_t symbol) const;
  double get_total_pnl() const;

  void maybe_reset_per_second_counter();
};

} // namespace hft::risk

#endif // HFT_RISK_SYSTEM_H
