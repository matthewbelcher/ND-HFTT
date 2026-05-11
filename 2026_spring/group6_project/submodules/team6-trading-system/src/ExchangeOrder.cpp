#include "../include/hft/ExchangeOrder.h"

#include <cinttypes>

namespace hft::order {
/**
 * Handle an order acknowledgement from the exchange
 *
 * Synchronizes local and exchange state with the confirmed order details
 * and transitions status to OPEN.
 *
 * @param resp Acknowledgement response from exchange
 */
void ExchangeOrder::handle_ack(const msg::oe::OrderAckResponse &resp) {
  // Update exchange state
  this->exchange_state.order_id = resp.exch_order_id;
  this->exchange_state.symbol = local_state.symbol;
  this->exchange_state.side = local_state.side;
  this->exchange_state.quantity = resp.quantity;
  this->exchange_state.price = resp.price;
  this->exchange_state.flags = local_state.flags;

  // Update local state
  this->local_state.order_id = resp.order_id;
  this->local_state.quantity = resp.quantity;
  this->local_state.price = resp.price;

  this->status = ORDER_STATUS::OPEN;
}

/**
 * Handle a fill notification from the exchange
 *
 * Accumulates filled quantity and records the fill price in the fills map.
 * Transitions status to CLOSED if the fill flags indicate the order is fully
 * filled.
 *
 * @param resp Fill response from exchange
 */
void ExchangeOrder::handle_fill(const msg::oe::OrderFillResponse &resp) {
  this->fill_quantity += resp.quantity;

  if (this->fills.contains(resp.price)) {
    this->fills[resp.price] = this->fills.at(resp.price) + resp.quantity;
  } else {
    this->fills[resp.price] = resp.quantity;
  }

  if (resp.flags == msg::oe::FILL_FLAGS::CLOSED) {
    this->status = ORDER_STATUS::CLOSED;
  }
}

/**
 * Handle an order closed notification from the exchange
 *
 * Transitions status to CLOSED. Called when the exchange closes an order
 * without a fill (e.g. cancelled by exchange or expiry).
 *
 * @param resp Closed response from exchange (unused)
 */
void ExchangeOrder::handle_close(const msg::oe::OrderClosedResponse &) {
  this->status = ORDER_STATUS::CLOSED;
}

/**
 * Handle an order rejection from the exchange
 *
 * Logs the rejection reason to stderr, rolls back local state to match
 * the last confirmed exchange state, and restores status to OPEN.
 *
 * @param resp Rejection response from exchange
 */
void ExchangeOrder::handle_reject(const msg::oe::OrderRejectResponse &resp) {

  fprintf(stderr, "[ERROR] Order rejected (order_id=%" PRIu64 "): %s\n",
          (uint64_t)resp.order_id,
          msg::oe::reject_reason_string(resp.reject_reason));

  if (this->status == ORDER_STATUS::IN_FLIGHT ||
      resp.reject_reason == msg::oe::REJECT_REASON::UNKNOWN_ORDER_ID) {
    // New order was rejected before it was ever acknowledged, or the exchange
    // does not recognise the order ID — the order never existed (or was already
    // closed).  Mark it closed so the strategy slot is cleared.
    this->status = ORDER_STATUS::CLOSED;
  } else {
    // Modify or cancel was rejected — the order is still live on the exchange
    // at the last confirmed state.  Roll back local state and return to OPEN.
    this->local_state.side = this->exchange_state.side;
    this->local_state.quantity = this->exchange_state.quantity;
    this->local_state.price = this->exchange_state.price;
    this->status = ORDER_STATUS::OPEN;
  }
}

/**
 * Create and submit a new limit order to the exchange
 *
 * Registers the order with the client and immediately calls commit() to
 * send the initial new order request.
 *
 * @param client Exchange client used to submit requests
 * @param personal_order_id Caller-assigned order ID (0 to auto-assign)
 * @param symbol Symbol ID to trade
 * @param side Buy or sell side
 * @param quantity Order quantity
 * @param price Limit price
 * @param order_flags Optional order flags (default: NONE)
 */
ExchangeOrder::ExchangeOrder(client::ExchangeClient &client,
                             const msg::order_id_t personal_order_id,
                             const msg::symbol_id_t symbol,
                             const msg::SIDE side,
                             const msg::quantity_t quantity,
                             const msg::price_t price,
                             const msg::oe::ORDER_FLAGS order_flags)
    : client(client) {

  this->local_state.symbol = symbol;
  this->local_state.side = side;
  this->local_state.quantity = quantity;
  this->local_state.price = price;
  this->local_state.flags = order_flags;
  this->local_state.order_id =
      this->client.register_order(personal_order_id, *this);
}

/**
 * Destroy the ExchangeOrder, cancelling it if still live on the exchange
 *
 * Issues a cancel request for any order that has been sent but not yet
 * closed or already pending cancellation.
 */
ExchangeOrder::~ExchangeOrder() {
  if (this->status != ORDER_STATUS::LOCAL_ONLY &&
      this->status != ORDER_STATUS::CLOSED &&
      this->status != ORDER_STATUS::CANCEL_IN_FLIGHT) {
    this->cancel();
  }
}

/**
 * Submit the current local state to the exchange
 *
 * Sends a new order request if the order has never been submitted, or a
 * modify request if the order is already open. Throws if called on a
 * closed or cancel-in-flight order.
 *
 * @throws std::logic_error If the order is closed or cancel-in-flight
 */
bool ExchangeOrder::commit() {

  if (this->status == ORDER_STATUS::CLOSED ||
      this->status == ORDER_STATUS::CANCEL_IN_FLIGHT) {
    throw std::logic_error("Attempted to commit a closed order");

  } else if (this->status == ORDER_STATUS::LOCAL_ONLY) {
    // New order — risk check happens inside client.new_order()
    if (!this->client.new_order(this->local_state))
      return false;
    this->status = ORDER_STATUS::IN_FLIGHT;

  } else {
    msg::oe::ModifyOrderRequest modify_request{};
    modify_request.order_id = this->local_state.order_id;
    modify_request.side = this->local_state.side;
    modify_request.quantity = this->local_state.quantity;
    modify_request.price = this->local_state.price;
    if (!this->client.modify_order(modify_request))
      return false;
    this->status = ORDER_STATUS::UPDATE_IN_FLIGHT;
  }

  return true;
}

/**
 * Dispatch an exchange response to the appropriate handler
 *
 * Routes ACK, FILL, CLOSE, and REJECT messages to their respective
 * handle_* methods. Logs an error for unrecognized message types.
 *
 * @param response Response message received from the exchange
 */
void ExchangeOrder::on_response(const msg::oe::Response &response) {
  switch (response.header.msg_type) {
  case msg::oe::MSG_TYPE::ACK:
    this->handle_ack(response.body.order_ack_response);
    break;
  case msg::oe::MSG_TYPE::FILL:
    this->handle_fill(response.body.order_fill_response);
    break;
  case msg::oe::MSG_TYPE::CLOSE:
    this->handle_close(response.body.order_closed_response);
    break;
  case msg::oe::MSG_TYPE::REJECT:
    this->handle_reject(response.body.order_reject_response);
    break;
  default:
    fprintf(stderr, "[ERROR] Unknown message type: %" PRIu8 "\n",
            (uint8_t)response.header.msg_type);
  }
}

/**
 * Update order parameters and submit the modification to the exchange
 *
 * @param new_side Updated side (buy/sell)
 * @param new_quantity Updated order quantity
 * @param new_price Updated limit price
 */
void ExchangeOrder::modify(const msg::SIDE new_side,
                           const msg::quantity_t new_quantity,
                           const msg::price_t new_price) {
  const auto old_side = this->local_state.side;
  const auto old_quantity = this->local_state.quantity;
  const auto old_price = this->local_state.price;

  this->local_state.side = new_side;
  this->local_state.quantity = new_quantity;
  this->local_state.price = new_price;

  if (!this->commit()) {
    this->local_state.side = old_side;
    this->local_state.quantity = old_quantity;
    this->local_state.price = old_price;
  }
}

/**
 * Send a cancel request for this order to the exchange
 *
 * Transitions status to CANCEL_IN_FLIGHT immediately. The order is
 * fully closed once a CLOSE response is received via on_response().
 */
void ExchangeOrder::cancel() {
  const msg::oe::DeleteOrderRequest delete_request{
      .order_id = this->local_state.order_id};
  this->client.delete_order(delete_request);
  this->status = ORDER_STATUS::CANCEL_IN_FLIGHT;
}
} // namespace hft::order
