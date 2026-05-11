#include "matching_main.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>

namespace ghostbook::matching {

MatchingEngine::MatchingEngine(logical_clock_t start_clock)
    : logical_clock_(start_clock) {}

std::optional<ExecutionEvent>
MatchingEngine::submit(const NewOrderCommand &command) {
  return accept_new_order(command, false);
}

std::optional<ExecutionEvent>
MatchingEngine::accept_new_order(const NewOrderCommand &command,
                                 bool is_replace) {
  if (command.quantity == 0) {
    ExecutionEvent reject_event{
        .type = EventType::Reject,
        .order_id = command.client_order_id,
        .instrument_id = command.instrument_id,
        .side = command.side,
        .quantity = 0,
        .remaining_quantity = 0,
        .cumulative_quantity = 0,
        .price_tick = command.price_tick,
        .logical_clock = logical_clock_,
        .reason = "quantity must be > 0",
    };
    emit_event(reject_event);
    return reject_event;
  }

  if (!SYMBOLS.contains(command.instrument_id)) {
    ExecutionEvent reject_event{
        .type = EventType::Reject,
        .order_id = command.client_order_id,
        .instrument_id = command.instrument_id,
        .side = command.side,
        .quantity = 0,
        .remaining_quantity = 0,
        .cumulative_quantity = 0,
        .price_tick = command.price_tick,
        .logical_clock = logical_clock_,
        .reason = "unknown instrument",
    };
    emit_event(reject_event);
    return reject_event;
  }

  if (order_index_.contains(command.client_order_id)) {
    ExecutionEvent reject_event{
        .type = EventType::Reject,
        .order_id = command.client_order_id,
        .instrument_id = command.instrument_id,
        .side = command.side,
        .quantity = 0,
        .remaining_quantity = 0,
        .cumulative_quantity = 0,
        .price_tick = command.price_tick,
        .logical_clock = logical_clock_,
        .reason = "duplicate order id",
    };
    emit_event(reject_event);
    return reject_event;
  }

  if (command.order_type == OrderType::Limit && command.price_tick <= 0) {
    ExecutionEvent reject_event{
        .type = EventType::Reject,
        .order_id = command.client_order_id,
        .instrument_id = command.instrument_id,
        .side = command.side,
        .quantity = 0,
        .remaining_quantity = 0,
        .cumulative_quantity = 0,
        .price_tick = command.price_tick,
        .logical_clock = logical_clock_,
        .reason = "limit price must be > 0",
    };
    emit_event(reject_event);
    return reject_event;
  }

  if (command.order_type == OrderType::Market && command.price_tick != 0) {
    ExecutionEvent reject_event{
        .type = EventType::Reject,
        .order_id = command.client_order_id,
        .instrument_id = command.instrument_id,
        .side = command.side,
        .quantity = 0,
        .remaining_quantity = 0,
        .cumulative_quantity = 0,
        .price_tick = command.price_tick,
        .logical_clock = logical_clock_,
        .reason = "market order price must be 0",
    };
    emit_event(reject_event);
    return reject_event;
  }

  if (command.order_type == OrderType::Market &&
      command.tif == TimeInForce::PostOnly) {
    ExecutionEvent reject_event{
        .type = EventType::Reject,
        .order_id = command.client_order_id,
        .instrument_id = command.instrument_id,
        .side = command.side,
        .quantity = 0,
        .remaining_quantity = 0,
        .cumulative_quantity = 0,
        .price_tick = command.price_tick,
        .logical_clock = logical_clock_,
        .reason = "post-only requires limit order",
    };
    emit_event(reject_event);
    return reject_event;
  }

  if (command.tif == TimeInForce::PostOnly && crosses_top(command)) {
    ExecutionEvent reject_event{
        .type = EventType::Reject,
        .order_id = command.client_order_id,
        .instrument_id = command.instrument_id,
        .side = command.side,
        .quantity = 0,
        .remaining_quantity = 0,
        .cumulative_quantity = 0,
        .price_tick = command.price_tick,
        .logical_clock = logical_clock_,
        .reason = "post-only would cross",
    };
    emit_event(reject_event);
    return reject_event;
  }

  if (command.tif == TimeInForce::FOK &&
      executable_quantity(command) < command.quantity) {
    ExecutionEvent reject_event{
        .type = EventType::Reject,
        .order_id = command.client_order_id,
        .instrument_id = command.instrument_id,
        .side = command.side,
        .quantity = 0,
        .remaining_quantity = command.quantity,
        .cumulative_quantity = 0,
        .price_tick = command.price_tick,
        .logical_clock = logical_clock_,
        .reason = "fok not fully executable",
    };
    emit_event(reject_event);
    return reject_event;
  }

  Order taker_order{
      .order_id = command.client_order_id,
      .instrument_id = command.instrument_id,
      .side = command.side,
      .order_type = command.order_type,
      .tif = command.tif,
      .price_tick = command.price_tick,
      .original_quantity = command.quantity,
      .remaining_quantity = command.quantity,
      .entry_clock = logical_clock_,
  };

  ExecutionEvent ack_event{
      .type = EventType::Ack,
      .order_id = command.client_order_id,
      .instrument_id = command.instrument_id,
      .side = command.side,
      .quantity = command.quantity,
      .remaining_quantity = command.quantity,
      .cumulative_quantity = 0,
      .price_tick = command.price_tick,
      .logical_clock = logical_clock_,
      .reason = is_replace ? "replaced" : "accepted",
  };
  emit_event(ack_event);

  match_order(taker_order);

  const bool can_rest = taker_order.order_type == OrderType::Limit &&
                        taker_order.tif == TimeInForce::Day;
  if (taker_order.remaining_quantity > 0 && can_rest) {
    auto resting_order = std::make_unique<RestingOrder>();
    resting_order->order = taker_order;
    RestingOrder *resting_order_ptr = resting_order.get();
    order_index_.insert_or_assign(taker_order.order_id,
                                  std::move(resting_order));
    insert_resting_order(resting_order_ptr);
  }

  if (taker_order.remaining_quantity > 0 && !can_rest) {
    ExecutionEvent cancel_event{
        .type = EventType::Cancel,
        .order_id = taker_order.order_id,
        .instrument_id = taker_order.instrument_id,
        .side = taker_order.side,
        .quantity = taker_order.remaining_quantity,
        .remaining_quantity = 0,
        .cumulative_quantity =
            taker_order.original_quantity - taker_order.remaining_quantity,
        .price_tick = taker_order.price_tick,
        .logical_clock = logical_clock_,
        .reason = "unfilled remainder canceled",
    };
    emit_event(cancel_event);
  }

  return ack_event;
}

std::optional<ExecutionEvent>
MatchingEngine::cancel(const CancelOrderCommand &command) {
  auto *order_ptr = order_index_.find(command.target_order_id);
  if (order_ptr == nullptr) {
    ExecutionEvent reject_event{
        .type = EventType::Reject,
        .order_id = command.client_order_id,
        .instrument_id = 0,
        .side = Side::Buy,
        .quantity = 0,
        .remaining_quantity = 0,
        .cumulative_quantity = 0,
        .price_tick = 0,
        .logical_clock = logical_clock_,
        .reason = "target order not found",
    };
    emit_event(reject_event);
    return reject_event;
  }

  RestingOrder &resting_order = *(*order_ptr);
  Order &order = resting_order.order;
  const quantity_t cancel_size =
      command.cancel_quantity == 0
          ? order.remaining_quantity
          : std::min(order.remaining_quantity, command.cancel_quantity);

  order.remaining_quantity -= cancel_size;
  if (order.remaining_quantity == 0) {
    remove_order_from_book(resting_order);
    order_index_.erase(command.target_order_id);
  }

  ExecutionEvent cancel_event{
      .type = EventType::Cancel,
      .order_id = command.target_order_id,
      .instrument_id = order.instrument_id,
      .side = order.side,
      .quantity = cancel_size,
      .remaining_quantity = order.remaining_quantity,
      .cumulative_quantity = order.original_quantity - order.remaining_quantity,
      .price_tick = order.price_tick,
      .logical_clock = logical_clock_,
      .reason = "canceled",
  };
  emit_event(cancel_event);
  return cancel_event;
}

std::optional<ExecutionEvent>
MatchingEngine::modify(const ModifyOrderCommand &command) {
  auto *original_ptr = order_index_.find(command.original_order_id);
  if (original_ptr == nullptr) {
    ExecutionEvent reject_event{
        .type = EventType::Reject,
        .order_id = command.new_order_id,
        .instrument_id = 0,
        .side = Side::Buy,
        .quantity = 0,
        .remaining_quantity = 0,
        .cumulative_quantity = 0,
        .price_tick = command.new_price_tick,
        .logical_clock = logical_clock_,
        .reason = "original order not found",
    };
    emit_event(reject_event);
    return reject_event;
  }

  RestingOrder &original_resting_order = *(*original_ptr);

  if (command.new_quantity == 0) {
    ExecutionEvent reject_event{
        .type = EventType::Reject,
        .order_id = command.new_order_id,
        .instrument_id = original_resting_order.order.instrument_id,
        .side = original_resting_order.order.side,
        .quantity = 0,
        .remaining_quantity = 0,
        .cumulative_quantity = 0,
        .price_tick = command.new_price_tick,
        .logical_clock = logical_clock_,
        .reason = "new quantity must be > 0",
    };
    emit_event(reject_event);
    return reject_event;
  }

  if (command.new_order_id != command.original_order_id &&
      order_index_.contains(command.new_order_id)) {
    ExecutionEvent reject_event{
        .type = EventType::Reject,
        .order_id = command.new_order_id,
        .instrument_id = original_resting_order.order.instrument_id,
        .side = original_resting_order.order.side,
        .quantity = 0,
        .remaining_quantity = 0,
        .cumulative_quantity = 0,
        .price_tick = command.new_price_tick,
        .logical_clock = logical_clock_,
        .reason = "new order id already exists",
    };
    emit_event(reject_event);
    return reject_event;
  }

  const Order &original_order = original_resting_order.order;

  if (original_order.order_type == OrderType::Limit &&
      command.new_price_tick <= 0) {
    ExecutionEvent reject_event{
        .type = EventType::Reject,
        .order_id = command.new_order_id,
        .instrument_id = original_order.instrument_id,
        .side = original_order.side,
        .quantity = 0,
        .remaining_quantity = 0,
        .cumulative_quantity = 0,
        .price_tick = command.new_price_tick,
        .logical_clock = logical_clock_,
        .reason = "limit price must be > 0",
    };
    emit_event(reject_event);
    return reject_event;
  }

  if (original_order.order_type == OrderType::Market &&
      command.new_price_tick != 0) {
    ExecutionEvent reject_event{
        .type = EventType::Reject,
        .order_id = command.new_order_id,
        .instrument_id = original_order.instrument_id,
        .side = original_order.side,
        .quantity = 0,
        .remaining_quantity = 0,
        .cumulative_quantity = 0,
        .price_tick = command.new_price_tick,
        .logical_clock = logical_clock_,
        .reason = "market order price must be 0",
    };
    emit_event(reject_event);
    return reject_event;
  }

  NewOrderCommand replacement_command{
      .client_order_id = command.new_order_id,
      .instrument_id = original_order.instrument_id,
      .side = original_order.side,
      .order_type = original_order.order_type,
      .tif = original_order.tif,
      .price_tick = command.new_price_tick,
      .quantity = command.new_quantity,
  };

  if (replacement_command.order_type == OrderType::Market &&
      replacement_command.tif == TimeInForce::PostOnly) {
    ExecutionEvent reject_event{
        .type = EventType::Reject,
        .order_id = command.new_order_id,
        .instrument_id = original_order.instrument_id,
        .side = original_order.side,
        .quantity = 0,
        .remaining_quantity = 0,
        .cumulative_quantity = 0,
        .price_tick = command.new_price_tick,
        .logical_clock = logical_clock_,
        .reason = "post-only requires limit order",
    };
    emit_event(reject_event);
    return reject_event;
  }

  if (replacement_command.tif == TimeInForce::PostOnly &&
      crosses_top(replacement_command)) {
    ExecutionEvent reject_event{
        .type = EventType::Reject,
        .order_id = command.new_order_id,
        .instrument_id = original_order.instrument_id,
        .side = original_order.side,
        .quantity = 0,
        .remaining_quantity = 0,
        .cumulative_quantity = 0,
        .price_tick = command.new_price_tick,
        .logical_clock = logical_clock_,
        .reason = "post-only would cross",
    };
    emit_event(reject_event);
    return reject_event;
  }

  if (replacement_command.tif == TimeInForce::FOK &&
      executable_quantity(replacement_command) < replacement_command.quantity) {
    ExecutionEvent reject_event{
        .type = EventType::Reject,
        .order_id = command.new_order_id,
        .instrument_id = original_order.instrument_id,
        .side = original_order.side,
        .quantity = 0,
        .remaining_quantity = replacement_command.quantity,
        .cumulative_quantity = 0,
        .price_tick = command.new_price_tick,
        .logical_clock = logical_clock_,
        .reason = "fok not fully executable",
    };
    emit_event(reject_event);
    return reject_event;
  }

  remove_order_from_book(original_resting_order);
  order_index_.erase(command.original_order_id);

  return accept_new_order(replacement_command, true);
}

void MatchingEngine::insert_resting_order(RestingOrder *resting_order) {
  const Order &order = resting_order->order;
  auto &book = books_[order.instrument_id];
  if (order.side == Side::Buy) {
    auto &level_queue = book.bids[order.price_tick];
    level_queue.push_back(resting_order);
    resting_order->level_it = std::prev(level_queue.end());
    return;
  }

  auto &level_queue = book.asks[order.price_tick];
  level_queue.push_back(resting_order);
  resting_order->level_it = std::prev(level_queue.end());
}

bool MatchingEngine::remove_order_from_book(const RestingOrder &resting_order) {
  const Order &order = resting_order.order;
  auto *book_ptr = books_.find(order.instrument_id);
  if (book_ptr == nullptr) {
    return false;
  }

  auto &book = *book_ptr;
  if (order.side == Side::Buy) {
    auto level_it = book.bids.find(order.price_tick);
    if (level_it == book.bids.end()) {
      return false;
    }
    auto &level_queue = level_it->second;
    level_queue.erase(resting_order.level_it);
    if (level_queue.empty()) {
      book.bids.erase(level_it);
    }
    return true;
  }

  auto level_it = book.asks.find(order.price_tick);
  if (level_it == book.asks.end()) {
    return false;
  }
  auto &level_queue = level_it->second;
  level_queue.erase(resting_order.level_it);
  if (level_queue.empty()) {
    book.asks.erase(level_it);
  }
  return true;
}

bool MatchingEngine::crosses_top(const NewOrderCommand &command) const {
  auto *book_ptr = books_.find(command.instrument_id);
  if (book_ptr == nullptr) {
    return false;
  }

  const auto &book = *book_ptr;
  if (command.order_type == OrderType::Market) {
    return command.side == Side::Buy ? !book.asks.empty() : !book.bids.empty();
  }

  if (command.side == Side::Buy) {
    if (book.asks.empty()) {
      return false;
    }
    return book.asks.begin()->first <= command.price_tick;
  }

  if (book.bids.empty()) {
    return false;
  }
  return book.bids.begin()->first >= command.price_tick;
}

quantity_t
MatchingEngine::executable_quantity(const NewOrderCommand &command) const {
  auto *book_ptr = books_.find(command.instrument_id);
  if (book_ptr == nullptr) {
    return 0;
  }

  const auto &book = *book_ptr;
  std::uint64_t executable = 0;

  if (command.side == Side::Buy) {
    for (const auto &[price_tick, level_queue] : book.asks) {
      if (command.order_type == OrderType::Limit &&
          price_tick > command.price_tick) {
        break;
      }
      for (const RestingOrder *maker_resting_order : level_queue) {
        executable += maker_resting_order->order.remaining_quantity;
        if (executable >= command.quantity) {
          return command.quantity;
        }
      }
    }
    return static_cast<quantity_t>(
        std::min<std::uint64_t>(executable, command.quantity));
  }

  for (const auto &[price_tick, level_queue] : book.bids) {
    if (command.order_type == OrderType::Limit &&
        price_tick < command.price_tick) {
      break;
    }
    for (const RestingOrder *maker_resting_order : level_queue) {
      executable += maker_resting_order->order.remaining_quantity;
      if (executable >= command.quantity) {
        return command.quantity;
      }
    }
  }
  return static_cast<quantity_t>(
      std::min<std::uint64_t>(executable, command.quantity));
}

void MatchingEngine::match_order(
    Order &taker_order) { // consider extracting some logic here to avoid code
                          // duplication with modify and other funcs
  auto *book_ptr = books_.find(taker_order.instrument_id);
  if (book_ptr == nullptr) {
    return;
  }

  auto &book = *book_ptr;

  if (taker_order.side == Side::Buy) {
    auto level_it = book.asks.begin();
    while (level_it != book.asks.end() && taker_order.remaining_quantity > 0) {
      const price_tick_t level_price = level_it->first;
      if (taker_order.order_type == OrderType::Limit &&
          level_price > taker_order.price_tick) {
        break;
      }

      auto &level_queue = level_it->second;
      while (!level_queue.empty() && taker_order.remaining_quantity > 0) {
        RestingOrder *maker_resting_order = level_queue.front();
        Order &maker_order = maker_resting_order->order;
        const quantity_t fill_size = std::min(taker_order.remaining_quantity,
                                              maker_order.remaining_quantity);

        taker_order.remaining_quantity -= fill_size;
        maker_order.remaining_quantity -= fill_size;

        ExecutionEvent taker_fill{
            .type = EventType::Fill,
            .order_id = taker_order.order_id,
            .counterparty_order_id = maker_order.order_id,
            .instrument_id = taker_order.instrument_id,
            .side = taker_order.side,
            .quantity = fill_size,
            .remaining_quantity = taker_order.remaining_quantity,
            .cumulative_quantity =
                taker_order.original_quantity - taker_order.remaining_quantity,
            .price_tick = level_price,
            .logical_clock = logical_clock_,
            .reason = "taker fill",
        };
        emit_event(taker_fill);

        ExecutionEvent maker_fill{
            .type = EventType::Fill,
            .order_id = maker_order.order_id,
            .counterparty_order_id = taker_order.order_id,
            .instrument_id = maker_order.instrument_id,
            .side = maker_order.side,
            .quantity = fill_size,
            .remaining_quantity = maker_order.remaining_quantity,
            .cumulative_quantity =
                maker_order.original_quantity - maker_order.remaining_quantity,
            .price_tick = level_price,
            .logical_clock = logical_clock_,
            .reason = "maker fill",
        };
        emit_event(maker_fill);

        if (maker_order.remaining_quantity == 0) {
          level_queue.pop_front();
          order_index_.erase(maker_order.order_id);
        }
      }

      if (level_queue.empty()) {
        level_it = book.asks.erase(level_it);
      } else {
        ++level_it;
      }
    }
    return;
  }

  auto level_it = book.bids.begin();
  while (level_it != book.bids.end() && taker_order.remaining_quantity > 0) {
    const price_tick_t level_price = level_it->first;
    if (taker_order.order_type == OrderType::Limit &&
        level_price < taker_order.price_tick) {
      break;
    }

    auto &level_queue = level_it->second;
    while (!level_queue.empty() && taker_order.remaining_quantity > 0) {
      RestingOrder *maker_resting_order = level_queue.front();
      Order &maker_order = maker_resting_order->order;
      const quantity_t fill_size = std::min(taker_order.remaining_quantity,
                                            maker_order.remaining_quantity);

      taker_order.remaining_quantity -= fill_size;
      maker_order.remaining_quantity -= fill_size;

      ExecutionEvent taker_fill{
          .type = EventType::Fill,
          .order_id = taker_order.order_id,
          .counterparty_order_id = maker_order.order_id,
          .instrument_id = taker_order.instrument_id,
          .side = taker_order.side,
          .quantity = fill_size,
          .remaining_quantity = taker_order.remaining_quantity,
          .cumulative_quantity =
              taker_order.original_quantity - taker_order.remaining_quantity,
          .price_tick = level_price,
          .logical_clock = logical_clock_,
          .reason = "taker fill",
      };
      emit_event(taker_fill);

      ExecutionEvent maker_fill{
          .type = EventType::Fill,
          .order_id = maker_order.order_id,
          .counterparty_order_id = taker_order.order_id,
          .instrument_id = maker_order.instrument_id,
          .side = maker_order.side,
          .quantity = fill_size,
          .remaining_quantity = maker_order.remaining_quantity,
          .cumulative_quantity =
              maker_order.original_quantity - maker_order.remaining_quantity,
          .price_tick = level_price,
          .logical_clock = logical_clock_,
          .reason = "maker fill",
      };
      emit_event(maker_fill);

      if (maker_order.remaining_quantity == 0) {
        level_queue.pop_front();
        order_index_.erase(maker_order.order_id);
      }
    }

    if (level_queue.empty()) {
      level_it = book.bids.erase(level_it);
    } else {
      ++level_it;
    }
  }
}

void MatchingEngine::run_once() { ++logical_clock_; }

logical_clock_t MatchingEngine::logical_clock() const { return logical_clock_; }

std::vector<ExecutionEvent> MatchingEngine::drain_events() {
  std::vector<ExecutionEvent> drained;
  drained.swap(events_);
  return drained;
}

void MatchingEngine::emit_event(ExecutionEvent event) {
  events_.push_back(std::move(event));
}

} // namespace ghostbook::matching
