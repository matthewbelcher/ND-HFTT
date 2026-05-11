#include "../include/hft/orderbook.h"
#include "hft/msg/market_data.h"

#include <bitset>
#include <cerrno>
#include <cinttypes>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace hft::orderbook {

/* Class Implementations */

/**
 * Create a new book to track the given side of the given symbol
 *
 * @param symbol Symbol number to track
 * @param side Contract side to track
 */
SideOrderBook::SideOrderBook(const uint32_t symbol, const msg::SIDE side)
    : symbol(symbol), side(side), total_volume_traded(0) {}

/**
 * Apply a message from the exchange to the order book
 *
 * @param message Exchange message structure
 */
bool SideOrderBook::apply_message(const msg::md::MdMessage &message) {
  bool success = false;
  const auto msg_type = static_cast<msg::md::MSG_TYPE>(message.header.msg_type);
  switch (msg_type) {
  case msg::md::MSG_TYPE::NEW_ORDER:
    success = this->insert_order(message.body.new_order);
    break;
  case msg::md::MSG_TYPE::DELETE_ORDER:
    success = this->delete_order(message.body.delete_order);
    break;
  case msg::md::MSG_TYPE::TRADE_SUMMARY:
    success = this->apply_trade_summary(message.body.trade_summary);
    break;
  default:;
  }

  return success;
}

/**
 * Get the top price level iterator
 *
 * @return Best price level pointer (if exists)
 */
std::optional<PriceLevelPtr> SideOrderBook::get_best_price_level_ptr() {
  // Skip stale entries (lazy deletion)
  while (!this->price_heap.empty() &&
         !this->price_heap.top().price_level->quantity) {
    this->price_heap.pop();
  }

  if (this->price_heap.empty()) {
    return std::nullopt;
  }
  return this->price_heap.top().price_level;
}

/**
 * Get the top price level structure in the orderbook
 *
 * @return Best price level structure
 */
[[nodiscard]] std::optional<PriceLevel> SideOrderBook::get_best_price_level() {
  if (const auto pl_iter = this->get_best_price_level_ptr();
      pl_iter.has_value()) {
    return *pl_iter.value();
  }
  return std::nullopt;
}

/**
 * Remove a price level iterator from all data structures in the orderbook
 *
 * @param price_level Price level pointer to delete from order book
 */
void SideOrderBook::delete_price_level(const PriceLevelPtr &price_level) {
  if (price_level->quantity != 0) {
    throw std::runtime_error("Attempted to delete non-empty price level");
  }

  // Delete from data structures (heap entries cleaned up lazily)
  this->price_map.erase(price_level->price);
}

/**
 * Add a new order from the exchange to the order book
 *
 * @param new_order order structure to add to order book
 * @return `true` if insertion succeeded
 */
bool SideOrderBook::insert_order(const msg::md::NewOrder &new_order) {
  // Ensure order attributes are correct
  if (new_order.symbol != this->symbol || new_order.side != this->side) {
    return false;
  }

  if (this->order_id_map.contains(new_order.order_id)) {
    // The replay channel can re-broadcast live frames that also appear in
    // the live backlog; skip the duplicate rather than crash.
    return false;
  }

  const int32_t price = new_order.price;
  const uint32_t quantity = new_order.quantity;

  if (this->price_map.contains(price)) {
    // PriceLevel already exists... update
    const auto price_level = this->price_map.at(price);
    price_level->quantity += quantity;
    price_level->num_orders++;

    price_level->orders.push_back(new_order);
    this->order_id_map[new_order.order_id] =
        std::prev(price_level->orders.end());
  } else {
    // PriceLevel must be added to heap and hashmaps

    const auto price_level = std::make_shared<PriceLevel>();
    price_level->price = price;
    price_level->quantity = quantity;
    price_level->num_orders = 1;

    const auto priority = (this->side == msg::SIDE::BUY) ? price : -price;

    price_heap.push({priority, price, price_level});
    price_map[price] = price_level;

    price_level->orders.push_back(new_order);
    this->order_id_map[new_order.order_id] =
        std::prev(price_level->orders.end());
  }
  return true;
}

/**
 * Update the order book to reflect a new trade
 *
 * @param trade_summary trade summary structure from exchange
 * @return `true` if update succeeded
 * @throws std::runtime_error if trade summary can't be executed on orderbook
 */
bool SideOrderBook::apply_trade_summary(
    const msg::md::TradeSummary &trade_summary) {
  // Ensure trade is from contracts of the opposite side
  if (trade_summary.symbol != this->symbol ||
      trade_summary.aggressor_side == this->side) {
    return false;
  }

  const auto total_quantity = trade_summary.total_quantity;
  // const auto last_price = trade_summary.last_price;

  auto quantity_remaining = total_quantity;
  while (quantity_remaining > 0) {
    const auto opt_price_level = this->get_best_price_level_ptr();
    if (!opt_price_level.has_value()) {
      throw std::runtime_error(
          "Attempted to apply trade summary on empty orderbook");
    }
    const auto &price_level = opt_price_level.value();

    // Iterate over orders at this price level
    auto order = price_level->orders.begin();
    while (order != price_level->orders.end() && quantity_remaining > 0) {
      if (quantity_remaining >= order->quantity) {
        // This order gets filled - delete from data structures
        quantity_remaining -= order->quantity;
        price_level->quantity -= order->quantity;
        order_id_map.erase(order->order_id);
        price_level->num_orders -= 1;
        // Advance the order iterator
        order = price_level->orders.erase(order);
      } else {
        // This order is not filled completely
        order->quantity -= quantity_remaining;
        price_level->quantity -= quantity_remaining;
        quantity_remaining = 0;
        ++order;
      }
    }

    // Handle case if price level is cleared
    if (!price_level->quantity) {
      // This price level is eliminated by the trade - delete from data
      // structures
      this->delete_price_level(price_level);
      this->price_heap.pop();
    }
  }

  this->total_volume_traded += trade_summary.total_quantity;
  return true;
}

/**
 * Delete an order from the order book data structures
 *
 * @param delete_order delete order struct from exchange
 * @return `true` if deletion succeeded
 */
bool SideOrderBook::delete_order(const msg::md::DeleteOrder &delete_order) {
  // Verify deletion request is for this book
  if (!this->order_id_map.contains(delete_order.order_id)) {
    return false;
  }

  const auto order = this->order_id_map[delete_order.order_id];
  const auto price_level = this->price_map[order->price];
  price_level->quantity -= order->quantity;
  price_level->orders.erase(order);
  price_level->num_orders -= 1;

  if (!price_level->quantity) {
    this->delete_price_level(price_level);
  }
  this->order_id_map.erase(delete_order.order_id);
  return true;
}

/**
 * Create a new book to track the given symbol
 *
 * @param symbol Symbol to track
 */
SymbolOrderBook::SymbolOrderBook(const uint32_t symbol)
    : side_order_books({SideOrderBook(symbol, msg::SIDE::BUY),
                        SideOrderBook(symbol, msg::SIDE::SELL)}),
      symbol(symbol) {}

/**
 * Apply an exchange update message to the book
 *
 * @param message Exchange message struct to apply
 * @return `true` if the operation was successful
 */
bool SymbolOrderBook::apply_message(const msg::md::MdMessage &message) {

  bool result;
  switch (message.header.msg_type) {
  case msg::md::MSG_TYPE::MODIFY_ORDER:
    result = this->modify_order(message.body.modify_order);
    break;
  default:
    result = this->side_order_books.first.apply_message(message) ||
             this->side_order_books.second.apply_message(message);
  }
  return result;
}

std::optional<PriceLevel>
SymbolOrderBook::get_best_price_level(const msg::SIDE side) {
  if (side == msg::SIDE::BUY) {
    return this->side_order_books.first.get_best_price_level();
  }
  return this->side_order_books.second.get_best_price_level();
}

/**
 * Get the total volume of trades
 *
 * @return Total volume of trades tracked by orderbook
 */
size_t SymbolOrderBook::get_total_volume_traded() const {
  return this->side_order_books.first.total_volume_traded +
         this->side_order_books.second.total_volume_traded;
}

/**
 * Synchronize the orderbook with exchange snapshot
 *
 * @param replay_seq List of snapshot orders
 * @param last_seq_num Sequence number of last snapshot order
 * @param backlog List of backlog messages from live channel
 */
void SymbolOrderBook::synchronize(
    const std::list<msg::md::MdMessage> &replay_seq,
    const uint64_t last_seq_num, const std::list<msg::md::MdMessage> &backlog) {

  std::cout << "[DEBUG] Synchronizing orderbook for symbol " << this->symbol
            << std::endl;

  // Process replay sequence
  for (const auto &m : replay_seq) {
    if (m.header.msg_type != msg::md::MSG_TYPE::NEW_ORDER) {
      throw std::runtime_error("Unexpected message type in replay sequence");
    }

    this->apply_message(m);
  }

  // Catch up to current exchange state using backlog
  for (const auto &m : backlog) {
    if (m.header.seq_num > last_seq_num) {
      this->apply_message(m);
    }
  }
}

/**
 * Modify an order in the orderbook by deleting the old one and replacing it
 *
 * @param modify_order modify order structure from exchange
 * @return `true` if modification succeeded
 */
bool SymbolOrderBook::modify_order(const msg::md::ModifyOrder &modify_order) {
  const msg::md::DeleteOrder delete_order = {.order_id = modify_order.order_id};
  const bool found_in_buy = this->side_order_books.first.delete_order(delete_order);
  const bool found_in_sell = this->side_order_books.second.delete_order(delete_order);

  if (!found_in_buy && !found_in_sell) {
    return false;
  }

  const msg::md::NewOrder new_order = {
      .order_id = modify_order.order_id,
      .symbol = this->symbol,
      .side = modify_order.side,
      .quantity = modify_order.quantity,
      .price = modify_order.price,
      .flags = 0
  };

  return this->side_order_books.first.insert_order(new_order) ||
         this->side_order_books.second.insert_order(new_order);
}

/**
 * Create an orderbook to track the price of multiple symbols
 *
 * @param symbol_count Number of symbols to track in orderbook
 */
MultiSymbolOrderBook::MultiSymbolOrderBook(const size_t symbol_count)
    : symbol_books(std::vector<SymbolOrderBook>{}) {
  for (size_t i = 0; i < symbol_count; i++) {
    this->symbol_books.emplace_back(i + 1);
  }
}

/**
 * Apply an exchange message to the orderbook
 *
 * @param message NDFEX exchange message
 * @return `true` if the operation succeeded
 */
bool MultiSymbolOrderBook::apply_message(const msg::md::MdMessage &message) {
  auto result = false;
  for (auto &symbol_book : this->symbol_books) {
    result = symbol_book.apply_message(message) || result;
  }

  this->last_sequence_number = message.header.seq_num;
  return result;
}

/**
 * Get the best price level for a given contract side
 *
 * @param symbol Contract symbol number
 * @param side BUY or SELL side
 * @return PriceLevel struct representing the best offer (`nullopt` if no orders
 * exist)
 * @throw std::out_of_range If given symbol does not exist
 */
std::optional<PriceLevel>
MultiSymbolOrderBook::get_best_price_level(const uint32_t symbol,
                                           const msg::SIDE side) {
  if (0 >= symbol || symbol > this->symbol_books.size()) {
    throw std::out_of_range("Given symbol id is not tracked by the orderbook");
  }
  return this->symbol_books.at(symbol - 1).get_best_price_level(side);
}

/**
 * Get the total volume of trades
 *
 * @return Total volume of trades tracked by orderbook
 * @throw std::out_of_range If given symbol does not exist
 */
size_t MultiSymbolOrderBook::get_total_volume_traded() const {
  size_t volume_traded = 0;
  for (auto &symbol_book : this->symbol_books) {
    volume_traded += symbol_book.get_total_volume_traded();
  }
  return volume_traded;
}

void MultiSymbolOrderBook::dump_update() {
  std::cout << "=== BEGIN ORDER BOOK UPDATE (seq#="
            << this->last_sequence_number << ") ===" << std::endl;
  for (size_t i = 0; i < this->symbol_books.size(); i++) {
    auto &symbol_book = this->symbol_books[i];
    const auto bid_pl = symbol_book.get_best_price_level(msg::SIDE::BUY);
    const auto ask_pl = symbol_book.get_best_price_level(msg::SIDE::SELL);

    const auto volume = symbol_book.get_total_volume_traded();

    i &&printf("\n");
    printf(" [Symbol #%" PRIuMAX "] volume=%" PRIuMAX "\n", i + 1, volume);
    printf(" |- BID: %5" PRIu32 " @ $%-5" PRId32 "\n", bid_pl->quantity,
           bid_pl->price);
    printf(" |- ASK: %5" PRIu32 " @ $%-5" PRId32 "\n", ask_pl->quantity,
           ask_pl->price);
  }
  std::cout << "=== END ORDER BOOK UPDATE (seq#=" << this->last_sequence_number
            << ") ===" << std::endl;
}

/**
 * Synchronize orderbook to the exchange
 *
 * @param live MulticastListener subscribed to live market data
 * @param snapshot MulticastListener subscribed to replay channel
 */
void MultiSymbolOrderBook::synchronize(
    const mcast::MulticastListener &live,
    const mcast::MulticastListener &snapshot) {

  std::cout << "[DEBUG] Beginning orderbook synchronization" << std::endl;

  // Per-symbol countdown of orders still expected from the current snapshot.
  // Initialized to -1 as a sentinel meaning "SNAPSHOT_INFO not yet received".
  std::unordered_map<uint32_t, ssize_t> bids_remaining;
  std::unordered_map<uint32_t, ssize_t> asks_remaining;

  std::bitset<32> symbols_to_sync = {};
  for (const auto &symbol_book : this->symbol_books) {
    symbols_to_sync.set(symbol_book.symbol);
    bids_remaining[symbol_book.symbol] = -1;
    asks_remaining[symbol_book.symbol] = -1;
  }

  // Completed and in-progress snapshots, keyed by symbol for O(1) lookup
  std::list<std::pair<msg::md::SnapshotInfo, std::list<msg::md::MdMessage>>>
      snapshots;
  std::unordered_map<uint32_t, decltype(snapshots)::iterator> symbol_to_snap;

  std::list<msg::md::MdMessage> backlog;

  uint8_t buf[2048];
  while (symbols_to_sync.any()) {
    // Buffer all available live updates
    if (const auto bytes_live = live.receive(buf, sizeof(buf));
        bytes_live > 0) {
      msg::md::MdMessage message;
      for (ssize_t offset = 0; offset < bytes_live;) {
        const auto n =
            msg::md::parse_message(buf + offset, bytes_live - offset, &message);
        if (!n)
          break;
        backlog.emplace_back(message);
        offset += n;
      }
    }

    // Process one snapshot datagram
    const auto bytes_snap = snapshot.receive(buf, sizeof(buf));
    if (bytes_snap <= 0)
      continue;

    for (ssize_t offset = 0; offset < bytes_snap;) {
      msg::md::MdMessage m;
      const auto n =
          msg::md::parse_message(buf + offset, bytes_snap - offset, &m);
      if (!n)
        break;
      offset += n;

      if (m.header.msg_type == msg::md::MSG_TYPE::HEARTBEAT)
        continue;

      if (m.header.msg_type == msg::md::MSG_TYPE::SNAPSHOT_INFO) {
        const auto &info = m.body.snapshot_info;
        const uint32_t sym = info.symbol;

        // Skip untracked or completed symbols
        try {
          if (!symbols_to_sync.test(sym))
            continue;
        } catch (const std::out_of_range &) {
          continue;
        }

        bids_remaining[sym] = info.bid_count;
        asks_remaining[sym] = info.ask_count;

        if (const auto it = symbol_to_snap.find(sym);
            it != symbol_to_snap.end()) {
          snapshots.erase(it->second);
          symbol_to_snap.erase(it);
        }
        snapshots.emplace_back(info, std::list<msg::md::MdMessage>{});
        symbol_to_snap[sym] = std::prev(snapshots.end());

        if (info.bid_count == 0 && info.ask_count == 0) {
          symbols_to_sync.reset(sym);
          symbol_to_snap.erase(sym);
        }
      } else if (m.header.msg_type == msg::md::MSG_TYPE::NEW_ORDER) {
        const uint32_t sym = m.body.new_order.symbol;
        const auto it = symbol_to_snap.find(sym);

        // Ignore orders if SNAPSHOT_INFO hasn't come in for symbol
        if (it == symbol_to_snap.end())
          continue;

        if (m.body.new_order.side == msg::SIDE::BUY) {
          --bids_remaining[sym];
        } else {
          --asks_remaining[sym];
        }
        it->second->second.emplace_back(m);

        if (bids_remaining[sym] == 0 && asks_remaining[sym] == 0) {
          symbols_to_sync.reset(sym);
          symbol_to_snap.erase(sym);
        }
      }
    }
  }

  // Apply completed snapshots to individual orderbooks
  for (const auto &[snapshot_info, replay_seq] : snapshots) {
    auto &orderbook = this->symbol_books.at(snapshot_info.symbol - 1);
    orderbook.synchronize(replay_seq, snapshot_info.last_md_seq_num, backlog);
  }

  std::cout << "[DEBUG] Orderbook synchronization complete" << std::endl;
}
} // namespace hft::orderbook
