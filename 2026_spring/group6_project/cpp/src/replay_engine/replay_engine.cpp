#include "replay_engine.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

namespace ghostbook::replay {

namespace {

uint64_t nanotime() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return ts.tv_sec * 1e9 + ts.tv_nsec;
}

const char *side_str(ndfex::SIDE s) {
  return s == ndfex::SIDE::BUY ? "BUY" : "SELL";
}

const char *side_str(matching::Side s) {
  return s == matching::Side::Buy ? "BUY" : "SELL";
}

const char *msg_type_str(ndfex::md::MSG_TYPE t) {
  switch (t) {
  case ndfex::md::MSG_TYPE::NEW_ORDER:      return "NEW_ORDER";
  case ndfex::md::MSG_TYPE::DELETE_ORDER:   return "DELETE_ORDER";
  case ndfex::md::MSG_TYPE::MODIFY_ORDER:   return "MODIFY_ORDER";
  case ndfex::md::MSG_TYPE::TRADE:          return "TRADE";
  case ndfex::md::MSG_TYPE::TRADE_SUMMARY:  return "TRADE_SUMMARY";
  case ndfex::md::MSG_TYPE::SNAPSHOT_INFO:  return "SNAPSHOT_INFO";
  default:                                   return "UNKNOWN";
  }
}

ndfex::SIDE opposite_side(ndfex::SIDE s) {
  return s == ndfex::SIDE::BUY ? ndfex::SIDE::SELL : ndfex::SIDE::BUY;
}

matching::ExecutionEvent make_ack(matching::order_id_t oid,
                                  matching::instrument_id_t sym,
                                  matching::Side side, matching::quantity_t qty,
                                  matching::price_tick_t price,
                                  matching::logical_clock_t clk) {
  matching::ExecutionEvent ev{};
  ev.type = matching::EventType::Ack;
  ev.order_id = oid;
  ev.instrument_id = sym;
  ev.side = side;
  ev.quantity = qty;
  ev.remaining_quantity = qty;
  ev.cumulative_quantity = 0;
  ev.price_tick = price;
  ev.logical_clock = clk;
  return ev;
}

matching::ExecutionEvent make_reject(matching::order_id_t oid,
                                     matching::instrument_id_t sym,
                                     matching::Side side,
                                     const std::string &reason,
                                     matching::logical_clock_t clk) {
  matching::ExecutionEvent ev{};
  ev.type = matching::EventType::Reject;
  ev.order_id = oid;
  ev.instrument_id = sym;
  ev.side = side;
  ev.logical_clock = clk;
  ev.reason = reason;
  return ev;
}

matching::Side ndfex_to_matching_side(ndfex::SIDE s) {
  return s == ndfex::SIDE::BUY ? matching::Side::Buy : matching::Side::Sell;
}

ndfex::SIDE matching_to_ndfex_side(matching::Side s) {
  return s == matching::Side::Buy ? ndfex::SIDE::BUY : ndfex::SIDE::SELL;
}

} // namespace

// =============================================================================
// Construction
// =============================================================================

ReplayEngine::ReplayEngine(matching::logical_clock_t start_clock)
    : initial_clock_(nanotime()), logical_clock_(start_clock) {}

// =============================================================================
// File I/O
// =============================================================================

bool ReplayEngine::read_message(ndfex::md::Message &out, bool snapshot) {
  const std::uint64_t expected_magic =
      snapshot ? ndfex::md::SNAPSHOT_MAGIC_NUMBER : ndfex::md::MAGIC_NUMBER;

  while (true) {
    ndfex::md::Header hdr{};
    if (!feed_stream_.read(reinterpret_cast<char *>(&hdr), sizeof(hdr))) {
      return false;
    }
    if (hdr.length < sizeof(hdr)) {
      return false;
    }

    const std::size_t body_size = hdr.length - sizeof(hdr);

    if (hdr.magic_number != expected_magic) {
      // Skip unknown/mismatched message
      if (body_size > 0) {
        feed_stream_.seekg(static_cast<std::streamoff>(body_size),
                           std::ios::cur);
      }
      if (!snapshot)
        return false;
      continue;
    }

    out.header = hdr;
    if (body_size > 0) {
      const std::size_t copy_sz = std::min(body_size, sizeof(out.body));
      if (!feed_stream_.read(reinterpret_cast<char *>(&out.body), copy_sz)) {
        return false;
      }
      if (body_size > copy_sz) {
        feed_stream_.seekg(static_cast<std::streamoff>(body_size - copy_sz),
                           std::ios::cur);
      }
    }
    return true;
  }
}

bool ReplayEngine::load_snapshot(const std::filesystem::path &path) {
  std::cout << "[REPLAY] Loading snapshot: " << path << "\n";
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cout << "[REPLAY] Failed to open snapshot: " << path << "\n";
    return false;
  }

  // Temporarily use feed_stream_ for reading the snapshot and restore after.
  feed_stream_ = std::move(file);

  ndfex::md::Message msg{};
  std::size_t order_count = 0;
  while (true) {
    if (!read_message(msg, /*snapshot=*/true))
      break;

    switch (msg.header.msg_type) {
    case ndfex::md::MSG_TYPE::SNAPSHOT_INFO: {
      const auto &info = msg.body.snapshot_info;
      if (info.last_md_seq_num > snapshot_watermark_) {
        snapshot_watermark_ = info.last_md_seq_num;
        std::cout << "[REPLAY] Snapshot watermark: seq=" << snapshot_watermark_ << "\n";
      }
      break;
    }
    case ndfex::md::MSG_TYPE::NEW_ORDER: {
      const auto &no = msg.body.new_order;
      shadow_insert(/*is_live=*/false, no.order_id, no.symbol, no.side,
                    no.price, no.quantity);
      ++order_count;
      break;
    }
    default:
      break;
    }
  }

  feed_stream_.close();
  std::cout << "[REPLAY] Snapshot loaded: " << order_count << " orders, watermark=" << snapshot_watermark_ << "\n";
  return true;
}

bool ReplayEngine::open_feed(const std::filesystem::path &path) {
  std::cout << "[REPLAY] Opening feed: " << path << "\n";
  feed_stream_.open(path, std::ios::binary);
  if (!feed_stream_) {
    std::cout << "[REPLAY] Failed to open feed: " << path << "\n";
    return false;
  }
  feed_exhausted_ = false;

  // Skip messages already reflected in the snapshot.
  if (snapshot_watermark_ == 0) {
    std::cout << "[REPLAY] No snapshot; replaying feed from start\n";
    return true;
  }

  std::size_t skipped = 0;
  while (true) {
    const auto start_pos = feed_stream_.tellg();
    ndfex::md::Header hdr{};
    if (!feed_stream_.read(reinterpret_cast<char *>(&hdr), sizeof(hdr))) {
      feed_exhausted_ = true;
      std::cout << "[REPLAY] Feed fast-forward exhausted after skipping " << skipped << " messages\n";
      return true;
    }
    if (hdr.magic_number != ndfex::md::MAGIC_NUMBER ||
        hdr.length < sizeof(hdr)) {
      feed_exhausted_ = true;
      std::cout << "[REPLAY] Feed fast-forward hit invalid header after skipping " << skipped << " messages\n";
      return true;
    }
    if (hdr.seq_num > snapshot_watermark_) {
      // This message is new — rewind and stop.
      feed_stream_.seekg(start_pos);
      std::cout << "[REPLAY] Feed ready: skipped " << skipped << " snapshot-covered messages, resuming at seq=" << hdr.seq_num << "\n";
      return true;
    }
    // Skip body of this already-applied message.
    const std::size_t body_size = hdr.length - sizeof(hdr);
    if (body_size > 0) {
      feed_stream_.seekg(static_cast<std::streamoff>(body_size), std::ios::cur);
    }
    ++skipped;
  }
}

bool ReplayEngine::advance() {
  if (feed_exhausted_ || !feed_stream_.is_open())
    return false;

  ndfex::md::Message msg{};
  if (defered_msg_.has_value()) {
    msg = defered_msg_.value();
  } else {
    if (!read_message(msg, /*snapshot=*/false)) {
      feed_exhausted_ = true;
      std::cout << "[INFO] Replay feed exhausted" << std::endl;
      return false;
    }
  }

  logical_clock_ = static_cast<matching::logical_clock_t>(msg.header.seq_num);

  // Check if message must be deferred
  if (!first_replay_clock_)
    first_replay_clock_ = msg.header.timestamp;

  const auto msg_clock_adj = msg.header.timestamp - first_replay_clock_;
  const auto curr_time_adj =
      (nanotime() - initial_clock_) * 1; // NOTE: 1x speed
  if (curr_time_adj < msg_clock_adj) {
    if (!defered_msg_.has_value()) {
      std::cout << "[REPLAY] Deferring seq=" << msg.header.seq_num
                << " type=" << msg_type_str(msg.header.msg_type)
                << " (behind by " << (msg_clock_adj - curr_time_adj) / 1000 << "us)\n";
    }
    defered_msg_ = msg;
    return true;
  }

  std::cout << "[REPLAY] Dispatching seq=" << msg.header.seq_num
            << " type=" << msg_type_str(msg.header.msg_type) << "\n";

  switch (msg.header.msg_type) {
  case ndfex::md::MSG_TYPE::NEW_ORDER:
    handle_new_order(msg.body.new_order);
    break;
  case ndfex::md::MSG_TYPE::DELETE_ORDER:
    handle_delete_order(msg.body.delete_order);
    break;
  case ndfex::md::MSG_TYPE::MODIFY_ORDER:
    handle_modify_order(msg.body.modify_order);
    break;
  case ndfex::md::MSG_TYPE::TRADE:
    handle_trade(msg.body.trade);
    break;
  case ndfex::md::MSG_TYPE::TRADE_SUMMARY:
    handle_trade_summary(msg.body.trade_summary);
    break;
  default:
    std::cout << "[REPLAY] Unknown message type, skipping\n";
    break;
  }

  defered_msg_.reset();

  if (feed_msg_cb_) {
    feed_msg_cb_(msg);
  }

  return true;
}

bool ReplayEngine::feed_exhausted() const { return feed_exhausted_; }

void ReplayEngine::set_feed_message_callback(FeedMessageCallback cb) {
  feed_msg_cb_ = std::move(cb);
}

void ReplayEngine::run_once() { advance(); }

matching::logical_clock_t ReplayEngine::logical_clock() const {
  return logical_clock_;
}

// =============================================================================
// Shadow LOB mutations
// =============================================================================

void ReplayEngine::shadow_insert(bool is_live, ndfex::order_id_t oid,
                                 ndfex::symbol_id_t sym, ndfex::SIDE side,
                                 ndfex::price_t price,
                                 matching::quantity_t qty) {
  auto &book = books_[sym];
  LevelQueue *queue_ptr = nullptr;

  if (side == ndfex::SIDE::BUY) {
    queue_ptr = &book.bids[price];
  } else {
    queue_ptr = &book.asks[price];
  }

  queue_ptr->push_back(
      LevelEntry{is_live ? EntryKind::Live : EntryKind::Historical, oid, qty});

  OrderMeta meta{sym, side, price, std::prev(queue_ptr->end())};

  if (is_live) {
    live_index_[oid] = meta;
  } else {
    hist_index_[oid] = meta;
  }
}

void ReplayEngine::shadow_remove_hist(ndfex::order_id_t oid) {
  auto it = hist_index_.find(oid);
  if (it == hist_index_.end())
    return;

  const OrderMeta &meta = it->second;
  auto &book = books_[meta.symbol];
  LevelQueue *q = nullptr;

  if (meta.side == ndfex::SIDE::BUY) {
    auto level_it = book.bids.find(meta.price);
    if (level_it != book.bids.end()) {
      level_it->second.erase(meta.queue_it);
      if (level_it->second.empty())
        book.bids.erase(level_it);
    }
  } else {
    auto level_it = book.asks.find(meta.price);
    if (level_it != book.asks.end()) {
      level_it->second.erase(meta.queue_it);
      if (level_it->second.empty())
        book.asks.erase(level_it);
    }
  }

  hist_index_.erase(it);
}

void ReplayEngine::shadow_remove_live(ndfex::order_id_t oid) {
  auto it = live_index_.find(oid);
  if (it == live_index_.end())
    return;

  const OrderMeta &meta = it->second;
  erase_live_entry(meta.symbol, meta.side, meta.price, meta.queue_it);
  live_index_.erase(it);
}

ReplayEngine::LevelQueue::iterator
ReplayEngine::erase_live_entry(ndfex::symbol_id_t sym, ndfex::SIDE side,
                               ndfex::price_t price, LevelQueue::iterator it) {
  auto &book = books_[sym];
  if (side == ndfex::SIDE::BUY) {
    auto level_it = book.bids.find(price);
    if (level_it == book.bids.end())
      return {};
    auto next = level_it->second.erase(it);
    if (level_it->second.empty())
      book.bids.erase(level_it);
    return next;
  } else {
    auto level_it = book.asks.find(price);
    if (level_it == book.asks.end())
      return {};
    auto next = level_it->second.erase(it);
    if (level_it->second.empty())
      book.asks.erase(level_it);
    return next;
  }
}

// =============================================================================
// Feed message handlers
// =============================================================================

void ReplayEngine::handle_new_order(const ndfex::md::NewOrder &msg) {
  std::cout << "[REPLAY] hist NEW_ORDER oid=" << msg.order_id
            << " sym=" << msg.symbol << " side=" << side_str(msg.side)
            << " px=" << msg.price << " qty=" << msg.quantity << "\n";
  shadow_insert(/*is_live=*/false, msg.order_id, msg.symbol, msg.side,
                msg.price, msg.quantity);
}

void ReplayEngine::handle_delete_order(const ndfex::md::DeleteOrder &msg) {
  std::cout << "[REPLAY] hist DELETE_ORDER oid=" << msg.order_id << "\n";
  shadow_remove_hist(msg.order_id);
}

void ReplayEngine::handle_modify_order(const ndfex::md::ModifyOrder &msg) {
  auto it = hist_index_.find(msg.order_id);
  if (it == hist_index_.end()) {
    std::cout << "[REPLAY] hist MODIFY_ORDER oid=" << msg.order_id << " (not found, ignoring)\n";
    return;
  }
  std::cout << "[REPLAY] hist MODIFY_ORDER oid=" << msg.order_id
            << " new_qty=" << msg.quantity << "\n";
  it->second.queue_it->remaining_qty = msg.quantity;
}

void ReplayEngine::handle_trade_summary(
    const ndfex::md::TradeSummary & /*msg*/) {
  // Context from TRADE_SUMMARY is available in handle_trade via hist_index_
  // lookup. No action needed here for ghost fill logic.
}

void ReplayEngine::handle_trade(const ndfex::md::Trade &msg) {
  auto it = hist_index_.find(msg.order_id);
  if (it == hist_index_.end()) {
    std::cout << "[REPLAY] TRADE oid=" << msg.order_id << " qty=" << msg.quantity << " (not a known hist maker, skipping ghost fill)\n";
    return; // Not a known historical maker — taker ID or stale reference.
  }

  const OrderMeta &meta = it->second;
  std::cout << "[REPLAY] TRADE hist maker oid=" << msg.order_id
            << " sym=" << meta.symbol << " side=" << side_str(meta.side)
            << " px=" << meta.price << " qty=" << msg.quantity << " — checking ghost fills\n";
  check_ghost_fills(msg.order_id, meta.symbol, meta.side, meta.price,
                    msg.quantity);
}

// =============================================================================
// Ghost fill core algorithm
// =============================================================================

void ReplayEngine::check_ghost_fills(ndfex::order_id_t hist_maker_id,
                                     ndfex::symbol_id_t sym,
                                     ndfex::SIDE maker_side,
                                     ndfex::price_t price,
                                     matching::quantity_t traded_qty) {
  auto book_it = books_.find(sym);
  if (book_it == books_.end())
    return;

  auto &book = book_it->second;
  LevelQueue *q = nullptr;

  if (maker_side == ndfex::SIDE::BUY) {
    auto level_it = book.bids.find(price);
    if (level_it == book.bids.end())
      return;
    q = &level_it->second;
  } else {
    auto level_it = book.asks.find(price);
    if (level_it == book.asks.end())
      return;
    q = &level_it->second;
  }

  matching::quantity_t fill_budget = traded_qty;
  std::vector<ndfex::order_id_t> to_remove;

  for (auto &entry : *q) {
    if (entry.kind == EntryKind::Historical &&
        entry.order_id == hist_maker_id) {
      break; // reached the target; all higher-priority entries processed
    }

    if (entry.kind == EntryKind::Live && fill_budget > 0) {
      const matching::quantity_t fill_qty =
          std::min(entry.remaining_qty, fill_budget);
      entry.remaining_qty -= fill_qty;
      fill_budget -= fill_qty;

      std::cout << "[REPLAY] Ghost fill: live oid=" << entry.order_id
                << " sym=" << sym << " side=" << side_str(maker_side)
                << " px=" << price << " fill_qty=" << fill_qty
                << " remaining=" << entry.remaining_qty << "\n";

      matching::ExecutionEvent ev{};
      ev.type = matching::EventType::Fill;
      ev.order_id = entry.order_id;
      ev.instrument_id = sym;
      ev.side = ndfex_to_matching_side(maker_side);
      ev.quantity = fill_qty;
      ev.remaining_quantity = entry.remaining_qty;
      ev.cumulative_quantity = 0; // gateway doesn't use this for fills
      ev.price_tick = static_cast<matching::price_tick_t>(price);
      ev.logical_clock = logical_clock_;
      ev.reason = "ghost fill";
      emit_event(ev);

      if (entry.remaining_qty == 0) {
        to_remove.push_back(entry.order_id);
      }
    }
  }

  for (ndfex::order_id_t oid : to_remove) {
    shadow_remove_live(oid);
  }
}

// =============================================================================
// Live order operations (MatchingEngine-compatible interface)
// =============================================================================

std::optional<matching::ExecutionEvent>
ReplayEngine::submit(const matching::NewOrderCommand &cmd) {
  const auto oid = cmd.client_order_id;
  const auto sym = static_cast<ndfex::symbol_id_t>(cmd.instrument_id);
  const auto side = cmd.side;
  const auto ndfex_side = matching_to_ndfex_side(side);
  const auto price = static_cast<ndfex::price_t>(cmd.price_tick);
  const auto qty = cmd.quantity;

  std::cout << "[REPLAY] SUBMIT oid=" << oid << " sym=" << sym
            << " side=" << side_str(side) << " px=" << price << " qty=" << qty << "\n";

  // Validation
  if (qty == 0) {
    std::cout << "[REPLAY] REJECT oid=" << oid << " reason=invalid quantity\n";
    auto ev = make_reject(oid, cmd.instrument_id, side, "invalid quantity",
                          logical_clock_);
    emit_event(ev);
    return ev;
  }
  if (cmd.order_type == matching::OrderType::Limit && price <= 0) {
    std::cout << "[REPLAY] REJECT oid=" << oid << " reason=invalid price\n";
    auto ev = make_reject(oid, cmd.instrument_id, side, "invalid price",
                          logical_clock_);
    emit_event(ev);
    return ev;
  }
  if (live_index_.count(oid) > 0) {
    std::cout << "[REPLAY] REJECT oid=" << oid << " reason=duplicate order id\n";
    auto ev = make_reject(oid, cmd.instrument_id, side, "duplicate order id",
                          logical_clock_);
    emit_event(ev);
    return ev;
  }

  std::cout << "[REPLAY] ACK oid=" << oid << "\n";
  auto ack = make_ack(oid, cmd.instrument_id, side, qty, price, logical_clock_);
  emit_event(ack);

  // Check for immediate cross (live client as taker)
  matching::quantity_t remaining = qty;
  auto &book = books_[sym];

  auto try_fill = [&](LevelQueue &level, ndfex::price_t level_price) {
    std::vector<ndfex::order_id_t> live_filled;
    for (auto &entry : level) {
      if (remaining == 0)
        break;
      const matching::quantity_t fill_qty =
          std::min(entry.remaining_qty, remaining);
      remaining -= fill_qty;

      matching::ExecutionEvent fill_ev{};
      fill_ev.type = matching::EventType::Fill;
      fill_ev.order_id = oid;
      fill_ev.instrument_id = sym;
      fill_ev.side = side;
      fill_ev.quantity = fill_qty;
      fill_ev.remaining_quantity = remaining;
      fill_ev.cumulative_quantity = qty - remaining;
      fill_ev.price_tick = static_cast<matching::price_tick_t>(level_price);
      fill_ev.logical_clock = logical_clock_;
      fill_ev.reason = "taker fill";
      std::cout << "[REPLAY] Taker fill: oid=" << oid << " px=" << level_price
                << " fill_qty=" << fill_qty << " remaining=" << remaining << "\n";
      emit_event(fill_ev);

      // Live-vs-live: consume the passive live order
      if (entry.kind == EntryKind::Live) {
        entry.remaining_qty -= fill_qty;
        if (entry.remaining_qty == 0) {
          live_filled.push_back(entry.order_id);
        }
      }
      // Historical entries are NOT consumed (v1 simplification)
    }
    for (auto passive_oid : live_filled) {
      // Emit fill for the passive live order too
      auto passive_meta_it = live_index_.find(passive_oid);
      if (passive_meta_it != live_index_.end()) {
        matching::ExecutionEvent passive_fill{};
        passive_fill.type = matching::EventType::Fill;
        passive_fill.order_id = passive_oid;
        passive_fill.instrument_id = sym;
        passive_fill.side = opposite_side(ndfex_side) == ndfex::SIDE::BUY
                                ? matching::Side::Buy
                                : matching::Side::Sell;
        passive_fill.quantity = 0; // already emitted above
        passive_fill.remaining_quantity = 0;
        passive_fill.logical_clock = logical_clock_;
        passive_fill.reason = "passive fill";
        // We already emitted fills; just clean up.
        shadow_remove_live(passive_oid);
      }
    }
  };

  if (side == matching::Side::Buy) {
    // Taker buys: look at asks (ascending price)
    for (auto &[level_price, level_q] : book.asks) {
      if (remaining == 0)
        break;
      if (cmd.order_type == matching::OrderType::Limit && level_price > price)
        break;
      try_fill(level_q, level_price);
    }
  } else {
    // Taker sells: look at bids (descending price)
    for (auto &[level_price, level_q] : book.bids) {
      if (remaining == 0)
        break;
      if (cmd.order_type == matching::OrderType::Limit && level_price < price)
        break;
      try_fill(level_q, level_price);
    }
  }

  if (remaining > 0) {
    if (cmd.tif == matching::TimeInForce::IOC) {
      std::cout << "[REPLAY] IOC cancel oid=" << oid << " unfilled_qty=" << remaining << "\n";
      matching::ExecutionEvent cancel_ev{};
      cancel_ev.type = matching::EventType::Cancel;
      cancel_ev.order_id = oid;
      cancel_ev.instrument_id = sym;
      cancel_ev.side = side;
      cancel_ev.remaining_quantity = remaining;
      cancel_ev.logical_clock = logical_clock_;
      cancel_ev.reason = "IOC unfilled";
      emit_event(cancel_ev);
    } else {
      std::cout << "[REPLAY] Order resting oid=" << oid << " remaining=" << remaining << " px=" << price << "\n";
      // Day: rest the remainder in shadow LOB
      shadow_insert(/*is_live=*/true, oid, sym, ndfex_side, price, remaining);
    }
  }

  return ack;
}

std::optional<matching::ExecutionEvent>
ReplayEngine::cancel(const matching::CancelOrderCommand &cmd) {
  const auto oid = cmd.target_order_id;
  std::cout << "[REPLAY] CANCEL target_oid=" << oid << "\n";

  auto it = live_index_.find(oid);
  if (it == live_index_.end()) {
    std::cout << "[REPLAY] REJECT cancel oid=" << oid << " reason=unknown order id\n";
    auto ev = make_reject(cmd.client_order_id, 0, matching::Side::Buy,
                          "unknown order id", logical_clock_);
    emit_event(ev);
    return ev;
  }

  const matching::quantity_t rem = it->second.queue_it->remaining_qty;
  shadow_remove_live(oid);

  std::cout << "[REPLAY] Cancel confirmed oid=" << oid << " remaining_qty=" << rem << "\n";
  matching::ExecutionEvent ev{};
  ev.type = matching::EventType::Cancel;
  ev.order_id = oid;
  ev.instrument_id = static_cast<matching::instrument_id_t>(it->second.symbol);
  ev.side = ndfex_to_matching_side(it->second.side);
  ev.remaining_quantity = rem;
  ev.logical_clock = logical_clock_;
  ev.reason = "cancel";
  emit_event(ev);
  return ev;
}

std::optional<matching::ExecutionEvent>
ReplayEngine::modify(const matching::ModifyOrderCommand &cmd) {
  const auto orig_oid = cmd.original_order_id;
  const auto new_oid = cmd.new_order_id;

  std::cout << "[REPLAY] MODIFY orig_oid=" << orig_oid << " new_oid=" << new_oid
            << " new_px=" << cmd.new_price_tick << " new_qty=" << cmd.new_quantity << "\n";

  auto it = live_index_.find(orig_oid);
  if (it == live_index_.end()) {
    std::cout << "[REPLAY] REJECT modify orig_oid=" << orig_oid << " reason=unknown order id\n";
    auto ev = make_reject(orig_oid, 0, matching::Side::Buy, "unknown order id",
                          logical_clock_);
    emit_event(ev);
    return ev;
  }

  const ndfex::symbol_id_t sym = it->second.symbol;
  const ndfex::SIDE side = it->second.side;

  // Cancel old order (gateway will suppress this)
  const matching::quantity_t old_rem = it->second.queue_it->remaining_qty;
  shadow_remove_live(orig_oid);

  matching::ExecutionEvent cancel_ev{};
  cancel_ev.type = matching::EventType::Cancel;
  cancel_ev.order_id = orig_oid;
  cancel_ev.instrument_id = sym;
  cancel_ev.side = ndfex_to_matching_side(side);
  cancel_ev.remaining_quantity = old_rem;
  cancel_ev.logical_clock = logical_clock_;
  cancel_ev.reason = "modify cancel";
  emit_event(cancel_ev);

  // Submit new order (may immediately cross or rest)
  matching::NewOrderCommand new_cmd{};
  new_cmd.client_order_id = new_oid;
  new_cmd.instrument_id = sym;
  new_cmd.side = ndfex_to_matching_side(side);
  new_cmd.order_type = matching::OrderType::Limit;
  new_cmd.tif = matching::TimeInForce::Day;
  new_cmd.price_tick = cmd.new_price_tick;
  new_cmd.quantity = cmd.new_quantity;

  return submit(new_cmd);
}

std::vector<matching::ExecutionEvent> ReplayEngine::drain_events() {
  std::vector<matching::ExecutionEvent> result;
  result.swap(events_);
  return result;
}

void ReplayEngine::emit_event(matching::ExecutionEvent ev) {
  events_.push_back(std::move(ev));
}

// =============================================================================
// Shadow LOB inspection (for tests)
// =============================================================================

std::vector<ReplayEngine::LevelView>
ReplayEngine::bid_levels(ndfex::symbol_id_t sym) const {
  std::vector<LevelView> result;
  auto book_it = books_.find(sym);
  if (book_it == books_.end())
    return result;

  for (const auto &[price, q] : book_it->second.bids) {
    if (q.empty())
      continue;
    LevelView v{};
    v.price = price;
    for (const auto &e : q) {
      v.total_qty += e.remaining_qty;
      if (e.kind == EntryKind::Live)
        ++v.live_count;
    }
    result.push_back(v);
  }
  return result;
}

std::vector<ReplayEngine::LevelView>
ReplayEngine::ask_levels(ndfex::symbol_id_t sym) const {
  std::vector<LevelView> result;
  auto book_it = books_.find(sym);
  if (book_it == books_.end())
    return result;

  for (const auto &[price, q] : book_it->second.asks) {
    if (q.empty())
      continue;
    LevelView v{};
    v.price = price;
    for (const auto &e : q) {
      v.total_qty += e.remaining_qty;
      if (e.kind == EntryKind::Live)
        ++v.live_count;
    }
    result.push_back(v);
  }
  return result;
}

} // namespace ghostbook::replay
