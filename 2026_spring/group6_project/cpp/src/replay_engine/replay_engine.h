#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <list>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

#include "../matching_engine/matching_main.h"
#include "ghostbook/ndfex/protocol.h"

namespace ghostbook::replay {

class ReplayEngine {
public:
  using FeedMessageCallback = std::function<void(const ndfex::md::Message &)>;

  explicit ReplayEngine(matching::logical_clock_t start_clock = 0);

  // MatchingEngine-compatible interface
  std::optional<matching::ExecutionEvent>
  submit(const matching::NewOrderCommand &cmd);
  std::optional<matching::ExecutionEvent>
  cancel(const matching::CancelOrderCommand &cmd);
  std::optional<matching::ExecutionEvent>
  modify(const matching::ModifyOrderCommand &cmd);
  std::vector<matching::ExecutionEvent> drain_events();
  matching::logical_clock_t logical_clock() const;
  void run_once(); // advances feed by one message

  // Replay-specific
  bool load_snapshot(const std::filesystem::path &path);
  bool open_feed(const std::filesystem::path &path);
  bool advance(); // process one feed message; returns false on EOF
  bool feed_exhausted() const;
  // True when advance() read the next message but its timestamp hasn't
  // arrived yet.  The engine_loop uses this to decide whether to yield.
  bool has_deferred_message() const { return defered_msg_.has_value(); }

  // Called for every feed message processed by advance(). Used by the gateway
  // to forward historical market data to the live client's MD channel.
  void set_feed_message_callback(FeedMessageCallback cb);

  // Inspection for tests
  struct LevelView {
    ndfex::price_t price;
    matching::quantity_t total_qty;
    std::size_t live_count;
  };
  std::vector<LevelView> bid_levels(ndfex::symbol_id_t sym) const;
  std::vector<LevelView> ask_levels(ndfex::symbol_id_t sym) const;

private:
  enum class EntryKind : std::uint8_t { Historical, Live };

  struct LevelEntry {
    EntryKind kind;
    ndfex::order_id_t order_id;
    matching::quantity_t remaining_qty;
  };

  using LevelQueue = std::list<LevelEntry>;
  using BidLevels =
      std::map<ndfex::price_t, LevelQueue, std::greater<ndfex::price_t>>;
  using AskLevels = std::map<ndfex::price_t, LevelQueue>;

  struct ShadowBook {
    BidLevels bids;
    AskLevels asks;
  };

  struct OrderMeta {
    ndfex::symbol_id_t symbol;
    ndfex::SIDE side;
    ndfex::price_t price;
    LevelQueue::iterator queue_it;
  };

  const uint64_t initial_clock_{};
  uint64_t first_replay_clock_{};
  std::optional<ndfex::md::Message> defered_msg_{};

  matching::logical_clock_t logical_clock_{};
  std::unordered_map<ndfex::order_id_t, OrderMeta> hist_index_;
  std::unordered_map<ndfex::order_id_t, OrderMeta> live_index_;
  std::unordered_map<ndfex::symbol_id_t, ShadowBook> books_;
  std::vector<matching::ExecutionEvent> events_;

  std::ifstream feed_stream_;
  bool feed_exhausted_{false};
  ndfex::seq_num_t snapshot_watermark_{0};

  FeedMessageCallback feed_msg_cb_;

  void handle_new_order(const ndfex::md::NewOrder &msg);
  void handle_delete_order(const ndfex::md::DeleteOrder &msg);
  void handle_modify_order(const ndfex::md::ModifyOrder &msg);
  void handle_trade(const ndfex::md::Trade &msg);
  void handle_trade_summary(const ndfex::md::TradeSummary &msg);

  void check_ghost_fills(ndfex::order_id_t hist_maker_id,
                         ndfex::symbol_id_t sym, ndfex::SIDE maker_side,
                         ndfex::price_t price, matching::quantity_t traded_qty);

  void shadow_insert(bool is_live, ndfex::order_id_t oid,
                     ndfex::symbol_id_t sym, ndfex::SIDE side,
                     ndfex::price_t price, matching::quantity_t qty);
  void shadow_remove_hist(ndfex::order_id_t oid);
  void shadow_remove_live(ndfex::order_id_t oid);

  // Remove a live entry by iterator and clean up the price level if empty.
  // Returns the iterator after the erased element.
  LevelQueue::iterator erase_live_entry(ndfex::symbol_id_t sym,
                                        ndfex::SIDE side, ndfex::price_t price,
                                        LevelQueue::iterator it);

  bool read_message(ndfex::md::Message &out, bool snapshot);
  void emit_event(matching::ExecutionEvent ev);
};

} // namespace ghostbook::replay
