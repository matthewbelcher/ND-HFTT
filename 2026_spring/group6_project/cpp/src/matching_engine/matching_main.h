#pragma once

#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "flat_hash_map.h"

namespace ghostbook::matching {

using logical_clock_t = std::uint64_t;
using order_id_t = std::uint64_t;
using instrument_id_t = std::uint32_t;
using price_tick_t = std::int64_t;
using quantity_t = std::uint32_t;

inline const std::set<instrument_id_t> SYMBOLS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};

enum class Side : std::uint8_t { Buy = 1, Sell = 2 };
enum class OrderType : std::uint8_t { Limit = 1, Market = 2 };
enum class TimeInForce : std::uint8_t { Day = 1, IOC = 2, FOK = 3, PostOnly = 4 };

struct NewOrderCommand {
	order_id_t client_order_id{};
	instrument_id_t instrument_id{};
	Side side{Side::Buy};
	OrderType order_type{OrderType::Limit};
	TimeInForce tif{TimeInForce::Day};
	price_tick_t price_tick{};
	quantity_t quantity{};
};

struct CancelOrderCommand {
	order_id_t client_order_id{};
	order_id_t target_order_id{};
	quantity_t cancel_quantity{};
};

struct ModifyOrderCommand {
	order_id_t original_order_id{};
	order_id_t new_order_id{};
	price_tick_t new_price_tick{};
	quantity_t new_quantity{};
};

struct Order {
	order_id_t order_id{};
	instrument_id_t instrument_id{};
	Side side{Side::Buy};
	OrderType order_type{OrderType::Limit};
	TimeInForce tif{TimeInForce::Day};
	price_tick_t price_tick{};
	quantity_t original_quantity{};
	quantity_t remaining_quantity{};
	logical_clock_t entry_clock{};
};

enum class EventType : std::uint8_t { Ack = 1, Fill = 2, Cancel = 3, Reject = 4 };

struct ExecutionEvent {
	EventType type{EventType::Ack};
	order_id_t order_id{};
	order_id_t counterparty_order_id{};
	instrument_id_t instrument_id{};
	Side side{Side::Buy};
	quantity_t quantity{};
	quantity_t remaining_quantity{};
	quantity_t cumulative_quantity{};
	price_tick_t price_tick{};
	logical_clock_t logical_clock{};
	std::string reason;
};

class MatchingEngine {
public:
	explicit MatchingEngine(logical_clock_t start_clock = 0);

	std::optional<ExecutionEvent> submit(const NewOrderCommand& command);
	std::optional<ExecutionEvent> cancel(const CancelOrderCommand& command);
	std::optional<ExecutionEvent> modify(const ModifyOrderCommand& command);

	void run_once();
	logical_clock_t logical_clock() const;
	std::vector<ExecutionEvent> drain_events();

private:
	struct RestingOrder;
	using LevelQueue = std::list<RestingOrder*>;
	using BidLevels = std::map<price_tick_t, LevelQueue, std::greater<price_tick_t>>;
	using AskLevels = std::map<price_tick_t, LevelQueue, std::less<price_tick_t>>;

	struct RestingOrder {
		Order order;
		LevelQueue::iterator level_it{};
	};

	struct InstrumentBook {
		BidLevels bids;
		AskLevels asks;
	};

	logical_clock_t logical_clock_{};
	FlatHashMap<order_id_t, std::unique_ptr<RestingOrder>> order_index_;
	FlatHashMap<instrument_id_t, InstrumentBook> books_;
	std::vector<ExecutionEvent> events_;

	void emit_event(ExecutionEvent event);
	std::optional<ExecutionEvent> accept_new_order(const NewOrderCommand& command, bool is_replace);
	void insert_resting_order(RestingOrder* resting_order);
	bool remove_order_from_book(const RestingOrder& resting_order);
	bool crosses_top(const NewOrderCommand& command) const;
	quantity_t executable_quantity(const NewOrderCommand& command) const;
	void match_order(Order& taker_order);
};

}  // namespace ghostbook::matching