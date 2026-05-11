#ifndef TRADING_PROJECT_ORDERBOOK_H
#define TRADING_PROJECT_ORDERBOOK_H

#include <list>
#include <memory>
#include <optional>
#include <queue>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "msg/market_data.h"
#include "multicast.h"

namespace hft::orderbook {

    /* Class Templates */

    class PriceLevel {
    public:
        int32_t price{};
        uint32_t quantity{};
        uint64_t num_orders{};

        std::list<msg::md::NewOrder> orders{};
    };
    using PriceLevelPtr = std::shared_ptr<PriceLevel>;

    /**
     * Holds either BUY or SELL orders for a single symbol
     */
    class SideOrderBook {
        const uint32_t symbol;
        const msg::SIDE side;

        struct PairType {
            int32_t priority{};
            int32_t price{};
            PriceLevelPtr price_level;

            bool operator<(const PairType &other) const { return priority < other.priority; }
            bool operator==(const PairType &other) const { return priority == other.priority; }
            bool operator!=(const PairType &other) const { return priority != other.priority; }
            bool operator>(const PairType &other) const { return priority > other.priority; }
        };

        // uint64_t last_sequence_number{};

        std::priority_queue<PairType> price_heap;
        std::unordered_map<uint32_t, PriceLevelPtr> price_map;
        std::unordered_map<uint64_t, std::list<msg::md::NewOrder>::iterator> order_id_map;

        std::optional<PriceLevelPtr> get_best_price_level_ptr();
        void delete_price_level(const PriceLevelPtr &price_level);

    public:
        size_t total_volume_traded;

        SideOrderBook(uint32_t symbol, msg::SIDE side);
        bool apply_message(const msg::md::MdMessage &message);
        std::optional<PriceLevel> get_best_price_level();
        bool insert_order(const msg::md::NewOrder &new_order);
        bool apply_trade_summary(const msg::md::TradeSummary &trade_summary);
        bool delete_order(const msg::md::DeleteOrder &delete_order);
    };


    /**
     * Holds orders for a single symbol
     */
    class SymbolOrderBook {
        // uint64_t last_sequence_number{};
        std::pair<SideOrderBook, SideOrderBook> side_order_books;

        bool modify_order(const msg::md::ModifyOrder &modify_order);

    public:
        const uint32_t symbol;
        explicit SymbolOrderBook(uint32_t symbol);
        bool apply_message(const msg::md::MdMessage &message);
        std::optional<PriceLevel> get_best_price_level(msg::SIDE side);
        [[nodiscard]] size_t get_total_volume_traded() const;
        void synchronize(const std::list<msg::md::MdMessage> &replay_seq, uint64_t last_seq_num, const std::list<msg::md::MdMessage> &backlog);
    };

    /**
     * Holds orders for multiple symbols
     */
    class MultiSymbolOrderBook {
        uint64_t last_sequence_number{};
        std::vector<SymbolOrderBook> symbol_books;

    public:
        explicit MultiSymbolOrderBook(size_t symbol_count);
        bool apply_message(const msg::md::MdMessage &message);
        std::optional<PriceLevel> get_best_price_level(uint32_t symbol, msg::SIDE side);
        [[nodiscard]] size_t get_total_volume_traded() const;
        void dump_update();
        void synchronize(const mcast::MulticastListener &live, const mcast::MulticastListener &snapshot);
    };

}

#endif //TRADING_PROJECT_ORDERBOOK_H
