/*
 * Base class for trading strategies
 */

#ifndef HFT_STRATEGY_H
#define HFT_STRATEGY_H

#include <vector>

#include "ExchangeClient.h"
#include "PositionTracker.h"
#include "orderbook.h"
#include "msg/market_data.h"

namespace hft::strategy {

    /**
     * Abstract base class for trading strategies.
     *
     * Subclasses call subscribe_md() in their constructor for each symbol they
     * want to react to.  The base destructor automatically unsubscribes all
     * registered symbols, so there is no risk of the client calling back into a
     * destroyed strategy object.
     *
     * on_market_data() is called by the ExchangeClient whenever a market data
     * message arrives for a subscribed symbol.  on_tick() has a default no-op
     * implementation for strategies that are fully event-driven.
     */
    class Strategy {
    public:
        Strategy(client::ExchangeClient &client,
                 orderbook::MultiSymbolOrderBook &orderbook,
                 risk::PositionTracker &tracker)
            : client_(client), orderbook_(orderbook), tracker_(tracker) {}

        virtual ~Strategy() {
            for (const auto sym : subscribed_symbols_) {
                client_.unsubscribe_md(sym, this);
            }
        }

        // Called when a market data message arrives for a subscribed symbol.
        virtual void on_market_data(msg::symbol_id_t symbol,
                                    const msg::md::MdMessage &msg) = 0;

        // Called each iteration of the main loop.  Default no-op for strategies
        // that handle all logic inside on_market_data().
        virtual void on_tick() {}

    protected:
        client::ExchangeClient &client_;
        orderbook::MultiSymbolOrderBook &orderbook_;
        risk::PositionTracker &tracker_;

        // Register this strategy to receive on_market_data() calls for symbol.
        void subscribe_md(msg::symbol_id_t symbol) {
            client_.subscribe_md(symbol, this);
            subscribed_symbols_.push_back(symbol);
        }

    private:
        std::vector<msg::symbol_id_t> subscribed_symbols_;
    };

}

#endif //HFT_STRATEGY_H
