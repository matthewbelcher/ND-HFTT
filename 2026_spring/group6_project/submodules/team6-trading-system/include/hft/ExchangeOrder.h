#ifndef HFT_EXCHANGEORDER_H
#define HFT_EXCHANGEORDER_H

#include <unordered_map>

#include "ExchangeClient.h"
#include "msg/common.h"
#include "msg/order_entry.h"

namespace hft::order {

    /* Enums */

    enum class ORDER_STATUS : uint8_t {
        LOCAL_ONLY,
        IN_FLIGHT,
        UPDATE_IN_FLIGHT,
        CANCEL_IN_FLIGHT,
        OPEN,
        CLOSED,
    };


    /* Class Definitions */

    /**
     * A dynamic object representing a limit order posted to the exchange
     */
    class ExchangeOrder {
    private:
        client::ExchangeClient &client;

        void handle_ack(const msg::oe::OrderAckResponse &resp);
        void handle_fill(const msg::oe::OrderFillResponse &resp);
        void handle_close(const msg::oe::OrderClosedResponse &resp);
        void handle_reject(const msg::oe::OrderRejectResponse &resp);

    public:
        msg::oe::NewOrderRequest local_state{};
        msg::oe::NewOrderRequest exchange_state{};
        ORDER_STATUS status = ORDER_STATUS::LOCAL_ONLY;
        msg::quantity_t fill_quantity = 0;
        std::unordered_map<msg::price_t, msg::quantity_t> fills{};

        ExchangeOrder(
            client::ExchangeClient &client,
            msg::order_id_t personal_order_id,
            msg::symbol_id_t symbol,
            msg::SIDE side,
            msg::quantity_t quantity,
            msg::price_t price,
            msg::oe::ORDER_FLAGS order_flags = msg::oe::ORDER_FLAGS::NONE);

        ~ExchangeOrder();

        bool commit();
        void on_response(const msg::oe::Response &response);

        void modify(msg::SIDE new_side, msg::quantity_t new_quantity, msg::price_t new_price);

        void cancel();
    };
}

#endif //HFT_EXCHANGEORDER_H
