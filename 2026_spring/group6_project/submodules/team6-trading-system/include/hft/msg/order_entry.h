/*
 * NDFEX Protocol order entry definitions
 */

#ifndef HFT_MSG_ORDER_ENTRY_H
#define HFT_MSG_ORDER_ENTRY_H

#include <cstdint>
#include <exception>

#include "common.h"


namespace hft::msg::oe {

    /* Utility macro functions */

    #define OE_REQUEST_SIZE(_req_type)  (sizeof(hft::msg::oe::RequestHeader) + sizeof(_req_type))
    #define OE_RESPONSE_SIZE(_rsp_type) (sizeof(hft::msg::oe::ResponseHeader) + sizeof(_rsp_type))

    /* Constants */

    constexpr uint8_t OE_PROTOCOL_VERSION = 1;


    /* Type Aliases */

    using client_id_t = uint32_t;
    using session_id_t = uint64_t;
    using error_code_t = uint8_t;


    /* Enum Classes */

    enum class MSG_TYPE: msg_type_t {
        // Requests
        NEW_ORDER = 1,
        DELETE_ORDER = 2,
        MODIFY_ORDER = 3,

        // Login
        LOGIN = 99,
        LOGIN_RESPONSE = 100,

        // Responses
        ACK = 101,
        REJECT = 102,
        FILL = 103,
        CLOSE = 104,
        ERROR = 105
    };

    enum class REJECT_REASON: status_t {
        NONE = 0,
        UNKNOWN_SYMBOL = 1,
        INVALID_ORDER = 2,
        DUPLICATE_ORDER_ID = 3,
        UNKNOWN_ORDER_ID = 4,
        INVALID_PRICE = 5,
        INVALID_QUANTITY = 6,
        INVALID_SIDE = 7,
        UNKNOWN_SESSION_ID = 8,
        DUPLICATE_LOGIN = 9,
    };

    enum class LOGIN_STATUS: status_t {
        SUCCESS = 0,

        // login reject reasons
        INVALID_USERNAME = 5,
        INVALID_PASSWORD = 6,
        INVALID_SESSION = 7,
        SESSION_ALREADY_ACTIVE = 8,
        DUPLICATE_LOGIN = 9,
        INVALID_CLIENT_ID = 10,
    };

    enum class ORDER_FLAGS: order_flags_t {
        NONE = 0,
        IOC = 1,
    };

    enum class FILL_FLAGS: order_flags_t {
        NONE = 0,
        PARTIAL_FILL = 1,
        CLOSED = 2,
    };


    /* Message header structures */

    struct RequestHeader {
        uint16_t length;
        MSG_TYPE msg_type;
        uint8_t version;
        seq_num_t seq_num;
        client_id_t client_id;
        session_id_t session_id;
    } __attribute__((packed));

    struct ResponseHeader {
        uint16_t length;
        MSG_TYPE msg_type;
        uint8_t version;
        seq_num_t seq_num;
        seq_num_t last_seq_num;
        client_id_t client_id;
    } __attribute__((packed));


    /* Order entry request/response message bodies */

    struct LoginRequest {
        uint8_t username[16];
        uint8_t password[16];
    } __attribute__((packed));

    struct LoginResponse {
        session_id_t session_id;
        LOGIN_STATUS status;
    } __attribute__((packed));

    struct NewOrderRequest {
        order_id_t order_id;
        symbol_id_t symbol;
        SIDE side;
        quantity_t quantity;
        price_t price;
        ORDER_FLAGS flags;
    } __attribute__((packed));

    struct DeleteOrderRequest {
        order_id_t order_id;
    } __attribute__((packed));

    struct ModifyOrderRequest {
        order_id_t order_id;
        SIDE side;
        quantity_t quantity;
        price_t price;
    } __attribute__((packed));

    struct OrderAckResponse {
        order_id_t order_id;
        order_id_t exch_order_id;
        quantity_t quantity;
        price_t price;
    } __attribute__((packed));

    struct OrderRejectResponse {
        order_id_t order_id;
        REJECT_REASON reject_reason;
    } __attribute__((packed));

    struct OrderFillResponse {
        order_id_t order_id;
        quantity_t quantity;
        price_t price;
        FILL_FLAGS flags;
    } __attribute__((packed));

    struct OrderClosedResponse {
        order_id_t order_id;
    } __attribute__((packed));

    struct ErrorResponse {
        error_code_t error_code;
        uint16_t error_message_length;
        uint8_t error_message[32];
    } __attribute__((packed));


    /* Request and response body unions */

    union RequestBody {
        LoginRequest login_request;
        NewOrderRequest new_order_request;
        ModifyOrderRequest modify_order_request;
        DeleteOrderRequest delete_order_request;
    } __attribute__((packed));

    union ResponseBody {
        LoginResponse login_response;
        OrderAckResponse order_ack_response;
        OrderRejectResponse order_reject_response;
        OrderFillResponse order_fill_response;
        OrderClosedResponse order_closed_response;
        ErrorResponse error_response;
    } __attribute__((packed));


    /* Full request and response structures */

    struct Request {
        RequestHeader header;
        RequestBody body;
    } __attribute__((packed));

    struct Response {
        ResponseHeader header;
        ResponseBody body;
    } __attribute__((packed));


    /* Function headers */

    const char *    reject_reason_string(REJECT_REASON reason);
    const char *    login_status_string(LOGIN_STATUS status);
    size_t          get_request_size(MSG_TYPE msg_type);
    void            dump_oe_request(const Request *req);
    void            dump_oe_response(const Response *rsp);
}

#endif //HFT_MSG_ORDER_ENTRY_H