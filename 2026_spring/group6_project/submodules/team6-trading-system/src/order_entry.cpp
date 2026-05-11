#include "../include/hft/msg/order_entry.h"

#include <cinttypes>
#include <cstdio>


namespace hft::msg::oe {

const char *reject_reason_string(const REJECT_REASON reason) {
    switch (reason) {
        case REJECT_REASON::NONE:               return "NONE";
        case REJECT_REASON::UNKNOWN_SYMBOL:     return "UNKNOWN_SYMBOL";
        case REJECT_REASON::INVALID_ORDER:      return "INVALID_ORDER";
        case REJECT_REASON::DUPLICATE_ORDER_ID: return "DUPLICATE_ORDER_ID";
        case REJECT_REASON::UNKNOWN_ORDER_ID:   return "UNKNOWN_ORDER_ID";
        case REJECT_REASON::INVALID_PRICE:      return "INVALID_PRICE";
        case REJECT_REASON::INVALID_QUANTITY:   return "INVALID_QUANTITY";
        case REJECT_REASON::INVALID_SIDE:       return "INVALID_SIDE";
        case REJECT_REASON::UNKNOWN_SESSION_ID: return "UNKNOWN_SESSION_ID";
        case REJECT_REASON::DUPLICATE_LOGIN:    return "DUPLICATE_LOGIN";
        default:                                return "UNKNOWN";
    }
}

const char *login_status_string(const LOGIN_STATUS status) {
    switch (status) {
        case LOGIN_STATUS::SUCCESS:                return "SUCCESS";
        case LOGIN_STATUS::INVALID_USERNAME:       return "INVALID_USERNAME";
        case LOGIN_STATUS::INVALID_PASSWORD:       return "INVALID_PASSWORD";
        case LOGIN_STATUS::INVALID_SESSION:        return "INVALID_SESSION";
        case LOGIN_STATUS::SESSION_ALREADY_ACTIVE: return "SESSION_ALREADY_ACTIVE";
        case LOGIN_STATUS::DUPLICATE_LOGIN:        return "DUPLICATE_LOGIN";
        case LOGIN_STATUS::INVALID_CLIENT_ID:      return "INVALID_CLIENT_ID";
        default:                                   return "UNKNOWN";
    }
}

size_t get_request_size(const MSG_TYPE msg_type) {
    switch (msg_type) {
        case (MSG_TYPE::NEW_ORDER):     return sizeof(NewOrderRequest);
        case (MSG_TYPE::DELETE_ORDER):  return sizeof(DeleteOrderRequest);
        case (MSG_TYPE::MODIFY_ORDER):  return sizeof(ModifyOrderRequest);
        case (MSG_TYPE::LOGIN):         return sizeof(LoginRequest);
        default:                        return 0;
    }
}


/**
 * Dump order entry request information to standard output
 *
 * @param req Pointer to OERequest structure
 */
void dump_oe_request(const Request *req) {
    printf("=== Begin OE request (%p) ===\n", static_cast<const void *>(req));
    printf("-> Length: %" PRIu16 "\n", req->header.length);
    printf("-> Version: %" PRIu8 "\n", req->header.version);
    printf("-> Sequence number: %" PRIu32 "\n", req->header.seq_num);
    printf("-> Client ID: %" PRIu32 "\n", req->header.client_id);
    printf("-> Session ID: %" PRIu64 "\n", req->header.session_id);

    if (req->header.msg_type == MSG_TYPE::LOGIN) {
        printf("-> Message type: LOGIN\n");
        const auto &r = req->body.login_request;
        printf("-> Username: %.16s\n", reinterpret_cast<const char *>(r.username));
        printf("-> Password: %.16s\n", reinterpret_cast<const char *>(r.password));

    } else if (req->header.msg_type == MSG_TYPE::NEW_ORDER) {
        printf("-> Message type: NEW_ORDER\n");
        const auto [order_id, symbol, side, quantity, price, flags] = req->body.new_order_request;
        printf("-> Order ID: %" PRIu64 "\n", order_id);
        printf("-> Symbol: %" PRIu32 "\n", symbol);
        printf("-> Side: %s\n", side_string(side));
        printf("-> Quantity: %" PRIu32 "\n", quantity);
        printf("-> Price: %" PRId32 "\n", price);
        printf("-> Flags: %" PRIx8 "\n", static_cast<uint8_t>(flags));

    } else if (req->header.msg_type == MSG_TYPE::DELETE_ORDER) {
        printf("-> Message type: DELETE_ORDER\n");
        const auto [order_id] = req->body.delete_order_request;
        printf("-> Order ID: %" PRIu64 "\n", order_id);

    } else if (req->header.msg_type == MSG_TYPE::MODIFY_ORDER) {
        printf("-> Message type: MODIFY_ORDER\n");
        const auto [order_id, side, quantity, price] = req->body.modify_order_request;
        printf("-> Order ID: %" PRIu64 "\n", order_id);
        printf("-> Side: %s\n", side_string(side));
        printf("-> Quantity: %" PRIu32 "\n", quantity);
        printf("-> Price: %" PRId32 "\n", price);
    }

    printf("=== End OE request (%p) ===\n", static_cast<const void *>(req));
}


/**
 * Dump order entry response information to standard output
 *
 * @param rsp Pointer to OEResponse structure
 */
void dump_oe_response(const Response *rsp) {
    printf("=== Begin OE response (%p) ===\n", static_cast<const void *>(rsp));
    printf("-> Length: %" PRIu16 "\n", rsp->header.length);
    printf("-> Version: %" PRIu8 "\n", rsp->header.version);
    printf("-> Sequence number: %" PRIu32 "\n", rsp->header.seq_num);
    printf("-> Last sequence number: %" PRIu32 "\n", rsp->header.last_seq_num);
    printf("-> Client ID: %" PRIu32 "\n", rsp->header.client_id);

    const auto msg_type = static_cast<MSG_TYPE>(rsp->header.msg_type);

    if (msg_type == MSG_TYPE::LOGIN_RESPONSE) {
        printf("-> Message type: LOGIN_RESPONSE\n");
        const auto [session_id, status] = rsp->body.login_response;
        printf("-> Session ID: %" PRIu64 "\n", session_id);
        printf("-> Status: %s\n", login_status_string(status));

    } else if (msg_type == MSG_TYPE::ACK) {
        printf("-> Message type: ACK\n");
        const auto [order_id, exch_order_id, quantity, price] = rsp->body.order_ack_response;
        printf("-> Order ID: %" PRIu64 "\n", order_id);
        printf("-> Exchange order ID: %" PRIu64 "\n", exch_order_id);
        printf("-> Quantity: %" PRIu32 "\n", quantity);
        printf("-> Price: %" PRId32 "\n", price);

    } else if (msg_type == MSG_TYPE::REJECT) {
        printf("-> Message type: REJECT\n");
        const auto [order_id, reject_reason] = rsp->body.order_reject_response;
        printf("-> Order ID: %" PRIu64 "\n", order_id);
        printf("-> Reject reason: %s\n", reject_reason_string(reject_reason));

    } else if (msg_type == MSG_TYPE::FILL) {
        printf("-> Message type: FILL\n");
        const auto [order_id, quantity, price, flags] = rsp->body.order_fill_response;
        printf("-> Order ID: %" PRIu64 "\n", order_id);
        printf("-> Quantity: %" PRIu32 "\n", quantity);
        printf("-> Price: %" PRId32 "\n", price);
        printf("-> Flags: %" PRIx8 "\n", static_cast<uint8_t>(flags));

    } else if (msg_type == MSG_TYPE::CLOSE) {
        printf("-> Message type: CLOSE\n");
        const auto [order_id] = rsp->body.order_closed_response;
        printf("-> Order ID: %" PRIu64 "\n", order_id);

    } else if (msg_type == MSG_TYPE::ERROR) {
        printf("-> Message type: ERROR\n");
        const auto &e = rsp->body.error_response;
        printf("-> Error code: %" PRIu8 "\n", e.error_code);
        printf("-> Error message: %.*s\n", static_cast<int>(e.error_message_length),
               reinterpret_cast<const char *>(e.error_message));
    }

    printf("=== End OE response (%p) ===\n", static_cast<const void *>(rsp));
}

} // namespace hft::msg::oe
