#pragma once

#include <cstddef>
#include <cstdint>

namespace ghostbook::ndfex {

/* ----- Common ----- */

/* Type aliases */

using seq_num_t = std::uint32_t;
using timestamp_t = std::uint64_t;
using msg_type_t = std::uint8_t;
using order_id_t = std::uint64_t;
using symbol_id_t = std::uint32_t;
using quantity_t = std::uint32_t;
using price_t = std::int32_t;
using order_flags_t = std::uint8_t;
using status_t = std::uint8_t;

/* Enums */

enum class SIDE : std::uint8_t { BUY = 1, SELL = 2 };

/* ----- Market data ----- */
namespace md {

/* Constants */

static const std::uint64_t MAGIC_NUMBER =
    *reinterpret_cast<const std::uint64_t *>("GOIRISH!");
static const std::uint64_t SNAPSHOT_MAGIC_NUMBER =
    *reinterpret_cast<const std::uint64_t *>("SNAPSHOT");

/* Enums */

enum class MSG_TYPE : std::uint8_t {
  HEARTBEAT = 0,
  NEW_ORDER = 1,
  DELETE_ORDER = 2,
  MODIFY_ORDER = 3,
  TRADE = 4,
  TRADE_SUMMARY = 5,
  SNAPSHOT_INFO = 6,
};

/* Message structs */

struct Header {
  std::uint64_t magic_number;
  std::uint16_t length;
  seq_num_t seq_num;
  timestamp_t timestamp;

  MSG_TYPE msg_type;
} __attribute__((packed));

struct NewOrder {
  order_id_t order_id;
  symbol_id_t symbol;
  SIDE side;
  quantity_t quantity;
  price_t price;
  order_flags_t flags;
} __attribute__((packed));

struct DeleteOrder {
  order_id_t order_id;
} __attribute__((packed));

struct ModifyOrder {
  order_id_t order_id;
  SIDE side;
  quantity_t quantity;
  price_t price;
} __attribute__((packed));

struct Trade {
  order_id_t order_id;
  quantity_t quantity;
  price_t price;
} __attribute__((packed));

struct TradeSummary {
  symbol_id_t symbol;
  SIDE aggressor_side;
  quantity_t total_quantity;
  price_t last_price;
} __attribute__((packed));

struct SnapshotInfo {
  symbol_id_t symbol;
  quantity_t bid_count;
  quantity_t ask_count;
  seq_num_t last_md_seq_num;
} __attribute__((packed));

union Body {
  NewOrder new_order;
  DeleteOrder delete_order;
  ModifyOrder modify_order;
  Trade trade;
  TradeSummary trade_summary;
  SnapshotInfo snapshot_info;
};

struct Message {
  Header header;
  Body body;
} __attribute__((packed));

/* Helper functions */

static constexpr std::size_t header_size = sizeof(Header);

} // namespace md

/* ----- Order entry ----- */
namespace oe {

/* Type Aliases */

using client_id_t = std::uint32_t;
using session_id_t = std::uint64_t;
using error_code_t = std::uint8_t;

/* Constants */

static const std::uint8_t OE_PROTOCOL_VERSION = 1;

/* Enums */

enum class MSG_TYPE : msg_type_t {
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

enum class REJECT_REASON : status_t {
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

enum class LOGIN_STATUS : status_t {
  SUCCESS = 0,

  // login reject reasons
  INVALID_USERNAME = 5,
  INVALID_PASSWORD = 6,
  INVALID_SESSION = 7,
  SESSION_ALREADY_ACTIVE = 8,
  DUPLICATE_LOGIN = 9,
  INVALID_CLIENT_ID = 10,
};

enum class ORDER_FLAGS : order_flags_t {
  NONE = 0,
  IOC = 1,
};

enum class FILL_FLAGS : order_flags_t {
  NONE = 0,
  PARTIAL_FILL = 1,
  CLOSED = 2,
};

/* Message structures */

struct RequestHeader {
  std::uint16_t length;
  MSG_TYPE msg_type;
  std::uint8_t version;
  seq_num_t seq_num;
  client_id_t client_id;
  session_id_t session_id;
} __attribute__((packed));

struct ResponseHeader {
  std::uint16_t length;
  MSG_TYPE msg_type;
  std::uint8_t version;
  seq_num_t seq_num;
  seq_num_t last_seq_num;
  client_id_t client_id;
} __attribute__((packed));

struct LoginRequest {
  std::uint8_t username[16];
  std::uint8_t password[16];
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
  std::uint16_t error_message_length;
  std::uint8_t error_message[32];
} __attribute__((packed));

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

struct Request {
  RequestHeader header;
  RequestBody body;
} __attribute__((packed));

struct Response {
  ResponseHeader header;
  ResponseBody body;
} __attribute__((packed));

/* Helper functions */

static constexpr std::size_t request_header_size = sizeof(RequestHeader);
static constexpr std::size_t response_header_sise = sizeof(ResponseHeader);

} // namespace oe

} // namespace ghostbook::ndfex
