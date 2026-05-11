#pragma once

#include <cstdint>
#include <cstring>
#include <array>

namespace ghostbook::protocol {

// Message type enum
enum class MessageType : std::uint16_t {
    // Session
    Logon = 100,
    Logoff = 101,
    HeartBeat = 102,
    SequenceReset = 103,
    // Order entry
    NewOrder = 200,
    CancelOrder = 201,
    ModifyOrder = 202,
    // Execution reports
    Ack = 300,
    Fill = 301,
    PartialFill = 302,
    Reject = 303,
    Cancel = 304,
    // Market data
    AddOrder = 400,
    DeleteOrder = 401,
    ModifyOrderMd = 402,
    Trade = 403,
    BBO = 404,
};

// Field enums
enum class Side : std::uint8_t { Buy = 1, Sell = 2 };
enum class OrderType : std::uint8_t { Limit = 1, Market = 2 };
enum class TimeInForce : std::uint8_t { Day = 1, IOC = 2, FOK = 3, PostOnly = 4 };
enum class ExecType : std::uint8_t { Ack = 1, Fill = 2, PartialFill = 3, Cancel = 4, Reject = 5 };
enum class OrdStatus : std::uint8_t { New = 1, PartiallyFilled = 2, Filled = 3, Canceled = 4, Rejected = 5 };
enum class LiquidityFlag : std::uint8_t { Maker = 1, Taker = 2 };
enum class Capacity : std::uint8_t { Principal = 1, Agent = 2, Riskless = 3 };

// Common frame header (32 bytes)
struct FrameHeader {
    std::uint16_t msg_type;
    std::uint8_t msg_version;
    std::uint8_t flags;
    std::uint16_t body_len;
    std::uint16_t header_len;
    std::uint64_t seq_no;
    std::uint64_t logical_clock;
    std::uint32_t session_id;
    std::uint32_t crc32;
} __attribute__((packed));

// Logon body (48 bytes)
struct LogonBody {
    std::uint16_t comp_id;
    std::array<std::uint8_t, 16> password_hash;
    std::uint16_t heartbeat_interval_ms;
    std::uint32_t requested_session;
    std::uint32_t capabilities;
    std::uint32_t client_ip_v4;
    std::uint64_t app_instance;
    std::uint64_t reserved;
} __attribute__((packed));

// Logoff body (16 bytes)
struct LogoffBody {
    std::uint16_t reason_code;
    std::uint16_t reserved0;
    std::uint32_t text_code;
    std::uint64_t reserved1;
} __attribute__((packed));

// HeartBeat body (8 bytes)
struct HeartBeatBody {
    std::uint64_t last_recv_seq;
} __attribute__((packed));

// SequenceReset body (24 bytes)
struct SequenceResetBody {
    std::uint64_t new_seq_no;
    std::uint16_t reason_code;
    std::uint16_t reserved0;
    std::uint32_t reserved1;
    std::uint64_t reserved2;
} __attribute__((packed));

// NewOrder body (48 bytes)
struct NewOrderBody {
    std::uint64_t client_order_id;
    std::uint32_t instrument_id;
    std::uint8_t side;
    std::uint8_t order_type;
    std::uint8_t time_in_force;
    std::uint8_t capacity;
    std::int64_t price_i64;
    std::uint32_t qty_u32;
    std::uint32_t max_floor_u32;
    std::uint64_t expire_clock;
    std::uint32_t user_tag;
    std::uint32_t reserved;
} __attribute__((packed));

// CancelOrder body (24 bytes)
struct CancelOrderBody {
    std::uint64_t client_order_id;
    std::uint64_t target_order_id;
    std::uint32_t cancel_qty_u32;
    std::uint16_t reason_code;
    std::uint16_t reserved;
} __attribute__((packed));

// ModifyOrder body (40 bytes)
struct ModifyOrderBody {
    std::uint64_t orig_client_order_id;
    std::uint64_t new_client_order_id;
    std::int64_t new_price_i64;
    std::uint32_t new_qty_u32;
    std::uint8_t side;
    std::uint8_t replace_flags;
    std::uint16_t reserved0;
    std::uint32_t user_tag;
    std::uint32_t reserved1;
} __attribute__((packed));

// Execution report body (48 bytes) - used for Ack, Fill, PartialFill, Reject, Cancel
struct ExecutionReportBody {
    std::uint64_t client_order_id;
    std::uint64_t exchange_order_id;
    std::uint32_t instrument_id;
    std::uint8_t exec_type;
    std::uint8_t ord_status;
    std::uint8_t side;
    std::uint8_t liquidity_flag;
    std::uint32_t last_qty_u32;
    std::int64_t last_price_i64;
    std::uint32_t leaves_qty_u32;
    std::uint32_t cum_qty_u32;
    std::uint16_t code_u16;
    std::uint16_t reserved;
} __attribute__((packed));

// AddOrder body (40 bytes)
struct AddOrderMdBody {
    std::uint64_t md_order_id;
    std::uint32_t instrument_id;
    std::uint8_t side;
    std::uint8_t reserved0;
    std::uint16_t reserved1;
    std::int64_t price_i64;
    std::uint32_t qty_u32;
    std::uint16_t level_u16;
    std::uint16_t reserved2;
    std::uint32_t participant_id;
    std::uint32_t reserved3;
} __attribute__((packed));

// DeleteOrder body (24 bytes)
struct DeleteOrderMdBody {
    std::uint64_t md_order_id;
    std::uint32_t instrument_id;
    std::uint8_t side;
    std::uint8_t reserved0;
    std::uint16_t reserved1;
    std::uint32_t remaining_qty_u32;
    std::uint32_t reserved2;
} __attribute__((packed));

// ModifyOrderMd body (32 bytes)
struct ModifyOrderMdBody {
    std::uint64_t md_order_id;
    std::uint32_t instrument_id;
    std::uint8_t side;
    std::uint8_t reserved0;
    std::uint16_t reserved1;
    std::int64_t new_price_i64;
    std::uint32_t new_qty_u32;
    std::uint32_t reserved2;
} __attribute__((packed));

// Trade body (32 bytes)
struct TradeBody {
    std::uint64_t md_order_id;
    std::uint32_t instrument_id;
    std::uint8_t side;
    std::uint8_t reserved0;
    std::uint16_t reserved1;
    std::int64_t price_i64;
    std::uint32_t qty_u32;
    std::uint32_t reserved2;
} __attribute__((packed));

// BBO body (32 bytes)
struct BboBody {
    std::uint32_t instrument_id;
    std::uint8_t reserved0;
    std::uint8_t reserved1;
    std::uint16_t reserved2;
    std::int64_t bid_price_i64;
    std::int64_t ask_price_i64;
    std::uint32_t bid_qty_u32;
    std::uint32_t ask_qty_u32;
} __attribute__((packed));

// Helper for calculating frame size
static constexpr std::uint16_t header_size = sizeof(FrameHeader);

// Helper functions for building messages
inline FrameHeader make_frame_header(
    MessageType msg_type,
    std::uint16_t body_len,
    std::uint64_t seq_no,
    std::uint64_t logical_clock,
    std::uint32_t session_id,
    std::uint8_t flags = 0
) {
    return FrameHeader{
        .msg_type = static_cast<std::uint16_t>(msg_type),
        .msg_version = 1,
        .flags = flags,
        .body_len = body_len,
        .header_len = header_size,
        .seq_no = seq_no,
        .logical_clock = logical_clock,
        .session_id = session_id,
        .crc32 = 0,  // Will be calculated by caller
    };
}

}  // namespace ghostbook::protocol
