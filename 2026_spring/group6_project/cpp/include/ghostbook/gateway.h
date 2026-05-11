#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "protocol.h"

namespace ghostbook::gateway {

using order_id_t = std::uint64_t;
using instrument_id_t = std::uint32_t;
using logical_clock_t = std::uint64_t;
using session_id_t = std::uint32_t;

// Session state
struct SessionState {
    session_id_t session_id{};
    std::uint16_t comp_id{};
    std::uint64_t last_seq_received{};
    std::uint64_t last_seq_sent{};
    std::uint64_t app_instance{};
    std::uint32_t client_ip{};
    std::uint16_t heartbeat_interval_ms{1000};
    bool logged_in{false};
    std::map<order_id_t, order_id_t> client_to_exchange_order_id;  // Maps client_order_id to exchange_order_id
};

// Gateway statistics
struct GatewayStats {
    std::uint64_t total_messages_received{};
    std::uint64_t total_orders_submitted{};
    std::uint64_t total_cancels_submitted{};
    std::uint64_t total_modifies_submitted{};
    std::uint64_t total_rejections{};
    std::uint64_t protocol_errors{};
};

// Order info for tracking
struct OrderInfo {
    order_id_t client_order_id{};
    order_id_t exchange_order_id{};
    instrument_id_t instrument_id{};
    protocol::Side side{protocol::Side::Buy};
    std::uint32_t original_qty{};
    std::uint32_t filled_qty{};
    std::uint32_t remaining_qty{};
};

// Frame buffer for receiving/sending
struct FrameBuffer {
    static constexpr std::size_t MAX_FRAME_SIZE = 4096;
    std::uint8_t data[MAX_FRAME_SIZE]{};
    std::size_t size{};
    std::size_t pos{};

    void reset() {
        size = 0;
        pos = 0;
    }

    void append(const std::uint8_t* buf, std::size_t len) {
        if (size + len > MAX_FRAME_SIZE) {
            throw std::runtime_error("Frame buffer overflow");
        }
        std::memcpy(data + size, buf, len);
        size += len;
    }

    std::optional<std::pair<protocol::FrameHeader, std::vector<std::uint8_t>>> try_read_frame() {
        if (size < protocol::header_size) {
            return std::nullopt;
        }

        auto header_ptr = reinterpret_cast<const protocol::FrameHeader*>(data);
        std::size_t total_size = protocol::header_size + header_ptr->body_len;

        if (size < total_size) {
            return std::nullopt;
        }

        auto header = *header_ptr;
        std::vector<std::uint8_t> body(header.body_len);
        std::memcpy(body.data(), data + protocol::header_size, header.body_len);

        // Shift remaining data
        size -= total_size;
        if (size > 0) {
            std::memmove(data, data + total_size, size);
        }

        return std::make_pair(header, body);
    }

    std::vector<std::uint8_t> build_frame(
        protocol::MessageType msg_type,
        const std::uint8_t* body,
        std::uint16_t body_len,
        std::uint64_t seq_no,
        logical_clock_t logical_clock,
        session_id_t session_id
    ) {
        std::vector<std::uint8_t> result;
        result.resize(protocol::header_size + body_len);

        auto header = protocol::make_frame_header(msg_type, body_len, seq_no, logical_clock, session_id);
        std::memcpy(result.data(), &header, protocol::header_size);

        if (body_len > 0 && body != nullptr) {
            std::memcpy(result.data() + protocol::header_size, body, body_len);
        }

        return result;
    }
};

// Forward declarations for callback functions
using ExecutionReportCallback = std::function<void(
    session_id_t,
    const protocol::ExecutionReportBody&
)>;

using OrderAckCallback = std::function<void(
    session_id_t,
    order_id_t client_order_id,
    order_id_t exchange_order_id,
    logical_clock_t logical_clock
)>;

class SessionManager {
public:
    SessionManager() : next_session_id_(1) {}

    session_id_t create_session(
        std::uint16_t comp_id,
        std::uint64_t app_instance,
        std::uint32_t client_ip,
        std::uint16_t heartbeat_interval_ms
    ) {
        session_id_t sid = next_session_id_.fetch_add(1, std::memory_order_relaxed);
        SessionState state{};
        state.session_id = sid;
        state.comp_id = comp_id;
        state.app_instance = app_instance;
        state.client_ip = client_ip;
        state.heartbeat_interval_ms = heartbeat_interval_ms;
        state.logged_in = true;
        sessions_[sid] = state;
        return sid;
    }

    std::optional<SessionState> get_session(session_id_t session_id) {
        auto it = sessions_.find(session_id);
        if (it != sessions_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    void update_session(const SessionState& state) {
        sessions_[state.session_id] = state;
    }

    void close_session(session_id_t session_id) {
        sessions_.erase(session_id);
    }

    void map_order_id(session_id_t session_id, order_id_t client_id, order_id_t exchange_id) {
        if (auto s = get_session(session_id)) {
            s->client_to_exchange_order_id[client_id] = exchange_id;
            update_session(*s);
        }
    }

    std::optional<order_id_t> get_exchange_order_id(session_id_t session_id, order_id_t client_id) {
        if (auto s = get_session(session_id)) {
            auto it = s->client_to_exchange_order_id.find(client_id);
            if (it != s->client_to_exchange_order_id.end()) {
                return it->second;
            }
        }
        return std::nullopt;
    }

private:
    std::atomic<session_id_t> next_session_id_;
    std::map<session_id_t, SessionState> sessions_;
};

// Gateway interface for order processing
class Gateway {
public:
    explicit Gateway(logical_clock_t start_clock = 0)
        : logical_clock_(start_clock) {}
    virtual ~Gateway() = default;

    // Process incoming frame
    virtual void process_frame(
        session_id_t session_id,
        const protocol::FrameHeader& header,
        const std::vector<std::uint8_t>& body
    ) = 0;

    // Get current logical clock
    logical_clock_t get_logical_clock() const { return logical_clock_; }

    // Update logical clock (called by matching engine or replay engine)
    void set_logical_clock(logical_clock_t clock) { logical_clock_ = clock; }

    // Register callbacks
    void set_execution_report_callback(ExecutionReportCallback cb) {
        execution_report_cb_ = cb;
    }

    void set_order_ack_callback(OrderAckCallback cb) {
        order_ack_cb_ = cb;
    }

    // Get statistics
    const GatewayStats& get_stats() const { return stats_; }

protected:
    SessionManager session_mgr_;
    logical_clock_t logical_clock_;
    GatewayStats stats_;
    std::map<order_id_t, OrderInfo> order_tracking_;

    ExecutionReportCallback execution_report_cb_;
    OrderAckCallback order_ack_cb_;

    void emit_execution_report(
        session_id_t session_id,
        const protocol::ExecutionReportBody& report
    ) {
        if (execution_report_cb_) {
            execution_report_cb_(session_id, report);
        }
    }

    void emit_order_ack(
        session_id_t session_id,
        order_id_t client_order_id,
        order_id_t exchange_order_id,
        logical_clock_t clock
    ) {
        if (order_ack_cb_) {
            order_ack_cb_(session_id, client_order_id, exchange_order_id, clock);
        }
    }
};

}  // namespace ghostbook::gateway
