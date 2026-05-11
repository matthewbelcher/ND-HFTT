#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "ghostbook/protocol.h"
#include "tcp_gateway.h"

using ghostbook::gateway::TcpGateway;
using ghostbook::gateway::session_id_t;
using ghostbook::protocol::FrameHeader;
using ghostbook::protocol::MessageType;

namespace {

bool expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        return false;
    }
    return true;
}

struct CapturedReport {
    session_id_t session_id{};
    ghostbook::protocol::ExecutionReportBody report{};
};

struct CapturedAck {
    session_id_t session_id{};
    std::uint64_t client_order_id{};
    std::uint64_t exchange_order_id{};
    std::uint64_t logical_clock{};
};

FrameHeader make_header(MessageType type, std::uint16_t body_len) {
    return ghostbook::protocol::make_frame_header(type, body_len, 1, 10, 1);
}

bool test_new_order_ack_and_execution_report() {
    TcpGateway gateway;
    const auto session_id = gateway.create_test_session();

    std::vector<CapturedReport> reports;
    std::vector<CapturedAck> acks;

    gateway.set_execution_report_callback(
        [&reports](session_id_t sid, const ghostbook::protocol::ExecutionReportBody& report) {
            reports.push_back({sid, report});
        });

    gateway.set_order_ack_callback(
        [&acks](session_id_t sid, std::uint64_t client_id, std::uint64_t exch_id, std::uint64_t clock) {
            acks.push_back({sid, client_id, exch_id, clock});
        });

    ghostbook::protocol::NewOrderBody new_order{};
    new_order.client_order_id = 1001;
    new_order.instrument_id = 7;
    new_order.side = static_cast<std::uint8_t>(ghostbook::protocol::Side::Buy);
    new_order.order_type = static_cast<std::uint8_t>(ghostbook::protocol::OrderType::Limit);
    new_order.time_in_force = static_cast<std::uint8_t>(ghostbook::protocol::TimeInForce::Day);
    new_order.price_i64 = 100;
    new_order.qty_u32 = 5;

    gateway.process_frame(
        session_id,
        make_header(MessageType::NewOrder, static_cast<std::uint16_t>(sizeof(new_order))),
        std::vector<std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(&new_order),
            reinterpret_cast<const std::uint8_t*>(&new_order) + sizeof(new_order)));

    bool ok = true;
    ok &= expect(!acks.empty(), "new order should emit order ack callback");
    ok &= expect(!reports.empty(), "new order should emit execution report");
    ok &= expect(acks[0].session_id == session_id, "ack should target correct session");
    ok &= expect(acks[0].client_order_id == 1001, "ack should target submitted order id");
    ok &= expect(reports[0].report.client_order_id == 1001, "report should reference submitted order id");
    ok &= expect(reports[0].report.exec_type == static_cast<std::uint8_t>(ghostbook::protocol::ExecType::Ack),
                 "report should be ack");
    ok &= expect(gateway.get_stats().total_orders_submitted == 1, "stats should track submitted order");
    return ok;
}

bool test_reject_on_invalid_quantity() {
    TcpGateway gateway;
    const auto session_id = gateway.create_test_session();

    std::vector<CapturedReport> reports;
    gateway.set_execution_report_callback(
        [&reports](session_id_t sid, const ghostbook::protocol::ExecutionReportBody& report) {
            reports.push_back({sid, report});
        });

    ghostbook::protocol::NewOrderBody new_order{};
    new_order.client_order_id = 2001;
    new_order.instrument_id = 7;
    new_order.side = static_cast<std::uint8_t>(ghostbook::protocol::Side::Buy);
    new_order.order_type = static_cast<std::uint8_t>(ghostbook::protocol::OrderType::Limit);
    new_order.time_in_force = static_cast<std::uint8_t>(ghostbook::protocol::TimeInForce::Day);
    new_order.price_i64 = 101;
    new_order.qty_u32 = 0;

    gateway.process_frame(
        session_id,
        make_header(MessageType::NewOrder, static_cast<std::uint16_t>(sizeof(new_order))),
        std::vector<std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(&new_order),
            reinterpret_cast<const std::uint8_t*>(&new_order) + sizeof(new_order)));

    bool ok = true;
    ok &= expect(reports.size() == 1, "invalid order should emit one reject report");
    ok &= expect(reports[0].report.exec_type == static_cast<std::uint8_t>(ghostbook::protocol::ExecType::Reject),
                 "invalid quantity should reject");
    ok &= expect(gateway.get_stats().total_rejections == 1, "stats should track rejection");
    return ok;
}

bool test_cancel_known_order() {
    TcpGateway gateway;
    const auto session_id = gateway.create_test_session();

    std::vector<CapturedReport> reports;
    gateway.set_execution_report_callback(
        [&reports](session_id_t sid, const ghostbook::protocol::ExecutionReportBody& report) {
            reports.push_back({sid, report});
        });

    ghostbook::protocol::NewOrderBody new_order{};
    new_order.client_order_id = 3001;
    new_order.instrument_id = 7;
    new_order.side = static_cast<std::uint8_t>(ghostbook::protocol::Side::Buy);
    new_order.order_type = static_cast<std::uint8_t>(ghostbook::protocol::OrderType::Limit);
    new_order.time_in_force = static_cast<std::uint8_t>(ghostbook::protocol::TimeInForce::Day);
    new_order.price_i64 = 100;
    new_order.qty_u32 = 10;

    gateway.process_frame(
        session_id,
        make_header(MessageType::NewOrder, static_cast<std::uint16_t>(sizeof(new_order))),
        std::vector<std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(&new_order),
            reinterpret_cast<const std::uint8_t*>(&new_order) + sizeof(new_order)));

    ghostbook::protocol::CancelOrderBody cancel{};
    cancel.client_order_id = 3002;
    cancel.target_order_id = 3001;
    cancel.cancel_qty_u32 = 0;

    gateway.process_frame(
        session_id,
        make_header(MessageType::CancelOrder, static_cast<std::uint16_t>(sizeof(cancel))),
        std::vector<std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(&cancel),
            reinterpret_cast<const std::uint8_t*>(&cancel) + sizeof(cancel)));

    bool seen_cancel = false;
    for (const auto& captured : reports) {
        if (captured.report.exec_type == static_cast<std::uint8_t>(ghostbook::protocol::ExecType::Cancel) &&
            captured.report.client_order_id == 3001) {
            seen_cancel = true;
            break;
        }
    }

    bool ok = true;
    ok &= expect(seen_cancel, "cancel should emit cancel execution report for target order");
    ok &= expect(gateway.get_stats().total_cancels_submitted == 1, "stats should track cancel submissions");
    return ok;
}

bool test_modify_known_order() {
    TcpGateway gateway;
    const auto session_id = gateway.create_test_session();

    std::vector<CapturedReport> reports;
    std::vector<CapturedAck> acks;

    gateway.set_execution_report_callback(
        [&reports](session_id_t sid, const ghostbook::protocol::ExecutionReportBody& report) {
            reports.push_back({sid, report});
        });
    gateway.set_order_ack_callback(
        [&acks](session_id_t sid, std::uint64_t client_id, std::uint64_t exch_id, std::uint64_t clock) {
            acks.push_back({sid, client_id, exch_id, clock});
        });

    ghostbook::protocol::NewOrderBody new_order{};
    new_order.client_order_id = 4001;
    new_order.instrument_id = 7;
    new_order.side = static_cast<std::uint8_t>(ghostbook::protocol::Side::Buy);
    new_order.order_type = static_cast<std::uint8_t>(ghostbook::protocol::OrderType::Limit);
    new_order.time_in_force = static_cast<std::uint8_t>(ghostbook::protocol::TimeInForce::Day);
    new_order.price_i64 = 100;
    new_order.qty_u32 = 4;

    gateway.process_frame(
        session_id,
        make_header(MessageType::NewOrder, static_cast<std::uint16_t>(sizeof(new_order))),
        std::vector<std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(&new_order),
            reinterpret_cast<const std::uint8_t*>(&new_order) + sizeof(new_order)));

    ghostbook::protocol::ModifyOrderBody modify{};
    modify.orig_client_order_id = 4001;
    modify.new_client_order_id = 4002;
    modify.new_price_i64 = 101;
    modify.new_qty_u32 = 6;
    modify.side = static_cast<std::uint8_t>(ghostbook::protocol::Side::Buy);

    gateway.process_frame(
        session_id,
        make_header(MessageType::ModifyOrder, static_cast<std::uint16_t>(sizeof(modify))),
        std::vector<std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(&modify),
            reinterpret_cast<const std::uint8_t*>(&modify) + sizeof(modify)));

    bool seen_modify_ack = false;
    for (const auto& ack : acks) {
        if (ack.client_order_id == 4002) {
            seen_modify_ack = true;
            break;
        }
    }

    bool ok = true;
    ok &= expect(seen_modify_ack, "modify should ack replacement order id");
    ok &= expect(gateway.get_stats().total_modifies_submitted == 1, "stats should track modify submissions");
    return ok;
}

bool test_protocol_body_too_small_increments_error_count() {
    TcpGateway gateway;
    const auto session_id = gateway.create_test_session();

    gateway.process_frame(
        session_id,
        make_header(MessageType::NewOrder, 1),
        std::vector<std::uint8_t>{0x01});

    gateway.process_frame(
        session_id,
        make_header(MessageType::CancelOrder, 1),
        std::vector<std::uint8_t>{0x01});

    gateway.process_frame(
        session_id,
        make_header(MessageType::ModifyOrder, 1),
        std::vector<std::uint8_t>{0x01});

    return expect(gateway.get_stats().protocol_errors == 3, "small bodies should increment protocol error counter");
}

}  // namespace

int main() {
    bool ok = true;

    ok &= test_new_order_ack_and_execution_report();
    ok &= test_reject_on_invalid_quantity();
    ok &= test_cancel_known_order();
    ok &= test_modify_known_order();
    ok &= test_protocol_body_too_small_increments_error_count();

    if (!ok) {
        return 1;
    }

    std::cout << "All gateway tests passed" << std::endl;
    return 0;
}
