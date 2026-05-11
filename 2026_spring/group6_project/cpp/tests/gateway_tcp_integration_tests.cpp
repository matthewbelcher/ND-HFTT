#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "ghostbook/protocol.h"
#include "tcp_gateway.h"

using ghostbook::gateway::TcpGateway;

namespace {

bool expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        return false;
    }
    return true;
}

bool recv_exact(int fd, std::uint8_t* buf, std::size_t len, int timeout_ms) {
    std::size_t read_total = 0;
    while (read_total < len) {
        pollfd pfd{fd, POLLIN, 0};
        const int poll_ret = poll(&pfd, 1, timeout_ms);
        if (poll_ret <= 0) {
            return false;
        }

        const ssize_t n = recv(fd, buf + read_total, len - read_total, 0);
        if (n <= 0) {
            return false;
        }
        read_total += static_cast<std::size_t>(n);
    }
    return true;
}

std::vector<std::uint8_t> build_frame(
    ghostbook::protocol::MessageType type,
    const std::uint8_t* body,
    std::uint16_t body_len,
    std::uint64_t seq_no,
    std::uint32_t session_id) {
    const auto header = ghostbook::protocol::make_frame_header(type, body_len, seq_no, 0, session_id);
    std::vector<std::uint8_t> frame(ghostbook::protocol::header_size + body_len);
    std::memcpy(frame.data(), &header, sizeof(header));
    if (body_len > 0 && body != nullptr) {
        std::memcpy(frame.data() + ghostbook::protocol::header_size, body, body_len);
    }
    return frame;
}

bool read_frame(
    int fd,
    ghostbook::protocol::FrameHeader* header,
    std::vector<std::uint8_t>* body,
    int timeout_ms) {
    if (!recv_exact(fd, reinterpret_cast<std::uint8_t*>(header), sizeof(*header), timeout_ms)) {
        return false;
    }

    body->assign(header->body_len, 0);
    if (header->body_len > 0) {
        if (!recv_exact(fd, body->data(), body->size(), timeout_ms)) {
            return false;
        }
    }

    return true;
}

bool test_logon_new_cancel_end_to_end() {
    constexpr std::uint16_t kPort = 23134;
    TcpGateway gateway;
    if (!gateway.start(kPort)) {
        return false;
    }

    int client_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (client_fd < 0) {
        gateway.shutdown();
        return false;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(kPort);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    bool connected = false;
    for (int i = 0; i < 20; ++i) {
        if (connect(client_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
            connected = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }

    if (!expect(connected, "client should connect to gateway")) {
        close(client_fd);
        gateway.shutdown();
        return false;
    }

    ghostbook::protocol::LogonBody logon{};
    logon.comp_id = 10;
    logon.heartbeat_interval_ms = 500;
    logon.app_instance = 42;

    auto logon_frame = build_frame(
        ghostbook::protocol::MessageType::Logon,
        reinterpret_cast<const std::uint8_t*>(&logon),
        static_cast<std::uint16_t>(sizeof(logon)),
        1,
        0);
    (void)send(client_fd, logon_frame.data(), logon_frame.size(), 0);

    ghostbook::protocol::FrameHeader rx_header{};
    std::vector<std::uint8_t> rx_body;
    bool ok = true;

    ok &= expect(read_frame(client_fd, &rx_header, &rx_body, 3000), "logon response frame should be received");
    ok &= expect(static_cast<ghostbook::protocol::MessageType>(rx_header.msg_type) ==
                     ghostbook::protocol::MessageType::Ack,
                 "logon should return Ack frame");
    ok &= expect(rx_body.size() == sizeof(ghostbook::protocol::ExecutionReportBody),
                 "logon ack body size should match execution report");

    auto* logon_ack = reinterpret_cast<const ghostbook::protocol::ExecutionReportBody*>(rx_body.data());
    ok &= expect(logon_ack->exec_type == static_cast<std::uint8_t>(ghostbook::protocol::ExecType::Ack),
                 "logon ack exec_type should be Ack");

    const std::uint32_t session_id = rx_header.session_id;
    ok &= expect(session_id != 0, "logon should assign non-zero session id");

    ghostbook::protocol::NewOrderBody new_order{};
    new_order.client_order_id = 101;
    new_order.instrument_id = 1;
    new_order.side = static_cast<std::uint8_t>(ghostbook::protocol::Side::Buy);
    new_order.order_type = static_cast<std::uint8_t>(ghostbook::protocol::OrderType::Limit);
    new_order.time_in_force = static_cast<std::uint8_t>(ghostbook::protocol::TimeInForce::Day);
    new_order.price_i64 = 100;
    new_order.qty_u32 = 3;

    auto new_order_frame = build_frame(
        ghostbook::protocol::MessageType::NewOrder,
        reinterpret_cast<const std::uint8_t*>(&new_order),
        static_cast<std::uint16_t>(sizeof(new_order)),
        2,
        session_id);
    (void)send(client_fd, new_order_frame.data(), new_order_frame.size(), 0);

    ok &= expect(read_frame(client_fd, &rx_header, &rx_body, 3000), "new order response frame should be received");
    ok &= expect(static_cast<ghostbook::protocol::MessageType>(rx_header.msg_type) ==
                     ghostbook::protocol::MessageType::Ack,
                 "new order should return Ack frame");

    auto* new_ack = reinterpret_cast<const ghostbook::protocol::ExecutionReportBody*>(rx_body.data());
    ok &= expect(new_ack->client_order_id == 101, "new order ack should reference client order id");
    ok &= expect(new_ack->exec_type == static_cast<std::uint8_t>(ghostbook::protocol::ExecType::Ack),
                 "new order exec_type should be Ack");

    ghostbook::protocol::CancelOrderBody cancel{};
    cancel.client_order_id = 102;
    cancel.target_order_id = 101;
    cancel.cancel_qty_u32 = 0;

    auto cancel_frame = build_frame(
        ghostbook::protocol::MessageType::CancelOrder,
        reinterpret_cast<const std::uint8_t*>(&cancel),
        static_cast<std::uint16_t>(sizeof(cancel)),
        3,
        session_id);
    (void)send(client_fd, cancel_frame.data(), cancel_frame.size(), 0);

    ok &= expect(read_frame(client_fd, &rx_header, &rx_body, 3000), "cancel response frame should be received");
    ok &= expect(static_cast<ghostbook::protocol::MessageType>(rx_header.msg_type) ==
                     ghostbook::protocol::MessageType::Cancel,
                 "cancel should return Cancel frame");

    auto* cancel_report = reinterpret_cast<const ghostbook::protocol::ExecutionReportBody*>(rx_body.data());
    ok &= expect(cancel_report->client_order_id == 101, "cancel report should reference target order id");
    ok &= expect(cancel_report->exec_type == static_cast<std::uint8_t>(ghostbook::protocol::ExecType::Cancel),
                 "cancel exec_type should be Cancel");

    ghostbook::protocol::NewOrderBody invalid_order{};
    invalid_order.client_order_id = 103;
    invalid_order.instrument_id = 1;
    invalid_order.side = static_cast<std::uint8_t>(ghostbook::protocol::Side::Buy);
    invalid_order.order_type = static_cast<std::uint8_t>(ghostbook::protocol::OrderType::Limit);
    invalid_order.time_in_force = static_cast<std::uint8_t>(ghostbook::protocol::TimeInForce::Day);
    invalid_order.price_i64 = 100;
    invalid_order.qty_u32 = 0;

    auto invalid_order_frame = build_frame(
        ghostbook::protocol::MessageType::NewOrder,
        reinterpret_cast<const std::uint8_t*>(&invalid_order),
        static_cast<std::uint16_t>(sizeof(invalid_order)),
        4,
        session_id);
    (void)send(client_fd, invalid_order_frame.data(), invalid_order_frame.size(), 0);

    ok &= expect(read_frame(client_fd, &rx_header, &rx_body, 3000), "invalid order response frame should be received");
    ok &= expect(static_cast<ghostbook::protocol::MessageType>(rx_header.msg_type) ==
                     ghostbook::protocol::MessageType::Reject,
                 "invalid order should return Reject frame");

    auto* reject_report = reinterpret_cast<const ghostbook::protocol::ExecutionReportBody*>(rx_body.data());
    ok &= expect(reject_report->client_order_id == 103, "reject report should reference invalid order id");
    ok &= expect(reject_report->exec_type == static_cast<std::uint8_t>(ghostbook::protocol::ExecType::Reject),
                 "invalid order exec_type should be Reject");
    ok &= expect(reject_report->ord_status == static_cast<std::uint8_t>(ghostbook::protocol::OrdStatus::Rejected),
                 "invalid order ord_status should be Rejected");
    ok &= expect(reject_report->code_u16 != 0, "invalid order reject should include non-zero code");

    close(client_fd);
    gateway.shutdown();
    return ok;
}

}  // namespace

int main() {
    const bool ok = test_logon_new_cancel_end_to_end();
    if (!ok) {
        return 1;
    }

    std::cout << "All gateway TCP integration tests passed" << std::endl;
    return 0;
}
