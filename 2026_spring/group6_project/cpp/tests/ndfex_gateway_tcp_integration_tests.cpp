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

#include "ghostbook/ndfex/protocol.h"
#include "ndfex_gateway.h"

using ghostbook::gateway::NdfexGateway;
using ghostbook::ndfex::oe::MSG_TYPE;
using ghostbook::ndfex::oe::OE_PROTOCOL_VERSION;
using ghostbook::ndfex::oe::Request;
using ghostbook::ndfex::oe::RequestHeader;
using ghostbook::ndfex::oe::Response;
using ghostbook::ndfex::oe::ResponseHeader;

// =============================================================================
// TCP utilities
// =============================================================================

namespace {

bool expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "  FAIL  " << message << "\n";
        return false;
    }
    std::cout << "  PASS  " << message << "\n";
    return true;
}

template <typename T>
bool expect_eq(T actual, T expected, const std::string& label) {
    if (actual == expected) {
        std::cout << "  PASS  " << label << " == " << static_cast<std::uint64_t>(actual) << "\n";
        return true;
    }
    std::cerr << "  FAIL  " << label << ": expected " << static_cast<std::uint64_t>(expected)
              << ", got " << static_cast<std::uint64_t>(actual) << "\n";
    return false;
}

bool recv_exact(int fd, std::uint8_t* buf, std::size_t len, int timeout_ms) {
    std::size_t total = 0;
    while (total < len) {
        pollfd pfd{fd, POLLIN, 0};
        if (poll(&pfd, 1, timeout_ms) <= 0) { return false; }
        const ssize_t n = recv(fd, buf + total, len - total, 0);
        if (n <= 0) { return false; }
        total += static_cast<std::size_t>(n);
    }
    return true;
}

// Read a complete NDFEX OE response from a TCP socket.
// Returns false on timeout or read error.
bool read_oe_response(int fd, Response* out, int timeout_ms = 3000) {
    ResponseHeader hdr{};
    if (!recv_exact(fd, reinterpret_cast<std::uint8_t*>(&hdr), sizeof(hdr), timeout_ms)) {
        return false;
    }

    const std::size_t body_size = hdr.length > sizeof(hdr) ? hdr.length - sizeof(hdr) : 0;
    out->header = hdr;
    if (body_size > 0 && body_size <= sizeof(out->body)) {
        if (!recv_exact(fd, reinterpret_cast<std::uint8_t*>(&out->body), body_size, timeout_ms)) {
            return false;
        }
    }
    return true;
}

// Serialize and send an NDFEX OE request over a TCP socket.
// Sets the header.length field correctly before sending.
bool send_oe_request(int fd, Request req, std::size_t body_size) {
    req.header.length = static_cast<std::uint16_t>(sizeof(RequestHeader) + body_size);
    // Send header + body (only the used portion of the body union)
    std::vector<std::uint8_t> buf(req.header.length);
    std::memcpy(buf.data(), &req.header, sizeof(req.header));
    if (body_size > 0) {
        std::memcpy(buf.data() + sizeof(req.header), &req.body, body_size);
    }
    const ssize_t sent = send(fd, buf.data(), buf.size(), 0);
    return sent == static_cast<ssize_t>(buf.size());
}

int connect_with_retry(std::uint16_t port, int max_tries = 20) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    for (int i = 0; i < max_tries; ++i) {
        if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
            return fd;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    close(fd);
    return -1;
}

// =============================================================================
// Integration test 1: Full login → new order → delete sequence over TCP
// =============================================================================

bool test_login_new_order_delete_e2e() {
    std::cout << "\n[TEST] login_new_order_delete_e2e\n";

    constexpr std::uint16_t kOePort = 24200;
    NdfexGateway gateway;
    if (!gateway.start(kOePort, "127.0.0.1", 24201, "127.0.0.1", 24202)) {
        std::cerr << "  FAIL  Failed to start NDFEX gateway\n";
        return false;
    }

    const int fd = connect_with_retry(kOePort);
    if (!expect(fd >= 0, "client connects to gateway OE port")) {
        gateway.shutdown();
        return false;
    }

    bool ok = true;
    ghostbook::ndfex::seq_num_t req_seq = 1;
    ghostbook::ndfex::oe::session_id_t ndfex_session_id = 0;
    constexpr ghostbook::ndfex::oe::client_id_t kClientId = 7;

    // --- Login ---
    {
        std::cout << "  Sending LoginRequest...\n";
        Request req{};
        req.header.msg_type  = MSG_TYPE::LOGIN;
        req.header.version   = OE_PROTOCOL_VERSION;
        req.header.seq_num   = req_seq++;
        req.header.client_id = kClientId;
        std::memset(&req.body.login_request, 0, sizeof(req.body.login_request));

        ok &= expect(send_oe_request(fd, req, sizeof(ghostbook::ndfex::oe::LoginRequest)),
                     "LoginRequest sent successfully");

        Response resp{};
        ok &= expect(read_oe_response(fd, &resp), "LoginResponse received");
        ok &= expect_eq(static_cast<int>(resp.header.msg_type),
                        static_cast<int>(MSG_TYPE::LOGIN_RESPONSE),
                        "login response msg_type");
        ok &= expect_eq(static_cast<int>(resp.body.login_response.status),
                        static_cast<int>(ghostbook::ndfex::oe::LOGIN_STATUS::SUCCESS),
                        "login status SUCCESS");
        ok &= expect(resp.body.login_response.session_id != 0, "login assigns non-zero session_id");
        ndfex_session_id = resp.body.login_response.session_id;
        ok &= expect_eq(resp.header.client_id, kClientId, "response echoes client_id");
        ok &= expect_eq(resp.header.seq_num, static_cast<ghostbook::ndfex::seq_num_t>(1u),
                        "response echoes request seq_num");
    }

    // --- New order ---
    {
        std::cout << "  Sending NewOrderRequest (buy 5 @ 100)...\n";
        Request req{};
        req.header.msg_type   = MSG_TYPE::NEW_ORDER;
        req.header.version    = OE_PROTOCOL_VERSION;
        req.header.seq_num    = req_seq++;
        req.header.client_id  = kClientId;
        req.header.session_id = ndfex_session_id;
        req.body.new_order_request.order_id = 101;
        req.body.new_order_request.symbol   = 1;
        req.body.new_order_request.side     = ghostbook::ndfex::SIDE::BUY;
        req.body.new_order_request.quantity = 5;
        req.body.new_order_request.price    = 100;
        req.body.new_order_request.flags    = ghostbook::ndfex::oe::ORDER_FLAGS::NONE;

        ok &= expect(send_oe_request(fd, req, sizeof(ghostbook::ndfex::oe::NewOrderRequest)),
                     "NewOrderRequest sent");

        Response resp{};
        ok &= expect(read_oe_response(fd, &resp), "NewOrder ACK received");
        ok &= expect_eq(static_cast<int>(resp.header.msg_type),
                        static_cast<int>(MSG_TYPE::ACK),
                        "new order response is ACK");
        ok &= expect_eq(resp.body.order_ack_response.order_id,
                        static_cast<ghostbook::ndfex::order_id_t>(101),
                        "ACK echoes order_id");
        ok &= expect_eq(resp.body.order_ack_response.quantity,
                        static_cast<ghostbook::ndfex::quantity_t>(5),
                        "ACK quantity matches submitted qty");
        ok &= expect_eq(resp.body.order_ack_response.price,
                        static_cast<ghostbook::ndfex::price_t>(100),
                        "ACK price matches submitted price");
        ok &= expect(resp.body.order_ack_response.exch_order_id != 101,
                     "exchange order_id differs from client order_id");
    }

    // --- Delete order ---
    {
        std::cout << "  Sending DeleteOrderRequest (cancel order 101)...\n";
        Request req{};
        req.header.msg_type   = MSG_TYPE::DELETE_ORDER;
        req.header.version    = OE_PROTOCOL_VERSION;
        req.header.seq_num    = req_seq++;
        req.header.client_id  = kClientId;
        req.header.session_id = ndfex_session_id;
        req.body.delete_order_request.order_id = 101;

        ok &= expect(send_oe_request(fd, req, sizeof(ghostbook::ndfex::oe::DeleteOrderRequest)),
                     "DeleteOrderRequest sent");

        Response resp{};
        ok &= expect(read_oe_response(fd, &resp), "DeleteOrder CLOSE received");
        ok &= expect_eq(static_cast<int>(resp.header.msg_type),
                        static_cast<int>(MSG_TYPE::CLOSE),
                        "delete response is CLOSE");
        ok &= expect_eq(resp.body.order_closed_response.order_id,
                        static_cast<ghostbook::ndfex::order_id_t>(101),
                        "CLOSE echoes original order_id");
    }

    close(fd);
    gateway.shutdown();
    return ok;
}

// =============================================================================
// Integration test 2: Two clients, crossing orders — both receive fills
// =============================================================================

bool test_cross_order_fill_e2e() {
    std::cout << "\n[TEST] cross_order_fill_e2e\n";

    constexpr std::uint16_t kOePort = 24210;
    NdfexGateway gateway;
    if (!gateway.start(kOePort, "127.0.0.1", 24211, "127.0.0.1", 24212)) {
        std::cerr << "  FAIL  Failed to start NDFEX gateway\n";
        return false;
    }

    const int seller_fd = connect_with_retry(kOePort);
    if (!expect(seller_fd >= 0, "seller connects to gateway")) {
        gateway.shutdown();
        return false;
    }

    const int buyer_fd = connect_with_retry(kOePort);
    if (!expect(buyer_fd >= 0, "buyer connects to gateway")) {
        close(seller_fd);
        gateway.shutdown();
        return false;
    }

    bool ok = true;

    // --- Seller login ---
    {
        Request req{};
        req.header.msg_type  = MSG_TYPE::LOGIN;
        req.header.version   = OE_PROTOCOL_VERSION;
        req.header.seq_num   = 1;
        req.header.client_id = 10;
        std::memset(&req.body.login_request, 0, sizeof(req.body.login_request));
        send_oe_request(seller_fd, req, sizeof(ghostbook::ndfex::oe::LoginRequest));

        Response resp{};
        ok &= expect(read_oe_response(seller_fd, &resp), "seller receives LoginResponse");
        ok &= expect_eq(static_cast<int>(resp.body.login_response.status),
                        static_cast<int>(ghostbook::ndfex::oe::LOGIN_STATUS::SUCCESS),
                        "seller login SUCCESS");
    }

    // --- Buyer login ---
    ghostbook::ndfex::oe::session_id_t buyer_session = 0;
    {
        Request req{};
        req.header.msg_type  = MSG_TYPE::LOGIN;
        req.header.version   = OE_PROTOCOL_VERSION;
        req.header.seq_num   = 1;
        req.header.client_id = 20;
        std::memset(&req.body.login_request, 0, sizeof(req.body.login_request));
        send_oe_request(buyer_fd, req, sizeof(ghostbook::ndfex::oe::LoginRequest));

        Response resp{};
        ok &= expect(read_oe_response(buyer_fd, &resp), "buyer receives LoginResponse");
        ok &= expect_eq(static_cast<int>(resp.body.login_response.status),
                        static_cast<int>(ghostbook::ndfex::oe::LOGIN_STATUS::SUCCESS),
                        "buyer login SUCCESS");
        buyer_session = resp.body.login_response.session_id;
    }

    // --- Seller places resting sell ---
    {
        std::cout << "  Seller placing resting SELL 4 @ 100...\n";
        Request req{};
        req.header.msg_type   = MSG_TYPE::NEW_ORDER;
        req.header.version    = OE_PROTOCOL_VERSION;
        req.header.seq_num    = 2;
        req.header.client_id  = 10;
        req.body.new_order_request.order_id = 201;
        req.body.new_order_request.symbol   = 1;
        req.body.new_order_request.side     = ghostbook::ndfex::SIDE::SELL;
        req.body.new_order_request.quantity = 4;
        req.body.new_order_request.price    = 100;
        req.body.new_order_request.flags    = ghostbook::ndfex::oe::ORDER_FLAGS::NONE;
        send_oe_request(seller_fd, req, sizeof(ghostbook::ndfex::oe::NewOrderRequest));

        Response resp{};
        ok &= expect(read_oe_response(seller_fd, &resp), "seller receives ACK for resting sell");
        ok &= expect_eq(static_cast<int>(resp.header.msg_type), static_cast<int>(MSG_TYPE::ACK),
                        "seller resting order is ACK'd");
    }

    // --- Buyer crosses with aggressive buy ---
    {
        std::cout << "  Buyer sending aggressive BUY 4 @ 105...\n";
        Request req{};
        req.header.msg_type   = MSG_TYPE::NEW_ORDER;
        req.header.version    = OE_PROTOCOL_VERSION;
        req.header.seq_num    = 2;
        req.header.client_id  = 20;
        req.header.session_id = buyer_session;
        req.body.new_order_request.order_id = 202;
        req.body.new_order_request.symbol   = 1;
        req.body.new_order_request.side     = ghostbook::ndfex::SIDE::BUY;
        req.body.new_order_request.quantity = 4;
        req.body.new_order_request.price    = 105;
        req.body.new_order_request.flags    = ghostbook::ndfex::oe::ORDER_FLAGS::NONE;
        send_oe_request(buyer_fd, req, sizeof(ghostbook::ndfex::oe::NewOrderRequest));

        // Engine emits ACK before FILL for all accepted orders; consume ACK first.
        Response buyer_ack{};
        ok &= expect(read_oe_response(buyer_fd, &buyer_ack, 3000), "buyer receives ACK");
        ok &= expect_eq(static_cast<int>(buyer_ack.header.msg_type),
                        static_cast<int>(MSG_TYPE::ACK),
                        "buyer first response is ACK");

        Response buyer_resp{};
        ok &= expect(read_oe_response(buyer_fd, &buyer_resp, 3000), "buyer receives FILL");
        ok &= expect_eq(static_cast<int>(buyer_resp.header.msg_type),
                        static_cast<int>(MSG_TYPE::FILL),
                        "buyer receives FILL");
        ok &= expect_eq(buyer_resp.body.order_fill_response.quantity,
                        static_cast<ghostbook::ndfex::quantity_t>(4),
                        "buyer fill quantity is 4");
        ok &= expect_eq(static_cast<int>(buyer_resp.body.order_fill_response.flags),
                        static_cast<int>(ghostbook::ndfex::oe::FILL_FLAGS::CLOSED),
                        "buyer fill is CLOSED");

        // Seller should also receive a fill (async delivery to seller socket)
        Response seller_resp{};
        ok &= expect(read_oe_response(seller_fd, &seller_resp, 3000), "seller receives fill notification");
        ok &= expect_eq(static_cast<int>(seller_resp.header.msg_type),
                        static_cast<int>(MSG_TYPE::FILL),
                        "seller receives FILL");
        ok &= expect_eq(seller_resp.body.order_fill_response.order_id,
                        static_cast<ghostbook::ndfex::order_id_t>(201),
                        "seller fill references seller order_id");
    }

    close(buyer_fd);
    close(seller_fd);
    gateway.shutdown();
    return ok;
}

// =============================================================================
// Integration test 3: Zero-quantity order is rejected over TCP
// =============================================================================

bool test_invalid_order_rejected_e2e() {
    std::cout << "\n[TEST] invalid_order_rejected_e2e\n";

    constexpr std::uint16_t kOePort = 24220;
    NdfexGateway gateway;
    if (!gateway.start(kOePort, "127.0.0.1", 24221, "127.0.0.1", 24222)) {
        std::cerr << "  FAIL  Failed to start NDFEX gateway\n";
        return false;
    }

    const int fd = connect_with_retry(kOePort);
    if (!expect(fd >= 0, "client connects to gateway")) {
        gateway.shutdown();
        return false;
    }

    bool ok = true;
    ghostbook::ndfex::oe::session_id_t sess = 0;

    // Login
    {
        Request req{};
        req.header.msg_type  = MSG_TYPE::LOGIN;
        req.header.version   = OE_PROTOCOL_VERSION;
        req.header.seq_num   = 1;
        req.header.client_id = 5;
        std::memset(&req.body.login_request, 0, sizeof(req.body.login_request));
        send_oe_request(fd, req, sizeof(ghostbook::ndfex::oe::LoginRequest));

        Response resp{};
        ok &= expect(read_oe_response(fd, &resp), "LoginResponse received");
        sess = resp.body.login_response.session_id;
    }

    // Zero-quantity order
    {
        std::cout << "  Sending zero-quantity NewOrderRequest...\n";
        Request req{};
        req.header.msg_type   = MSG_TYPE::NEW_ORDER;
        req.header.version    = OE_PROTOCOL_VERSION;
        req.header.seq_num    = 2;
        req.header.client_id  = 5;
        req.header.session_id = sess;
        req.body.new_order_request.order_id = 301;
        req.body.new_order_request.symbol   = 1;
        req.body.new_order_request.side     = ghostbook::ndfex::SIDE::BUY;
        req.body.new_order_request.quantity = 0;  // invalid
        req.body.new_order_request.price    = 100;
        req.body.new_order_request.flags    = ghostbook::ndfex::oe::ORDER_FLAGS::NONE;
        send_oe_request(fd, req, sizeof(ghostbook::ndfex::oe::NewOrderRequest));

        Response resp{};
        ok &= expect(read_oe_response(fd, &resp), "REJECT response received for zero-qty order");
        ok &= expect_eq(static_cast<int>(resp.header.msg_type),
                        static_cast<int>(MSG_TYPE::REJECT),
                        "zero-qty order response is REJECT");
        ok &= expect_eq(resp.body.order_reject_response.order_id,
                        static_cast<ghostbook::ndfex::order_id_t>(301),
                        "REJECT references submitted order_id");
        ok &= expect_eq(static_cast<int>(resp.body.order_reject_response.reject_reason),
                        static_cast<int>(ghostbook::ndfex::oe::REJECT_REASON::INVALID_QUANTITY),
                        "REJECT reason is INVALID_QUANTITY");
    }

    close(fd);
    gateway.shutdown();
    return ok;
}

}  // namespace

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== NDFEX Gateway TCP Integration Tests ===\n";

    bool ok = true;
    ok &= test_login_new_order_delete_e2e();
    ok &= test_cross_order_fill_e2e();
    ok &= test_invalid_order_rejected_e2e();

    std::cout << "\n";
    if (!ok) {
        std::cerr << "One or more NDFEX TCP integration tests FAILED\n";
        return 1;
    }
    std::cout << "All NDFEX gateway TCP integration tests passed\n";
    return 0;
}
