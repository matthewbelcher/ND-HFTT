#include "msg/message.h"
#include "tcp/connection.h"
#include "tcp/flow_control.h"

#include <arpa/inet.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// Mini test runner

static int g_pass = 0;
static int g_fail = 0;

static void pass(const char* name) {
    std::cout << "  PASS  " << name << '\n';
    ++g_pass;
}
static void fail(const char* name, const char* reason) {
    std::cout << "  FAIL  " << name << " -- " << reason << '\n';
    ++g_fail;
}

// FlowController tests

static void test_flow_initial_window() {
    const char* name = "FlowController: initial window equals RECV_BUF_SIZE";
    FlowController fc;
    if (fc.recv_window() != static_cast<uint16_t>(FlowController::RECV_BUF_SIZE)) {
        fail(name, "initial window != RECV_BUF_SIZE"); return;
    }
    pass(name);
}

static void test_flow_window_shrinks_on_receive() {
    const char* name = "FlowController: window shrinks by bytes received";
    FlowController fc;
    fc.on_receive(1000);
    const uint16_t expected = static_cast<uint16_t>(FlowController::RECV_BUF_SIZE - 1000);
    if (fc.recv_window() != expected) {
        fail(name, "window did not shrink correctly"); return;
    }
    pass(name);
}

static void test_flow_window_restores_on_consume() {
    const char* name = "FlowController: window restored after consume";
    FlowController fc;
    fc.on_receive(4000);
    fc.on_consume(4000);
    if (fc.recv_window() != static_cast<uint16_t>(FlowController::RECV_BUF_SIZE)) {
        fail(name, "window not fully restored after consume"); return;
    }
    pass(name);
}

static void test_flow_window_saturates_at_buf_size() {
    const char* name = "FlowController: receive beyond capacity clamps at RECV_BUF_SIZE";
    FlowController fc;
    fc.on_receive(FlowController::RECV_BUF_SIZE + 100);
    if (fc.recv_window() != 0) {
        fail(name, "window should be 0 when buffer is full"); return;
    }
    pass(name);
}

static void test_flow_consume_below_zero_clamps() {
    const char* name = "FlowController: over-consume clamps at 0 bytes in buffer";
    FlowController fc;
    fc.on_consume(1000); // consume with empty buffer
    if (fc.recv_window() != static_cast<uint16_t>(FlowController::RECV_BUF_SIZE)) {
        fail(name, "window changed on over-consume"); return;
    }
    pass(name);
}

// Encode/decode round-trip tests

static void test_encode_decode_heartbeat() {
    const char* name = "Message: HEARTBEAT encode/decode round-trip";

    Message orig{};
    orig.type        = MsgType::HEARTBEAT;
    orig.payload_len = 0;

    uint8_t buf[MSG_MAX_LEN];
    const size_t n = msg_encode(orig, buf, sizeof(buf));
    if (n != MSG_HDR_LEN) { fail(name, "encode returned wrong byte count"); return; }

    Message decoded{};
    if (!msg_decode(buf, n, decoded)) { fail(name, "decode failed"); return; }
    if (decoded.type != MsgType::HEARTBEAT) { fail(name, "type mismatch"); return; }
    if (decoded.payload_len != 0)           { fail(name, "payload_len should be 0"); return; }
    pass(name);
}

static void test_encode_decode_with_payload() {
    const char* name = "Message: arbitrary payload encode/decode round-trip";

    Message orig{};
    orig.type        = static_cast<MsgType>(0x02);
    orig.payload_len = 8;
    for (uint8_t i = 0; i < 8; ++i) orig.payload[i] = i * 11u;

    uint8_t buf[MSG_MAX_LEN];
    const size_t n = msg_encode(orig, buf, sizeof(buf));
    if (n != MSG_HDR_LEN + 8) { fail(name, "encode returned wrong byte count"); return; }

    Message decoded{};
    if (!msg_decode(buf, n, decoded))               { fail(name, "decode failed"); return; }
    if (decoded.type != static_cast<MsgType>(0x02)) { fail(name, "type mismatch"); return; }
    if (decoded.payload_len != 8)                   { fail(name, "payload_len mismatch"); return; }
    if (std::memcmp(decoded.payload, orig.payload, 8) != 0) {
        fail(name, "payload bytes mismatch"); return;
    }
    pass(name);
}

static void test_encode_buf_too_small() {
    const char* name = "Message: encode returns 0 when buffer too small";
    Message msg{};
    msg.type        = MsgType::HEARTBEAT;
    msg.payload_len = 0;

    uint8_t buf[2]; // smaller than MSG_HDR_LEN (3)
    const size_t n = msg_encode(msg, buf, sizeof(buf));
    if (n != 0) { fail(name, "encode should return 0"); return; }
    pass(name);
}

static void test_decode_buf_too_small() {
    const char* name = "Message: decode returns false when buffer too small";
    uint8_t buf[2] = {0x01, 0x00};
    Message msg{};
    if (msg_decode(buf, 2, msg)) { fail(name, "decode should return false"); return; }
    pass(name);
}

// MessageFramer tests

static Message make_heartbeat() {
    Message m{};
    m.type        = MsgType::HEARTBEAT;
    m.payload_len = 0;
    return m;
}

static void test_framer_single_complete_message() {
    const char* name = "MessageFramer: single complete message decoded";
    MessageFramer framer;
    std::vector<Message> received;

    Message hb = make_heartbeat();
    uint8_t buf[MSG_MAX_LEN];
    const size_t n = msg_encode(hb, buf, sizeof(buf));

    framer.feed(buf, n, [&](const Message& m) { received.push_back(m); });

    if (received.size() != 1) { fail(name, "expected 1 message"); return; }
    if (received[0].type != MsgType::HEARTBEAT) { fail(name, "type mismatch"); return; }
    pass(name);
}

static void test_framer_message_split_at_first_byte() {
    const char* name = "MessageFramer: message split after first byte";
    MessageFramer framer;
    std::vector<Message> received;

    Message hb = make_heartbeat();
    uint8_t buf[MSG_MAX_LEN];
    const size_t n = msg_encode(hb, buf, sizeof(buf));

    framer.feed(buf, 1, [&](const Message& m) { received.push_back(m); });
    if (!received.empty()) { fail(name, "message emitted prematurely"); return; }
    framer.feed(buf + 1, n - 1, [&](const Message& m) { received.push_back(m); });

    if (received.size() != 1) { fail(name, "expected 1 message after second feed"); return; }
    if (received[0].type != MsgType::HEARTBEAT) { fail(name, "type mismatch"); return; }
    pass(name);
}

static void test_framer_message_split_at_header_payload_boundary() {
    const char* name = "MessageFramer: message split between header and payload";
    MessageFramer framer;
    std::vector<Message> received;

    // Use a message with a known non-zero payload (no application-specific meaning).
    Message msg{};
    msg.type        = static_cast<MsgType>(0x02);
    msg.payload_len = 16;
    for (uint8_t i = 0; i < 16; ++i) msg.payload[i] = i;

    uint8_t buf[MSG_MAX_LEN];
    const size_t n = msg_encode(msg, buf, sizeof(buf));

    // Feed only the 3-byte header first.
    framer.feed(buf, MSG_HDR_LEN, [&](const Message& m) { received.push_back(m); });
    if (!received.empty()) { fail(name, "message emitted with only header"); return; }

    // Feed the payload.
    framer.feed(buf + MSG_HDR_LEN, n - MSG_HDR_LEN,
                [&](const Message& m) { received.push_back(m); });

    if (received.size() != 1) { fail(name, "expected 1 message after payload"); return; }
    if (received[0].type != static_cast<MsgType>(0x02)) { fail(name, "type mismatch"); return; }
    if (received[0].payload[0] != 0 || received[0].payload[15] != 15) {
        fail(name, "payload bytes mismatch"); return;
    }
    pass(name);
}

static void test_framer_two_messages_in_one_feed() {
    const char* name = "MessageFramer: two messages delivered in one feed call";
    MessageFramer framer;
    std::vector<Message> received;

    uint8_t buf[2 * MSG_MAX_LEN];
    Message hb = make_heartbeat();
    const size_t n1 = msg_encode(hb, buf, sizeof(buf));
    const size_t n2 = msg_encode(hb, buf + n1, sizeof(buf) - n1);

    framer.feed(buf, n1 + n2, [&](const Message& m) { received.push_back(m); });

    if (received.size() != 2) {
        fail(name, ("expected 2 messages, got " + std::to_string(received.size())).c_str());
        return;
    }
    if (received[0].type != MsgType::HEARTBEAT || received[1].type != MsgType::HEARTBEAT) {
        fail(name, "type mismatch"); return;
    }
    pass(name);
}

static void test_framer_byte_by_byte() {
    const char* name = "MessageFramer: message assembled byte by byte";
    MessageFramer framer;
    std::vector<Message> received;

    Message hb = make_heartbeat();
    uint8_t buf[MSG_MAX_LEN];
    const size_t n = msg_encode(hb, buf, sizeof(buf));

    for (size_t i = 0; i < n; ++i)
        framer.feed(buf + i, 1, [&](const Message& m) { received.push_back(m); });

    if (received.size() != 1) { fail(name, "expected 1 message"); return; }
    pass(name);
}

static void test_framer_reset_clears_state() {
    const char* name = "MessageFramer: reset discards partial state";
    MessageFramer framer;
    std::vector<Message> received;

    Message hb = make_heartbeat();
    uint8_t buf[MSG_MAX_LEN];
    msg_encode(hb, buf, sizeof(buf));

    // Feed 1 byte, then reset, then feed complete message.
    framer.feed(buf, 1, [&](const Message& m) { received.push_back(m); });
    framer.reset();
    framer.feed(buf, MSG_HDR_LEN, [&](const Message& m) { received.push_back(m); });

    if (received.size() != 1) {
        fail(name, "expected exactly 1 message after reset and re-feed"); return;
    }
    pass(name);
}

static void test_framer_corrupt_length_discarded_then_recovers() {
    const char* name = "MessageFramer: corrupt length field triggers reset; valid message after recovers";
    MessageFramer framer;
    std::vector<Message> received;

    // Build a 4-byte sequence whose length field (bytes 1-2) encodes 0xFF00,
    // which exceeds MSG_MAX_PAYLOAD (253) and should trigger a reset.
    uint8_t corrupt[4];
    corrupt[0] = static_cast<uint8_t>(MsgType::HEARTBEAT);
    corrupt[1] = 0xFF; // plen high byte
    corrupt[2] = 0x00; // plen = 0xFF00 = 65280 > MSG_MAX_PAYLOAD
    corrupt[3] = 0x00; // extra padding byte

    framer.feed(corrupt, sizeof(corrupt), [&](const Message& m) { received.push_back(m); });

    if (!received.empty()) {
        fail(name, "framer emitted a message from corrupt stream"); return;
    }

    // Framer should have reset; feed a valid heartbeat and expect it to decode.
    Message hb = make_heartbeat();
    uint8_t good[MSG_MAX_LEN];
    const size_t n = msg_encode(hb, good, sizeof(good));

    framer.feed(good, n, [&](const Message& m) { received.push_back(m); });

    if (received.size() != 1) {
        fail(name, "framer did not recover after corrupt stream"); return;
    }
    if (received[0].type != MsgType::HEARTBEAT) {
        fail(name, "recovered message has wrong type"); return;
    }
    pass(name);
}

// Integration: messages over in-process TCP

struct ConnPair {
    std::vector<uint8_t> b_recv;
    std::vector<uint8_t> a_recv;

    TCPConnection b;
    TCPConnection a;

    static constexpr uint32_t IP_A   = 0x0a000001u;
    static constexpr uint32_t IP_B   = 0x0a000002u;
    static constexpr uint16_t PORT_A = 54400;
    static constexpr uint16_t PORT_B = 9090;
    static constexpr uint32_t ISN_A  = 5000;
    static constexpr uint32_t ISN_B  = 6000;

    ConnPair()
        : b(htonl(IP_B), PORT_B, htonl(IP_A), PORT_A,
            [this](const uint8_t* buf, size_t len) { a.receive_packet(buf, len); },
            [this](const uint8_t* d, size_t n) {
                b_recv.insert(b_recv.end(), d, d + n);
            },
            ISN_B)
        , a(htonl(IP_A), PORT_A, htonl(IP_B), PORT_B,
            [this](const uint8_t* buf, size_t len) { b.receive_packet(buf, len); },
            [this](const uint8_t* d, size_t n) {
                a_recv.insert(a_recv.end(), d, d + n);
            },
            ISN_A)
    {}

    void handshake() {
        b.listen();
        a.connect();
    }
};

static void test_integration_message_over_tcp() {
    const char* name = "Integration: message framed and delivered over in-process TCP";
    ConnPair p;
    p.handshake();

    // Send a message with a small application-defined payload.
    Message msg{};
    msg.type        = static_cast<MsgType>(0x02);
    msg.payload_len = 8;
    for (uint8_t i = 0; i < 8; ++i) msg.payload[i] = i;

    uint8_t wire[MSG_MAX_LEN];
    const size_t wire_len = msg_encode(msg, wire, sizeof(wire));
    if (wire_len == 0) { fail(name, "encode failed"); return; }

    const size_t sent = p.a.send(wire, wire_len);
    if (sent != wire_len) { fail(name, "send() did not accept all bytes"); return; }

    if (p.b_recv.size() != wire_len) {
        fail(name, "B received wrong byte count"); return;
    }

    MessageFramer framer;
    std::vector<Message> decoded;
    framer.feed(p.b_recv.data(), p.b_recv.size(),
                [&](const Message& m) { decoded.push_back(m); });

    if (decoded.size() != 1) { fail(name, "framer did not produce 1 message"); return; }
    if (decoded[0].type != static_cast<MsgType>(0x02)) { fail(name, "type mismatch"); return; }
    if (std::memcmp(decoded[0].payload, msg.payload, 8) != 0) {
        fail(name, "payload mismatch"); return;
    }
    pass(name);
}

static void test_integration_heartbeat_sequence() {
    const char* name = "Integration: multiple heartbeats framed over in-process TCP";
    ConnPair p;
    p.handshake();

    Message hb = make_heartbeat();
    uint8_t wire[MSG_MAX_LEN];
    const size_t wire_len = msg_encode(hb, wire, sizeof(wire));

    for (int i = 0; i < 3; ++i) {
        const size_t sent = p.a.send(wire, wire_len);
        if (sent != wire_len) { fail(name, "send() short on heartbeat"); return; }
    }

    if (p.b_recv.size() != 3 * wire_len) {
        fail(name, "B received wrong total byte count"); return;
    }

    MessageFramer framer;
    std::vector<Message> decoded;
    framer.feed(p.b_recv.data(), p.b_recv.size(),
                [&](const Message& m) { decoded.push_back(m); });

    if (decoded.size() != 3) {
        fail(name, ("expected 3 messages, got " + std::to_string(decoded.size())).c_str());
        return;
    }
    for (const auto& m : decoded) {
        if (m.type != MsgType::HEARTBEAT) { fail(name, "type mismatch"); return; }
    }
    pass(name);
}

static void test_integration_window_advertised_nonzero() {
    const char* name = "Integration: receive window advertised in ACK is non-zero after data transfer";

    std::vector<std::vector<uint8_t>> to_b, to_a;

    TCPConnection a(htonl(0x0a000001u), 54401, htonl(0x0a000002u), 9091,
                    [&to_b](const uint8_t* buf, size_t len) {
                        to_b.push_back(std::vector<uint8_t>(buf, buf + len));
                    },
                    nullptr, 5001);
    TCPConnection b(htonl(0x0a000002u), 9091, htonl(0x0a000001u), 54401,
                    [&to_a](const uint8_t* buf, size_t len) {
                        to_a.push_back(std::vector<uint8_t>(buf, buf + len));
                    },
                    nullptr, 6001);

    b.listen();
    a.connect();
    b.receive_packet(to_b.back().data(), to_b.back().size()); to_b.clear();
    a.receive_packet(to_a.back().data(), to_a.back().size()); to_a.clear();
    b.receive_packet(to_b.back().data(), to_b.back().size()); to_b.clear();

    const uint8_t payload[] = {'t','e','s','t'};
    a.send(payload, 4);
    b.receive_packet(to_b.back().data(), to_b.back().size()); to_b.clear();

    if (to_a.empty()) { fail(name, "B did not send an ACK"); return; }
    const auto& ack_frame = to_a.back();

    const size_t ihl = (ack_frame[0] & 0x0Fu) * 4u;
    const uint16_t window =
        (static_cast<uint16_t>(ack_frame[ihl + 14]) << 8) | ack_frame[ihl + 15];

    if (window == 0) { fail(name, "B advertised zero receive window after delivering data"); return; }
    pass(name);
}

// main

int main() {
    std::cout << "=== Phase 4: Flow Control & Message Framing Tests ===\n\n";

    std::cout << "  -- FlowController --\n";
    test_flow_initial_window();
    test_flow_window_shrinks_on_receive();
    test_flow_window_restores_on_consume();
    test_flow_window_saturates_at_buf_size();
    test_flow_consume_below_zero_clamps();

    std::cout << "\n  -- Encode/decode round-trip --\n";
    test_encode_decode_heartbeat();
    test_encode_decode_with_payload();
    test_encode_buf_too_small();
    test_decode_buf_too_small();

    std::cout << "\n  -- MessageFramer --\n";
    test_framer_single_complete_message();
    test_framer_message_split_at_first_byte();
    test_framer_message_split_at_header_payload_boundary();
    test_framer_two_messages_in_one_feed();
    test_framer_byte_by_byte();
    test_framer_reset_clears_state();
    test_framer_corrupt_length_discarded_then_recovers();

    std::cout << "\n  -- Integration (messages over in-process TCP) --\n";
    test_integration_message_over_tcp();
    test_integration_heartbeat_sequence();
    test_integration_window_advertised_nonzero();

    std::cout << '\n'
              << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
