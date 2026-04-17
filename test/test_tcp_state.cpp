#include "tcp/connection.h"
#include "tcp/tcp_state.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <arpa/inet.h>   // htonl

// ── Mini test runner ──────────────────────────────────────────────────────────
static int g_pass = 0;
static int g_fail = 0;

static void pass(const char* name) {
    std::cout << "  PASS  " << name << '\n';
    ++g_pass;
}
static void fail(const char* name, const char* reason) {
    std::cout << "  FAIL  " << name << " — " << reason << '\n';
    ++g_fail;
}

// Assert helpers
#define ASSERT_EQ_STATE(conn, expected, test_name)                         \
    do {                                                                    \
        if ((conn).state() != (expected)) {                                 \
            std::string msg = "expected " + std::string(state_name(expected)) \
                            + ", got " + std::string(state_name((conn).state())); \
            fail(test_name, msg.c_str());                                  \
            return;                                                         \
        }                                                                   \
    } while (false)

// ── In-process wiring helpers ─────────────────────────────────────────────────

// Build a wired pair: A (client) connects to B (server).
// A.send_fn_ → B.receive_packet, B.send_fn_ → A.receive_packet.
// The full three-way handshake fires synchronously inside connect().
struct ConnPair {
    // Captured received data for each side.
    std::vector<uint8_t> a_recv;
    std::vector<uint8_t> b_recv;

    TCPConnection a;
    TCPConnection b;

    static constexpr uint32_t IP_A = 0x0a000001u; // 10.0.0.1 in host order
    static constexpr uint32_t IP_B = 0x0a000002u;
    static constexpr uint16_t PORT_A = 54321;
    static constexpr uint16_t PORT_B = 8080;
    static constexpr uint32_t ISN_A  = 1000;
    static constexpr uint32_t ISN_B  = 2000;

    ConnPair()
        // Construct B first (it only needs to exist when A's send fires).
        : b(htonl(IP_B), PORT_B, htonl(IP_A), PORT_A,
            // B's send_fn: route outbound frames to A.
            [this](const uint8_t* buf, size_t len) {
                a.receive_packet(buf, len);
            },
            // B's recv_fn: application data received BY B → b_recv.
            [this](const uint8_t* d, size_t n) {
                b_recv.insert(b_recv.end(), d, d + n);
            },
            ISN_B)

        , a(htonl(IP_A), PORT_A, htonl(IP_B), PORT_B,
            // A's send_fn: route outbound frames to B.
            [this](const uint8_t* buf, size_t len) {
                b.receive_packet(buf, len);
            },
            // A's recv_fn: application data received BY A → a_recv.
            [this](const uint8_t* d, size_t n) {
                a_recv.insert(a_recv.end(), d, d + n);
            },
            ISN_A)
    {}

    // Set up and complete the three-way handshake.
    void handshake() {
        b.listen();
        a.connect();   // SYN → B; B sends SYN-ACK → A; A sends ACK → B
    }
};

// ── State-transition unit tests ───────────────────────────────────────────────

static void test_initial_state_closed() {
    const char* name = "TCPConnection: initial state is CLOSED";
    ConnPair p;
    if (p.a.state() != TCPState::CLOSED || p.b.state() != TCPState::CLOSED)
        fail(name, "not CLOSED after construction");
    else
        pass(name);
}

static void test_listen_transitions_to_listen() {
    const char* name = "TCPConnection: listen() → LISTEN";
    ConnPair p;
    p.b.listen();
    ASSERT_EQ_STATE(p.b, TCPState::LISTEN, name);
    pass(name);
}

static void test_connect_without_listener_sets_syn_sent() {
    const char* name = "TCPConnection: connect() sets SYN_SENT on client";
    // Wire a to /dev/null so no SYN-ACK comes back.
    TCPConnection a(htonl(0x0a000001u), 12345, htonl(0x0a000002u), 80,
                    [](const uint8_t*, size_t) {}, nullptr, 1000);
    a.connect();
    ASSERT_EQ_STATE(a, TCPState::SYN_SENT, name);
    pass(name);
}

// ── Handshake tests ───────────────────────────────────────────────────────────

static void test_handshake_both_established() {
    const char* name = "Handshake: both sides reach ESTABLISHED";
    ConnPair p;
    p.handshake();
    ASSERT_EQ_STATE(p.a, TCPState::ESTABLISHED, name);
    ASSERT_EQ_STATE(p.b, TCPState::ESTABLISHED, name);
    pass(name);
}

static void test_handshake_sequence_numbers() {
    const char* name = "Handshake: sequence numbers are correct after handshake";
    ConnPair p;
    p.handshake();

    // After the handshake:
    //   A: snd_nxt = ISN_A + 1 (SYN consumed 1), rcv_nxt = ISN_B + 1
    //   B: snd_nxt = ISN_B + 1 (SYN-ACK consumed 1), rcv_nxt = ISN_A + 1
    constexpr uint32_t ISN_A = ConnPair::ISN_A;
    constexpr uint32_t ISN_B = ConnPair::ISN_B;

    if (p.a.snd_nxt() != ISN_A + 1) {
        fail(name, "A.snd_nxt wrong after handshake"); return;
    }
    if (p.a.rcv_nxt() != ISN_B + 1) {
        fail(name, "A.rcv_nxt wrong after handshake"); return;
    }
    if (p.b.snd_nxt() != ISN_B + 1) {
        fail(name, "B.snd_nxt wrong after handshake"); return;
    }
    if (p.b.rcv_nxt() != ISN_A + 1) {
        fail(name, "B.rcv_nxt wrong after handshake"); return;
    }
    pass(name);
}

// ── Data transfer tests ───────────────────────────────────────────────────────

static void test_data_a_to_b() {
    const char* name = "Data transfer: A → B, B receives correctly";
    ConnPair p;
    p.handshake();

    const std::string msg = "hello";
    p.a.send(reinterpret_cast<const uint8_t*>(msg.data()), msg.size());

    if (p.b_recv.size() != msg.size()) {
        fail(name, "B received wrong number of bytes"); return;
    }
    if (std::memcmp(p.b_recv.data(), msg.data(), msg.size()) != 0) {
        fail(name, "B received wrong payload"); return;
    }
    pass(name);
}

static void test_data_b_to_a() {
    const char* name = "Data transfer: B → A, A receives correctly";
    ConnPair p;
    p.handshake();

    const std::string msg = "world";
    p.b.send(reinterpret_cast<const uint8_t*>(msg.data()), msg.size());

    if (p.a_recv.size() != msg.size()) {
        fail(name, "A received wrong number of bytes"); return;
    }
    if (std::memcmp(p.a_recv.data(), msg.data(), msg.size()) != 0) {
        fail(name, "A received wrong payload"); return;
    }
    pass(name);
}

static void test_data_bidirectional() {
    const char* name = "Data transfer: bidirectional exchange";
    ConnPair p;
    p.handshake();

    const std::string ping = "ping";
    const std::string pong = "pong";
    p.a.send(reinterpret_cast<const uint8_t*>(ping.data()), ping.size());
    p.b.send(reinterpret_cast<const uint8_t*>(pong.data()), pong.size());

    const bool a_ok = (p.a_recv.size() == pong.size() &&
                       std::memcmp(p.a_recv.data(), pong.data(), pong.size()) == 0);
    const bool b_ok = (p.b_recv.size() == ping.size() &&
                       std::memcmp(p.b_recv.data(), ping.data(), ping.size()) == 0);

    if (!a_ok) { fail(name, "A received wrong pong"); return; }
    if (!b_ok) { fail(name, "B received wrong ping"); return; }
    pass(name);
}

static void test_sequence_advances_after_data() {
    const char* name = "Data transfer: snd_nxt / rcv_nxt advance correctly";
    ConnPair p;
    p.handshake();

    const std::string msg = "abcde";   // 5 bytes
    p.a.send(reinterpret_cast<const uint8_t*>(msg.data()), msg.size());

    // A.snd_nxt should advance by 5.
    if (p.a.snd_nxt() != ConnPair::ISN_A + 1 + 5) {
        fail(name, "A.snd_nxt did not advance by 5"); return;
    }
    // B.rcv_nxt should advance by 5.
    if (p.b.rcv_nxt() != ConnPair::ISN_A + 1 + 5) {
        fail(name, "B.rcv_nxt did not advance by 5"); return;
    }
    pass(name);
}

static void test_large_send_segmented() {
    const char* name = "Data transfer: payload > MSS is segmented correctly";
    ConnPair p;
    p.handshake();

    // Send slightly more than one MSS so the stack must split into two segments.
    const size_t data_len = TCP_MSS + 100;
    std::vector<uint8_t> data(data_len, 0xAB);
    const size_t sent = p.a.send(data.data(), data_len);

    if (sent != data_len) { fail(name, "send() returned wrong byte count"); return; }
    if (p.b_recv.size() != data_len) { fail(name, "B received wrong total bytes"); return; }
    for (size_t i = 0; i < data_len; ++i) {
        if (p.b_recv[i] != 0xAB) { fail(name, "payload byte mismatch"); return; }
    }
    pass(name);
}

// ── Connection teardown tests ─────────────────────────────────────────────────

// Full active-close sequence: A closes first.
//   A: ESTABLISHED → FIN_WAIT_1 → FIN_WAIT_2 → TIME_WAIT
//   B: ESTABLISHED → CLOSE_WAIT → LAST_ACK → CLOSED
static void test_teardown_a_closes_first() {
    const char* name = "Teardown (A active close): correct final states";
    ConnPair p;
    p.handshake();

    // Step 1: A sends FIN → B moves to CLOSE_WAIT.
    p.a.close();
    ASSERT_EQ_STATE(p.b, TCPState::CLOSE_WAIT, name);

    // After A's FIN is ACKed, A should advance to FIN_WAIT_2.
    // The ACK from B is sent synchronously inside handle_established when B
    // processes the FIN, which calls a.receive_packet → handle_fin_wait_1,
    // which advances A to FIN_WAIT_2 (retx queue drained).
    ASSERT_EQ_STATE(p.a, TCPState::FIN_WAIT_2, name);

    // Step 2: B's application calls close().
    // Because the pair is synchronously wired, the full sequence fires inside
    // p.b.close():  B→LAST_ACK, sends FIN → A→TIME_WAIT, sends ACK → B→CLOSED.
    // LAST_ACK is a transient state we cannot observe between two synchronous calls.
    p.b.close();
    ASSERT_EQ_STATE(p.a, TCPState::TIME_WAIT, name);
    ASSERT_EQ_STATE(p.b, TCPState::CLOSED,    name);

    pass(name);
}

// Symmetric teardown: both sides close simultaneously.
//   Both sides: ESTABLISHED → FIN_WAIT_1 → TIME_WAIT (simultaneous FIN)
static void test_teardown_simultaneous_close() {
    const char* name = "Teardown (simultaneous close): both sides reach TIME_WAIT";

    // Use a deferred-delivery pair so that FINs are injected manually,
    // simulating a network where both sides send FIN before either receives it.
    std::vector<std::vector<uint8_t>> a_outbox, b_outbox;
    std::vector<uint8_t> a_recv, b_recv;

    TCPConnection a(htonl(0x0a000001u), 54321, htonl(0x0a000002u), 8080,
                    [&a_outbox](const uint8_t* buf, size_t len) {
                        a_outbox.push_back(std::vector<uint8_t>(buf, buf + len));
                    },
                    nullptr, 1000);
    TCPConnection b(htonl(0x0a000002u), 8080, htonl(0x0a000001u), 54321,
                    [&b_outbox](const uint8_t* buf, size_t len) {
                        b_outbox.push_back(std::vector<uint8_t>(buf, buf + len));
                    },
                    nullptr, 2000);

    // Handshake (deliver each segment in order).
    b.listen();
    a.connect();  // → a_outbox: [SYN]

    b.receive_packet(a_outbox.back().data(), a_outbox.back().size());  // SYN → b sends SYN-ACK
    a_outbox.pop_back();

    a.receive_packet(b_outbox.back().data(), b_outbox.back().size());  // SYN-ACK → a sends ACK
    b_outbox.pop_back();

    b.receive_packet(a_outbox.back().data(), a_outbox.back().size());  // ACK → b ESTABLISHED
    a_outbox.pop_back();

    if (a.state() != TCPState::ESTABLISHED || b.state() != TCPState::ESTABLISHED) {
        fail(name, "handshake did not complete"); return;
    }

    // Both sides close without delivering each other's segments first.
    a.close();  // a_outbox: [FIN]
    b.close();  // b_outbox: [FIN]

    // Now deliver A's FIN to B.
    b.receive_packet(a_outbox.back().data(), a_outbox.back().size());
    a_outbox.pop_back();
    // B is now in TIME_WAIT (got FIN while in FIN_WAIT_1, per our simplified
    // simultaneous-close handling). B also sent an ACK → b_outbox.

    // Deliver B's FIN to A.
    // B's FIN is the first element in b_outbox (before the ACK).
    // Deliver all of B's queued packets to A.
    for (auto& pkt : b_outbox)
        a.receive_packet(pkt.data(), pkt.size());
    b_outbox.clear();

    if (a.state() != TCPState::TIME_WAIT) {
        fail(name, ("A expected TIME_WAIT, got " +
                    std::string(state_name(a.state()))).c_str());
        return;
    }
    if (b.state() != TCPState::TIME_WAIT) {
        fail(name, ("B expected TIME_WAIT, got " +
                    std::string(state_name(b.state()))).c_str());
        return;
    }

    pass(name);
}

// TIME_WAIT → CLOSED after 2×MSL via tick().
static void test_time_wait_expires() {
    const char* name = "Teardown: TIME_WAIT expires to CLOSED via tick()";
    ConnPair p;
    p.handshake();

    p.a.close();  // A: FIN_WAIT_1 (→ FIN_WAIT_2 after B's ACK)
    p.b.close();  // B: LAST_ACK; A: TIME_WAIT, B: CLOSED

    if (p.a.state() != TCPState::TIME_WAIT) {
        fail(name, "A not in TIME_WAIT before tick"); return;
    }

    // Synthetic tick just past TCP_TIME_WAIT_DURATION (4 s).
    auto future = std::chrono::steady_clock::now() + TCP_TIME_WAIT_DURATION
                  + std::chrono::milliseconds{1};
    p.a.tick(future);

    ASSERT_EQ_STATE(p.a, TCPState::CLOSED, name);
    pass(name);
}

// ── Retransmit timer integration ──────────────────────────────────────────────

// When a segment is not acknowledged, tick() must retransmit it.
static void test_retransmit_fires_on_timeout() {
    const char* name = "Retransmit: tick() retransmits on timeout, peer eventually ACKs";

    // Use separate outboxes so we can drop the first data packet (simulate loss).
    std::vector<std::vector<uint8_t>> to_b, to_a;
    std::vector<uint8_t> srv_recv_data;

    TCPConnection client(htonl(0x0a000001u), 54321, htonl(0x0a000002u), 8080,
                         [&to_b](const uint8_t* buf, size_t len) {
                             to_b.push_back(std::vector<uint8_t>(buf, buf + len));
                         },
                         nullptr, 1000);
    TCPConnection server(htonl(0x0a000002u), 8080, htonl(0x0a000001u), 54321,
                         [&to_a](const uint8_t* buf, size_t len) {
                             to_a.push_back(std::vector<uint8_t>(buf, buf + len));
                         },
                         [&srv_recv_data](const uint8_t* d, size_t n) {
                             srv_recv_data.insert(srv_recv_data.end(), d, d + n);
                         },
                         2000);

    auto flush_to_server = [&]() {
        for (auto& pkt : to_b) server.receive_packet(pkt.data(), pkt.size());
        to_b.clear();
    };
    auto flush_to_client = [&]() {
        for (auto& pkt : to_a) client.receive_packet(pkt.data(), pkt.size());
        to_a.clear();
    };

    // Handshake.
    server.listen();
    client.connect(); flush_to_server();  // SYN → server
    flush_to_client();                    // SYN-ACK → client; client sends ACK
    flush_to_server();                    // ACK → server

    if (client.state() != TCPState::ESTABLISHED ||
        server.state() != TCPState::ESTABLISHED) {
        fail(name, "handshake failed"); return;
    }

    // Send a data segment from client — drop it (don't flush to server).
    const std::string payload = "HFT";
    client.send(reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
    to_b.clear();   // "drop" the first data segment

    // Trigger a retransmit via tick() (simulate RTO_INITIAL elapsed).
    auto now = std::chrono::steady_clock::now() + RTO_INITIAL + std::chrono::milliseconds{1};
    client.tick(now);

    // Now deliver the retransmitted segment to the server.
    flush_to_server();   // server gets data, sends ACK
    flush_to_client();   // client gets ACK

    if (srv_recv_data.size() != payload.size() ||
        std::memcmp(srv_recv_data.data(), payload.data(), payload.size()) != 0) {
        fail(name, "server did not receive correct data after retransmit");
        return;
    }
    if (client.state() != TCPState::ESTABLISHED) {
        fail(name, "client not ESTABLISHED after successful retransmit");
        return;
    }
    pass(name);
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=== Phase 3: TCP State Machine Tests ===\n\n";

    test_initial_state_closed();
    test_listen_transitions_to_listen();
    test_connect_without_listener_sets_syn_sent();

    std::cout << "\n  -- Handshake --\n";
    test_handshake_both_established();
    test_handshake_sequence_numbers();

    std::cout << "\n  -- Data transfer --\n";
    test_data_a_to_b();
    test_data_b_to_a();
    test_data_bidirectional();
    test_sequence_advances_after_data();
    test_large_send_segmented();

    std::cout << "\n  -- Teardown --\n";
    test_teardown_a_closes_first();
    test_teardown_simultaneous_close();
    test_time_wait_expires();

    std::cout << "\n  -- Retransmit integration --\n";
    test_retransmit_fires_on_timeout();

    std::cout << '\n'
              << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
