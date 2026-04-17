#ifndef HFT_TCPSTACK_CONNECTION_H
#define HFT_TCPSTACK_CONNECTION_H

#include "tcp/tcp_state.h"
#include "tcp/retransmit.h"
#include "net/ipv4.h"
#include "net/tcp.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>

// Maximum segment size we advertise and honour for outgoing data segments.
constexpr uint16_t TCP_MSS = 1460;

// Per-connection TCP state machine implementing RFC 793.
//
// ── Data flow ──────────────────────────────────────────────────────────────
//   Outbound: whenever the connection needs to send a segment it calls
//             send_fn_ with the raw IP+TCP bytes (no Ethernet header).
//   Inbound:  the caller routes raw IP+TCP bytes to receive_packet().
//
// ── In-process wiring for tests ────────────────────────────────────────────
//   Wire two connections by pointing each one's send_fn_ at the other's
//   receive_packet(). The full handshake completes synchronously inside a
//   single call to connect() via nested callbacks.
//
// ── Thread safety ──────────────────────────────────────────────────────────
//   Not thread-safe. All methods must be called from the same thread.
//
// ── Sequence-number arithmetic ─────────────────────────────────────────────
//   We use int32_t-cast comparisons (serial-number arithmetic, RFC 1982).
//   Phase 3 uses small ISNs, so the simpler unsigned comparisons in
//   process_ack() are also safe in practice.
class TCPConnection {
public:
    // Callback invoked to transmit a segment (IP header + TCP header + payload).
    using SendFn = std::function<void(const uint8_t*, size_t)>;

    // Callback invoked when application data arrives from the peer.
    using RecvFn = std::function<void(const uint8_t*, size_t)>;

    // Callback invoked when the connection is aborted (e.g. retransmit limit
    // exceeded). The connection is already CLOSED when this fires.
    using ErrorFn = std::function<void()>;

    // Construct a connection endpoint.
    //   local_ip / remote_ip  — in NETWORK byte order (htonl or inet_pton).
    //   initial_seq           — ISN override (0 = time-based random; set
    //                           explicitly in tests for reproducibility).
    //   error_fn              — called when the connection aborts; may be null.
    //   time_wait_duration    — TIME_WAIT timeout; default is TCP_TIME_WAIT_DURATION.
    //                           Pass a shorter value in tests to avoid long waits.
    TCPConnection(uint32_t                  local_ip,
                  uint16_t                  local_port,
                  uint32_t                  remote_ip,
                  uint16_t                  remote_port,
                  SendFn                    send_fn,
                  RecvFn                    recv_fn            = nullptr,
                  uint32_t                  initial_seq        = 0,
                  ErrorFn                   error_fn           = nullptr,
                  std::chrono::milliseconds time_wait_duration = TCP_TIME_WAIT_DURATION);

    // Passive open: CLOSED → LISTEN.
    void listen();

    // Active open: CLOSED → SYN_SENT, sends SYN.
    void connect();

    // Send application data. Only valid in ESTABLISHED state.
    // Segments the data into MSS-sized chunks; honours the peer's window.
    // Returns the number of bytes accepted (may be less than len if the
    // send window is exhausted).
    size_t send(const uint8_t* data, size_t len);

    // Initiate graceful close: send FIN.
    //   ESTABLISHED  → FIN_WAIT_1
    //   CLOSE_WAIT   → LAST_ACK
    // Returns true if a FIN was sent; false if the current state does not
    // permit initiating a close (the call is a no-op in that case).
    bool close();

    // Process an incoming packet routed to this connection.
    // buf points to the first byte of the IPv4 header (no Ethernet header).
    void receive_packet(const uint8_t* buf, size_t len);

    // Timer tick: handle retransmit timeouts and TIME_WAIT expiry.
    // Pass std::chrono::steady_clock::now() (or a synthetic value in tests).
    // Returns the time until the next scheduled event.
    std::chrono::milliseconds tick(std::chrono::steady_clock::time_point now);

    // ── Accessors ────────────────────────────────────────────────────────
    TCPState state()       const { return state_; }
    uint32_t snd_una()     const { return snd_una_; }
    uint32_t snd_nxt()     const { return snd_nxt_; }
    uint32_t rcv_nxt()     const { return rcv_nxt_; }
    uint16_t local_port()  const { return local_port_; }
    uint16_t remote_port() const { return remote_port_; }

private:
    // Build and transmit one IP+TCP segment.
    // Segments that consume sequence space (SYN, FIN, data) are pushed onto
    // the retransmit queue automatically.
    // IMPORTANT: callers must advance snd_nxt_ BEFORE calling this so that
    // any re-entrant receive_packet() call sees the correct snd_nxt_.
    void send_segment(uint8_t        flags,
                      uint32_t       seq,
                      uint32_t       ack,
                      const uint8_t* payload,
                      size_t         payload_len);

    // Dispatch an incoming segment to the appropriate state handler.
    void process_segment(const IPv4Header& ip,
                         const TCPHeader&  seg,
                         const uint8_t*    payload,
                         size_t            payload_len);

    // Per-state handlers.
    void handle_closed      (const TCPHeader& seg, size_t payload_len);
    void handle_listen      (const TCPHeader& seg);
    void handle_syn_sent    (const TCPHeader& seg);
    void handle_syn_received(const TCPHeader& seg);
    void handle_established (const TCPHeader& seg, const uint8_t* payload, size_t plen);
    void handle_fin_wait_1  (const TCPHeader& seg, const uint8_t* payload, size_t plen);
    void handle_fin_wait_2  (const TCPHeader& seg, const uint8_t* payload, size_t plen);
    void handle_close_wait  (const TCPHeader& seg);
    void handle_last_ack    (const TCPHeader& seg);
    void handle_time_wait   (const TCPHeader& seg);

    // Advance snd_una_ to ack_num if it is a valid new acknowledgement.
    // Drains the retransmit queue and resets RTO on progress.
    // Returns true if snd_una_ advanced.
    bool process_ack(uint32_t ack_num);

    // Deliver in-order data to recv_fn_, advance rcv_nxt_, send an ACK.
    void process_data(const uint8_t* payload, size_t plen);

    // Construct the IPv4 header for an outgoing segment of the given payload size.
    IPv4Header make_ip_header(size_t tcp_payload_len) const;

    // Build and transmit a one-byte zero-window probe from the oldest
    // unacknowledged payload. Does NOT push onto the retransmit queue.
    void send_zero_window_probe();

    // ── Endpoints ────────────────────────────────────────────────────────
    uint32_t local_ip_;
    uint16_t local_port_;
    uint32_t remote_ip_;
    uint16_t remote_port_;

    // ── Callbacks ────────────────────────────────────────────────────────
    SendFn  send_fn_;
    RecvFn  recv_fn_;
    ErrorFn error_fn_;

    // ── State ────────────────────────────────────────────────────────────
    TCPState state_{ TCPState::CLOSED };

    // ── Send sequence variables (RFC 793 §3.2) ───────────────────────────
    uint32_t iss_    { 0 };      // initial send sequence number
    uint32_t snd_una_{ 0 };      // oldest unacknowledged sequence number
    uint32_t snd_nxt_{ 0 };      // next sequence number to send
    uint32_t snd_wnd_{ 65535 };  // peer's last advertised receive window

    // ── Receive sequence variables ────────────────────────────────────────
    uint32_t irs_    { 0 };      // initial receive sequence number
    uint32_t rcv_nxt_{ 0 };      // next expected inbound sequence number
    uint32_t rcv_wnd_{ 65535 };  // our receive window (advertised to peer)

    // ── Retransmit queue + RTO ────────────────────────────────────────────
    RetransmitQueue retx_queue_;

    // ── TIME_WAIT timer ───────────────────────────────────────────────────
    std::chrono::milliseconds             time_wait_duration_;
    std::chrono::steady_clock::time_point time_wait_start_{};

    // ── Zero-window persist timer ─────────────────────────────────────────
    bool                                  persist_active_{false};
    std::chrono::steady_clock::time_point persist_start_{};
    std::chrono::milliseconds             persist_rto_{RTO_INITIAL};
};

#endif //HFT_TCPSTACK_CONNECTION_H
