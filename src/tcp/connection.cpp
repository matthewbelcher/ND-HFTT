#include "tcp/connection.h"

#include "net/ipv4.h"
#include "net/tcp.h"

#include <algorithm>
#include <chrono>
#include <cstring>

#include <arpa/inet.h>   // htonl

// ─── Helpers ─────────────────────────────────────────────────────────────────

// Generate a time-based ISN (loosely follows RFC 793's 4 µs clock advice).
static uint32_t time_based_isn() {
    auto ns = std::chrono::steady_clock::now().time_since_epoch().count();
    return static_cast<uint32_t>(static_cast<uint64_t>(ns) >> 6);
}

// Serial-number less-than (RFC 1982): handles 32-bit wraparound correctly.
static bool seq_lt(uint32_t a, uint32_t b) {
    return static_cast<int32_t>(b - a) > 0;
}
static bool seq_le(uint32_t a, uint32_t b) {
    return a == b || seq_lt(a, b);
}

// ─── Constructor ─────────────────────────────────────────────────────────────

TCPConnection::TCPConnection(uint32_t local_ip,  uint16_t local_port,
                             uint32_t remote_ip, uint16_t remote_port,
                             SendFn   send_fn,
                             RecvFn   recv_fn,
                             uint32_t initial_seq)
    : local_ip_   (local_ip),
      local_port_ (local_port),
      remote_ip_  (remote_ip),
      remote_port_(remote_port),
      send_fn_    (std::move(send_fn)),
      recv_fn_    (std::move(recv_fn)),
      iss_        (initial_seq != 0 ? initial_seq : time_based_isn())
{}

// ─── IPv4 header factory ─────────────────────────────────────────────────────

IPv4Header TCPConnection::make_ip_header(size_t tcp_payload_len) const {
    IPv4Header ip{};
    ip.version_ihl  = (IPV4_VERSION << 4) | IPV4_IHL_MIN;
    ip.tos          = 0;
    ip.total_length = static_cast<uint16_t>(
                          IPV4_HDR_LEN + TCP_HDR_LEN + tcp_payload_len);
    ip.identification = 0;
    ip.flags_frag   = 0x4000;           // DF set, no fragmentation
    ip.ttl          = IPV4_DEFAULT_TTL;
    ip.protocol     = IPPROTO_TCP_V;
    ip.src_ip       = local_ip_;        // already in network byte order
    ip.dst_ip       = remote_ip_;
    ip.checksum     = 0;                // filled in by ipv4_encode
    return ip;
}

// ─── Segment construction & transmission ─────────────────────────────────────

void TCPConnection::send_segment(uint8_t        flags,
                                 uint32_t       seq,
                                 uint32_t       ack,
                                 const uint8_t* payload,
                                 size_t         payload_len)
{
    const IPv4Header ip = make_ip_header(payload_len);

    TCPHeader tcp{};
    tcp.src_port             = local_port_;
    tcp.dst_port             = remote_port_;
    tcp.seq_num              = seq;
    tcp.ack_num              = ack;
    tcp.data_offset_reserved = tcp_pack_data_offset(5);
    tcp.flags                = flags;
    tcp.window               = static_cast<uint16_t>(rcv_wnd_);
    tcp.urgent_ptr           = 0;

    // Stack-allocate a frame buffer large enough for any segment we build.
    uint8_t buf[IPV4_HDR_LEN + TCP_HDR_LEN + TCP_MSS + 64];
    size_t  n = ipv4_encode(ip, buf, sizeof(buf));
    n += tcp_encode(ip, tcp, payload, payload_len, buf + n, sizeof(buf) - n);

    // Transmit.
    send_fn_(buf, n);

    // Push onto the retransmit queue if this segment consumes sequence space
    // (i.e. carries data, or has SYN / FIN set).
    uint32_t seq_end = seq + static_cast<uint32_t>(payload_len);
    if (flags & TCP_FLAG_SYN) ++seq_end;
    if (flags & TCP_FLAG_FIN) ++seq_end;

    if (seq_end != seq) {
        auto now = std::chrono::steady_clock::now();
        retx_queue_.push(seq, seq_end, buf, n, now);
    }
}

// ─── Public interface ─────────────────────────────────────────────────────────

void TCPConnection::listen() {
    state_ = TCPState::LISTEN;
}

void TCPConnection::connect() {
    // SYN occupies one sequence number — advance snd_nxt_ BEFORE sending so
    // that any re-entrant receive_packet() calls see the correct value.
    iss_     = iss_;          // already set in constructor
    snd_una_ = iss_;
    snd_nxt_ = iss_ + 1;
    state_   = TCPState::SYN_SENT;
    send_segment(TCP_FLAG_SYN, iss_, 0, nullptr, 0);
}

size_t TCPConnection::send(const uint8_t* data, size_t len) {
    if (state_ != TCPState::ESTABLISHED) return 0;
    if (!data || len == 0) return 0;

    // Clamp to peer's advertised window.
    const size_t window_space = (snd_wnd_ > static_cast<uint32_t>(snd_nxt_ - snd_una_))
                                ? snd_wnd_ - static_cast<uint32_t>(snd_nxt_ - snd_una_)
                                : 0u;
    const size_t to_send = std::min(len, window_space);

    size_t sent = 0;
    while (sent < to_send) {
        const size_t seg_len = std::min(to_send - sent,
                                        static_cast<size_t>(TCP_MSS));
        const uint32_t seg_seq = snd_nxt_;
        snd_nxt_ += static_cast<uint32_t>(seg_len);  // advance before callback
        send_segment(TCP_FLAG_ACK | TCP_FLAG_PSH,
                     seg_seq, rcv_nxt_,
                     data + sent, seg_len);
        sent += seg_len;
    }
    return sent;
}

void TCPConnection::close() {
    switch (state_) {
    case TCPState::ESTABLISHED: {
        const uint32_t fin_seq = snd_nxt_;
        snd_nxt_ += 1;  // FIN consumes one sequence number
        state_ = TCPState::FIN_WAIT_1;
        send_segment(TCP_FLAG_FIN | TCP_FLAG_ACK, fin_seq, rcv_nxt_, nullptr, 0);
        break;
    }
    case TCPState::CLOSE_WAIT: {
        const uint32_t fin_seq = snd_nxt_;
        snd_nxt_ += 1;
        state_ = TCPState::LAST_ACK;
        send_segment(TCP_FLAG_FIN | TCP_FLAG_ACK, fin_seq, rcv_nxt_, nullptr, 0);
        break;
    }
    default:
        break;
    }
}

// ─── Inbound packet processing ────────────────────────────────────────────────

void TCPConnection::receive_packet(const uint8_t* buf, size_t len) {
    IPv4Header ip{};
    if (!ipv4_decode(buf, len, ip)) return;

    const size_t ihl_bytes = static_cast<size_t>(ip.version_ihl & 0x0Fu) * 4u;
    if (len < ihl_bytes + TCP_HDR_LEN) return;

    TCPHeader seg{};
    if (!tcp_decode(buf + ihl_bytes, len - ihl_bytes, seg)) return;

    const size_t doff_bytes = static_cast<size_t>(
        tcp_unpack_data_offset(seg.data_offset_reserved)) * 4u;
    if (len < ihl_bytes + doff_bytes) return;

    const uint8_t* payload     = buf + ihl_bytes + doff_bytes;
    const size_t   payload_len = len - ihl_bytes - doff_bytes;

    process_segment(ip, seg, payload, payload_len);
}

void TCPConnection::process_segment(const IPv4Header& ip,
                                    const TCPHeader&  seg,
                                    const uint8_t*    payload,
                                    size_t            payload_len)
{
    (void)ip;   // reserved for checksum validation in a future phase
    switch (state_) {
    case TCPState::CLOSED:       handle_closed      (seg);                    break;
    case TCPState::LISTEN:       handle_listen      (seg);                    break;
    case TCPState::SYN_SENT:     handle_syn_sent    (seg);                    break;
    case TCPState::SYN_RECEIVED: handle_syn_received(seg);                    break;
    case TCPState::ESTABLISHED:  handle_established (seg, payload, payload_len); break;
    case TCPState::FIN_WAIT_1:   handle_fin_wait_1  (seg, payload, payload_len); break;
    case TCPState::FIN_WAIT_2:   handle_fin_wait_2  (seg, payload, payload_len); break;
    case TCPState::CLOSE_WAIT:   handle_close_wait  (seg);                    break;
    case TCPState::LAST_ACK:     handle_last_ack    (seg);                    break;
    case TCPState::TIME_WAIT:    handle_time_wait   (seg);                    break;
    }
}

// ─── State handlers ───────────────────────────────────────────────────────────

void TCPConnection::handle_closed(const TCPHeader& seg) {
    // RFC 793: if the incoming segment is not a RST, send a RST.
    if (seg.flags & TCP_FLAG_RST) return;
    const uint32_t rst_seq = (seg.flags & TCP_FLAG_ACK) ? seg.ack_num : 0u;
    // RST/ACK — no sequence space consumed, skip retx queue.
    const IPv4Header ip = make_ip_header(0);
    TCPHeader rst{};
    rst.src_port             = local_port_;
    rst.dst_port             = remote_port_;
    rst.seq_num              = rst_seq;
    rst.ack_num              = seg.seq_num + 1;
    rst.data_offset_reserved = tcp_pack_data_offset(5);
    rst.flags                = TCP_FLAG_RST | TCP_FLAG_ACK;
    rst.window               = 0;
    uint8_t buf[IPV4_HDR_LEN + TCP_HDR_LEN];
    size_t n = ipv4_encode(ip, buf, sizeof(buf));
    n += tcp_encode(ip, rst, nullptr, 0, buf + n, sizeof(buf) - n);
    send_fn_(buf, n);
}

void TCPConnection::handle_listen(const TCPHeader& seg) {
    if (seg.flags & TCP_FLAG_RST) return;
    if (!(seg.flags & TCP_FLAG_SYN)) return;

    // Passive open: record peer's ISN, send SYN-ACK.
    irs_     = seg.seq_num;
    rcv_nxt_ = irs_ + 1;
    snd_wnd_ = seg.window;

    snd_una_ = iss_;
    snd_nxt_ = iss_ + 1;   // advance before send so callbacks see correct value
    state_   = TCPState::SYN_RECEIVED;

    send_segment(TCP_FLAG_SYN | TCP_FLAG_ACK, iss_, rcv_nxt_, nullptr, 0);
}

void TCPConnection::handle_syn_sent(const TCPHeader& seg) {
    if (seg.flags & TCP_FLAG_RST) {
        state_ = TCPState::CLOSED;
        return;
    }
    if (!(seg.flags & TCP_FLAG_SYN)) return;  // ignore non-SYN in this state

    irs_     = seg.seq_num;
    rcv_nxt_ = irs_ + 1;
    snd_wnd_ = seg.window;

    if (seg.flags & TCP_FLAG_ACK) {
        // Normal case: SYN-ACK received — complete the handshake.
        if (process_ack(seg.ack_num)) {
            state_ = TCPState::ESTABLISHED;
            send_segment(TCP_FLAG_ACK, snd_nxt_, rcv_nxt_, nullptr, 0);
        }
    } else {
        // Simultaneous open: SYN without ACK.
        state_ = TCPState::SYN_RECEIVED;
        send_segment(TCP_FLAG_SYN | TCP_FLAG_ACK, iss_, rcv_nxt_, nullptr, 0);
    }
}

void TCPConnection::handle_syn_received(const TCPHeader& seg) {
    if (seg.flags & TCP_FLAG_RST) {
        state_ = TCPState::CLOSED;
        return;
    }
    if (!(seg.flags & TCP_FLAG_ACK)) return;

    if (process_ack(seg.ack_num))
        state_ = TCPState::ESTABLISHED;
}

void TCPConnection::handle_established(const TCPHeader& seg,
                                       const uint8_t*   payload,
                                       size_t           payload_len)
{
    if (seg.flags & TCP_FLAG_RST) {
        state_ = TCPState::CLOSED;
        return;
    }

    if (seg.flags & TCP_FLAG_ACK) {
        process_ack(seg.ack_num);
        snd_wnd_ = seg.window;
    }

    // Accept in-order data only (Phase 3 does not buffer out-of-order segments).
    if (payload_len > 0 && seg.seq_num == rcv_nxt_)
        process_data(payload, payload_len);

    // FIN from peer: peer has finished sending, we move to CLOSE_WAIT.
    if (seg.flags & TCP_FLAG_FIN) {
        const uint32_t fin_seq = seg.seq_num + static_cast<uint32_t>(payload_len);
        if (fin_seq == rcv_nxt_) {
            rcv_nxt_ += 1;
            state_ = TCPState::CLOSE_WAIT;
            send_segment(TCP_FLAG_ACK, snd_nxt_, rcv_nxt_, nullptr, 0);
        }
    }
}

void TCPConnection::handle_fin_wait_1(const TCPHeader& seg,
                                      const uint8_t*   payload,
                                      size_t           payload_len)
{
    if (seg.flags & TCP_FLAG_RST) {
        state_ = TCPState::CLOSED;
        return;
    }

    if (seg.flags & TCP_FLAG_ACK) {
        process_ack(seg.ack_num);
        snd_wnd_ = seg.window;
    }

    if (payload_len > 0 && seg.seq_num == rcv_nxt_)
        process_data(payload, payload_len);

    if (seg.flags & TCP_FLAG_FIN) {
        const uint32_t fin_seq = seg.seq_num + static_cast<uint32_t>(payload_len);
        if (fin_seq == rcv_nxt_) {
            rcv_nxt_ += 1;
            send_segment(TCP_FLAG_ACK, snd_nxt_, rcv_nxt_, nullptr, 0);
            // Whether our FIN was also ACKed or not, we enter TIME_WAIT
            // (simultaneous close goes here too — CLOSING state omitted per plan).
            state_ = TCPState::TIME_WAIT;
            time_wait_start_ = std::chrono::steady_clock::now();
            return;
        }
    }

    // If our FIN was ACKed (retx queue now empty) and we didn't get a FIN yet,
    // advance to FIN_WAIT_2.
    if (state_ == TCPState::FIN_WAIT_1 && retx_queue_.empty())
        state_ = TCPState::FIN_WAIT_2;
}

void TCPConnection::handle_fin_wait_2(const TCPHeader& seg,
                                      const uint8_t*   payload,
                                      size_t           payload_len)
{
    if (seg.flags & TCP_FLAG_RST) {
        state_ = TCPState::CLOSED;
        return;
    }

    if (seg.flags & TCP_FLAG_ACK) {
        process_ack(seg.ack_num);
        snd_wnd_ = seg.window;
    }

    if (payload_len > 0 && seg.seq_num == rcv_nxt_)
        process_data(payload, payload_len);

    if (seg.flags & TCP_FLAG_FIN) {
        const uint32_t fin_seq = seg.seq_num + static_cast<uint32_t>(payload_len);
        if (fin_seq == rcv_nxt_) {
            rcv_nxt_ += 1;
            send_segment(TCP_FLAG_ACK, snd_nxt_, rcv_nxt_, nullptr, 0);
            state_ = TCPState::TIME_WAIT;
            time_wait_start_ = std::chrono::steady_clock::now();
        }
    }
}

void TCPConnection::handle_close_wait(const TCPHeader& seg) {
    // Waiting for the application to call close(). ACKs for anything we sent
    // earlier can still arrive.
    if (seg.flags & TCP_FLAG_RST) {
        state_ = TCPState::CLOSED;
        return;
    }
    if (seg.flags & TCP_FLAG_ACK)
        process_ack(seg.ack_num);
}

void TCPConnection::handle_last_ack(const TCPHeader& seg) {
    if (seg.flags & TCP_FLAG_ACK) {
        if (process_ack(seg.ack_num))
            state_ = TCPState::CLOSED;
    }
}

void TCPConnection::handle_time_wait(const TCPHeader& seg) {
    // Peer retransmitted its FIN — re-ACK it and restart the 2-MSL timer.
    if (seg.flags & TCP_FLAG_FIN) {
        send_segment(TCP_FLAG_ACK, snd_nxt_, rcv_nxt_, nullptr, 0);
        time_wait_start_ = std::chrono::steady_clock::now();
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

bool TCPConnection::process_ack(uint32_t ack_num) {
    // Accept only if ack_num is strictly greater than snd_una_ and does not
    // exceed snd_nxt_ (i.e. does not ack data we have not sent).
    // Uses serial-number arithmetic to handle potential wraparound.
    if (!seq_lt(snd_una_, ack_num)) return false;   // duplicate or old ACK
    if (seq_lt(snd_nxt_, ack_num))  return false;   // acks unsent data — discard

    snd_una_ = ack_num;
    if (retx_queue_.acknowledge(ack_num))
        retx_queue_.reset_rto();

    return true;
}

void TCPConnection::process_data(const uint8_t* payload, size_t plen) {
    rcv_nxt_ += static_cast<uint32_t>(plen);
    if (recv_fn_) recv_fn_(payload, plen);
    // Send a cumulative ACK for everything received so far.
    send_segment(TCP_FLAG_ACK, snd_nxt_, rcv_nxt_, nullptr, 0);
}

// ─── Timer ───────────────────────────────────────────────────────────────────

std::chrono::milliseconds TCPConnection::tick(
        std::chrono::steady_clock::time_point now)
{
    // Handle TIME_WAIT expiry.
    if (state_ == TCPState::TIME_WAIT) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           now - time_wait_start_);
        if (elapsed >= TCP_TIME_WAIT_DURATION) {
            state_ = TCPState::CLOSED;
            return std::chrono::milliseconds{0};
        }
        return TCP_TIME_WAIT_DURATION - elapsed;
    }

    // Handle retransmit timeout.
    if (auto* entry = retx_queue_.peek_expired(now)) {
        if (entry->retx_count >= MAX_RETRANSMIT) {
            // Too many retransmits — abort the connection.
            state_ = TCPState::CLOSED;
            return std::chrono::milliseconds{0};
        }
        send_fn_(entry->frame.data(), entry->frame.size());
        retx_queue_.record_retransmit(now);
    }

    return retx_queue_.time_until_expiry(now);
}
