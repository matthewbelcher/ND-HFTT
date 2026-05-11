#ifndef HFT_TCPSTACK_RETRANSMIT_H
#define HFT_TCPSTACK_RETRANSMIT_H

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

// RTO policy constants (RFC 6298 §2).
// Initial RTO of 200 ms is intentionally tighter than the RFC-recommended 1 s;
// in an HFT context sub-millisecond links make 1 s far too conservative.
constexpr std::chrono::milliseconds RTO_INITIAL{200};
constexpr std::chrono::milliseconds RTO_MAX    {60'000};  // 60 s hard cap

// Give up retransmitting after this many attempts.
constexpr uint32_t MAX_RETRANSMIT = 15;

// TIME_WAIT duration — 2 × MSL. We use 4 s (abbreviated from 2 min) so that
// in-process tests can exercise the transition without actually sleeping.
constexpr std::chrono::milliseconds TCP_TIME_WAIT_DURATION{4'000};

// One entry in the retransmit queue: a complete IP+TCP frame ready to resend,
// annotated with the sequence-number range it covers and send metadata.
struct RetransmitEntry {
    uint32_t             seq;        // first sequence number in this segment
    uint32_t             seq_end;    // seq + data_len (SYN / FIN each count as 1)
    std::vector<uint8_t> frame;      // raw IP+TCP bytes (no Ethernet header)
    std::chrono::steady_clock::time_point sent_at;
    uint32_t             retx_count; // number of times retransmitted (0 = first send)
};

// Retransmit queue: holds unacknowledged segments in send order.
//
// Also owns the RTO value for this connection. Exponential backoff is applied
// inside record_retransmit(); a fresh ACK resets it via reset_rto().
//
// Time is injected via the `now` parameters so tests can advance a synthetic
// clock without sleeping.
class RetransmitQueue {
public:
    explicit RetransmitQueue(
        std::chrono::milliseconds initial_rto = RTO_INITIAL);

    // Add a new segment. seq_end = seq + data_len; SYN and FIN each add 1.
    void push(uint32_t seq,
              uint32_t seq_end,
              const uint8_t* buf,
              size_t         len,
              std::chrono::steady_clock::time_point now);

    // Remove all entries fully covered by ack_num (seq_end <= ack_num).
    // Returns true if at least one entry was removed.
    bool acknowledge(uint32_t ack_num);

    // Return a pointer to the oldest entry if it has timed out (elapsed ≥ rto_),
    // otherwise nullptr. The caller should retransmit it and call record_retransmit().
    RetransmitEntry* peek_expired(std::chrono::steady_clock::time_point now);

    // Update the front entry after retransmission: reset its timer and
    // apply exponential backoff to rto_ (doubles, capped at RTO_MAX).
    void record_retransmit(std::chrono::steady_clock::time_point now);

    // Reset rto_ to RTO_INITIAL — call after a fresh ACK advances snd_una.
    void reset_rto();

    // Time until the oldest entry could expire (or zero if it already has).
    // Returns a large value if the queue is empty.
    std::chrono::milliseconds time_until_expiry(
        std::chrono::steady_clock::time_point now) const;

    bool   empty() const { return entries_.empty(); }
    size_t size()  const { return entries_.size();  }
    std::chrono::milliseconds rto() const { return rto_; }

    const RetransmitEntry* front() const {
        return entries_.empty() ? nullptr : &entries_.front();
    }

private:
    std::deque<RetransmitEntry> entries_;
    std::chrono::milliseconds   rto_;
};

#endif //HFT_TCPSTACK_RETRANSMIT_H
