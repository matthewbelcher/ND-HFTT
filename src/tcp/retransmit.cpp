#include "tcp/retransmit.h"

#include <algorithm>

RetransmitQueue::RetransmitQueue(std::chrono::milliseconds initial_rto)
    : rto_(initial_rto)
{}

void RetransmitQueue::push(uint32_t seq,
                           uint32_t seq_end,
                           const uint8_t* buf,
                           size_t         len,
                           std::chrono::steady_clock::time_point now)
{
    RetransmitEntry e;
    e.seq        = seq;
    e.seq_end    = seq_end;
    e.frame.assign(buf, buf + len);
    e.sent_at    = now;
    e.retx_count = 0;
    entries_.push_back(std::move(e));
}

bool RetransmitQueue::acknowledge(uint32_t ack_num) {
    bool removed = false;
    // Remove all entries whose sequence range is fully covered by ack_num.
    while (!entries_.empty() && entries_.front().seq_end <= ack_num) {
        entries_.pop_front();
        removed = true;
    }
    return removed;
}

RetransmitEntry* RetransmitQueue::peek_expired(
        std::chrono::steady_clock::time_point now)
{
    if (entries_.empty()) return nullptr;

    auto& front   = entries_.front();
    auto  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - front.sent_at);
    if (elapsed >= rto_)
        return &front;

    return nullptr;
}

void RetransmitQueue::record_retransmit(
        std::chrono::steady_clock::time_point now)
{
    if (entries_.empty()) return;
    entries_.front().sent_at = now;
    ++entries_.front().retx_count;

    // Exponential backoff: double RTO, cap at RTO_MAX.
    auto doubled = rto_ * 2;
    rto_ = (doubled > RTO_MAX) ? RTO_MAX : doubled;
}

void RetransmitQueue::reset_rto() {
    rto_ = RTO_INITIAL;
}

std::chrono::milliseconds RetransmitQueue::time_until_expiry(
        std::chrono::steady_clock::time_point now) const
{
    if (entries_.empty())
        return std::chrono::milliseconds{1'000};  // nothing pending

    auto& front   = entries_.front();
    auto  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - front.sent_at);
    if (elapsed >= rto_)
        return std::chrono::milliseconds{0};

    return rto_ - elapsed;
}
