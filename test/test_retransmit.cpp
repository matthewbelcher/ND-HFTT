#include "tcp/retransmit.h"

#include <cstdint>
#include <cstring>
#include <iostream>

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

// Synthetic time helpers — avoids real-clock sleeping in tests.
using TP = std::chrono::steady_clock::time_point;
using MS = std::chrono::milliseconds;

static TP epoch()          { return TP{}; }
static TP ms_after(int ms) { return epoch() + MS{ms}; }

// Dummy frame bytes (content does not matter for RetransmitQueue tests).
static const uint8_t kFrame[] = { 0x45, 0x00, 0x00, 0x28, 0x00, 0x00,
                                   0x40, 0x00, 0x40, 0x06, 0xde, 0xad,
                                   0x7f, 0x00, 0x00, 0x01, 0x7f, 0x00,
                                   0x00, 0x01, 0xd4, 0x31, 0x1f, 0x90,
                                   0x00, 0x00, 0x03, 0xe8, 0x00, 0x00,
                                   0x00, 0x00, 0x50, 0x02, 0xff, 0xff,
                                   0x00, 0x00, 0x00, 0x00 };

// ── Tests ─────────────────────────────────────────────────────────────────────

static void test_push_increases_size() {
    const char* name = "RetransmitQueue: push() increments size";
    RetransmitQueue q;
    if (!q.empty()) { fail(name, "queue not empty on construction"); return; }

    q.push(1000, 1001, kFrame, sizeof(kFrame), ms_after(0));
    if (q.size() != 1) { fail(name, "size not 1 after first push"); return; }

    q.push(1001, 1461, kFrame, sizeof(kFrame), ms_after(0));
    if (q.size() != 2) { fail(name, "size not 2 after second push"); return; }

    pass(name);
}

static void test_acknowledge_drains_covered_entries() {
    const char* name = "RetransmitQueue: acknowledge() removes fully-covered entries";
    RetransmitQueue q;
    // seq [1000, 1001), [1001, 1461), [1461, 2921)
    q.push(1000, 1001, kFrame, sizeof(kFrame), ms_after(0));
    q.push(1001, 1461, kFrame, sizeof(kFrame), ms_after(0));
    q.push(1461, 2921, kFrame, sizeof(kFrame), ms_after(0));

    // ACK 1461: covers first two entries (seq_end=1001 and 1461 both <= 1461).
    bool removed = q.acknowledge(1461);
    if (!removed)        { fail(name, "acknowledge returned false"); return; }
    if (q.size() != 1)   { fail(name, "wrong size after partial ack"); return; }
    if (q.front()->seq != 1461) { fail(name, "wrong front after partial ack"); return; }

    // ACK 2921: covers the last entry.
    q.acknowledge(2921);
    if (!q.empty()) { fail(name, "queue not empty after full ack"); return; }

    pass(name);
}

static void test_acknowledge_returns_false_on_no_removal() {
    const char* name = "RetransmitQueue: acknowledge() returns false when nothing removed";
    RetransmitQueue q;
    q.push(1000, 1461, kFrame, sizeof(kFrame), ms_after(0));

    // ACK 1000 — does not cover [1000, 1461).
    bool removed = q.acknowledge(1000);
    if (removed)      { fail(name, "returned true when ack < seq_end"); return; }
    if (q.size() != 1){ fail(name, "entry was removed unexpectedly"); return; }

    pass(name);
}

static void test_peek_expired_before_rto() {
    const char* name = "RetransmitQueue: peek_expired() returns nullptr before RTO";
    RetransmitQueue q(MS{200});
    q.push(1000, 1001, kFrame, sizeof(kFrame), ms_after(0));

    // Check at t=199ms — not yet expired.
    auto* e = q.peek_expired(ms_after(199));
    if (e != nullptr) { fail(name, "returned non-null before RTO elapsed"); return; }

    pass(name);
}

static void test_peek_expired_at_rto() {
    const char* name = "RetransmitQueue: peek_expired() returns entry at exactly RTO";
    RetransmitQueue q(MS{200});
    q.push(1000, 1001, kFrame, sizeof(kFrame), ms_after(0));

    // At exactly t=200ms the entry should be considered expired (elapsed >= rto).
    auto* e = q.peek_expired(ms_after(200));
    if (e == nullptr)     { fail(name, "returned nullptr at exactly RTO"); return; }
    if (e->seq != 1000)   { fail(name, "wrong seq in expired entry");      return; }
    if (e->retx_count != 0){ fail(name, "retx_count should still be 0");  return; }

    pass(name);
}

static void test_record_retransmit_doubles_rto() {
    const char* name = "RetransmitQueue: record_retransmit() doubles RTO (exponential backoff)";
    RetransmitQueue q(MS{200});
    q.push(1000, 1001, kFrame, sizeof(kFrame), ms_after(0));

    // First retransmit at t=200ms.
    q.record_retransmit(ms_after(200));
    if (q.rto() != MS{400}) {
        fail(name, "RTO not doubled after first retransmit");
        return;
    }
    if (q.front()->retx_count != 1) {
        fail(name, "retx_count not incremented");
        return;
    }

    // Second retransmit at t=600ms (400ms after last send).
    q.record_retransmit(ms_after(600));
    if (q.rto() != MS{800}) {
        fail(name, "RTO not doubled after second retransmit");
        return;
    }

    pass(name);
}

static void test_record_retransmit_caps_at_rto_max() {
    const char* name = "RetransmitQueue: record_retransmit() caps RTO at RTO_MAX";
    RetransmitQueue q(MS{32'000});  // start near the cap
    q.push(1000, 1001, kFrame, sizeof(kFrame), ms_after(0));

    q.record_retransmit(ms_after(32'000));  // 32000 * 2 = 64000 > 60000
    if (q.rto() != RTO_MAX) {
        fail(name, "RTO exceeded RTO_MAX");
        return;
    }
    // A second doubling must not exceed the cap either.
    q.record_retransmit(ms_after(96'000));
    if (q.rto() != RTO_MAX) {
        fail(name, "RTO exceeded RTO_MAX on subsequent retransmit");
        return;
    }

    pass(name);
}

static void test_reset_rto() {
    const char* name = "RetransmitQueue: reset_rto() restores RTO to RTO_INITIAL";
    RetransmitQueue q(MS{200});
    q.push(1000, 1001, kFrame, sizeof(kFrame), ms_after(0));

    q.record_retransmit(ms_after(200));  // RTO → 400
    q.record_retransmit(ms_after(600));  // RTO → 800
    if (q.rto() != MS{800}) { fail(name, "RTO not 800 before reset"); return; }

    q.reset_rto();
    if (q.rto() != RTO_INITIAL) {
        fail(name, "RTO not reset to RTO_INITIAL");
        return;
    }

    pass(name);
}

static void test_timer_resets_after_retransmit() {
    const char* name = "RetransmitQueue: entry not expired again immediately after retransmit";
    RetransmitQueue q(MS{200});
    q.push(1000, 1001, kFrame, sizeof(kFrame), ms_after(0));

    // Expire at t=200, retransmit — timer reset to t=200.
    q.peek_expired(ms_after(200));
    q.record_retransmit(ms_after(200));

    // At t=399 (199ms after retransmit) with new RTO=400ms, should not be expired.
    auto* e = q.peek_expired(ms_after(399));
    if (e != nullptr) {
        fail(name, "entry expired too early after retransmit");
        return;
    }

    // At t=600 (400ms after retransmit), should be expired.
    e = q.peek_expired(ms_after(600));
    if (e == nullptr) {
        fail(name, "entry not expired after new RTO elapsed");
        return;
    }

    pass(name);
}

static void test_time_until_expiry() {
    const char* name = "RetransmitQueue: time_until_expiry() returns correct value";
    RetransmitQueue q(MS{200});
    q.push(1000, 1001, kFrame, sizeof(kFrame), ms_after(0));

    // At t=50: 150 ms remain.
    auto remaining = q.time_until_expiry(ms_after(50));
    if (remaining != MS{150}) {
        fail(name, "wrong time_until_expiry before RTO");
        return;
    }

    // At t=200: already expired — returns 0.
    remaining = q.time_until_expiry(ms_after(200));
    if (remaining != MS{0}) {
        fail(name, "expected 0 when already expired");
        return;
    }

    pass(name);
}

static void test_front_seq_correct() {
    const char* name = "RetransmitQueue: front() reflects the oldest unacked entry";
    RetransmitQueue q;
    q.push(500, 501, kFrame, sizeof(kFrame), ms_after(0));
    q.push(501, 502, kFrame, sizeof(kFrame), ms_after(0));

    if (q.front()->seq != 500) {
        fail(name, "front seq is wrong on first entry");
        return;
    }
    q.acknowledge(501);
    if (q.front()->seq != 501) {
        fail(name, "front seq is wrong after partial ack");
        return;
    }

    pass(name);
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=== Phase 3: Retransmit Queue Tests ===\n\n";

    test_push_increases_size();
    test_acknowledge_drains_covered_entries();
    test_acknowledge_returns_false_on_no_removal();
    test_peek_expired_before_rto();
    test_peek_expired_at_rto();
    test_record_retransmit_doubles_rto();
    test_record_retransmit_caps_at_rto_max();
    test_reset_rto();
    test_timer_resets_after_retransmit();
    test_time_until_expiry();
    test_front_seq_correct();

    std::cout << '\n'
              << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
