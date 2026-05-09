#pragma once
#include <deque>
#include <utility>
#include <optional>
#include <chrono>

// ---------------------------------------------------------------------------
// BtcTracker
//
// Maintains a rolling window of (unix_sec, price) BTC price ticks and
// computes btc_mom_Ns = (price_now - price_Ns_ago) / price_Ns_ago.
//
// Fed from the Coinbase ticker channel (best_bid+best_ask mid).
// Returns nullopt until at least window_s seconds of history have accumulated.
//
// Thread-safety: NOT thread-safe — callers must hold a mutex.
// ---------------------------------------------------------------------------

class BtcTracker {
public:
    explicit BtcTracker(double window_s = 10.0) : window_s_(window_s) {}

    void add_price(double ts_sec, double price) {
        if (price <= 0.0) return;
        prices_.push_back({ts_sec, price});
        // Prune anything older than 3× window to keep memory bounded.
        double cutoff = ts_sec - window_s_ * 3.0;
        while (!prices_.empty() && prices_.front().first < cutoff)
            prices_.pop_front();
    }

    // Returns the percentage return over the last window_s seconds, or
    // nullopt if there is not enough history yet.
    std::optional<double> momentum(double now_sec) const {
        if (prices_.empty()) return std::nullopt;
        double current = prices_.back().second;
        double cutoff  = now_sec - window_s_;
        double past    = 0.0;
        bool   found   = false;
        for (auto& [ts, px] : prices_) {
            if (ts <= cutoff) { past = px; found = true; }
        }
        if (!found || past == 0.0) return std::nullopt;
        return (current - past) / past;
    }

    bool has_data() const { return !prices_.empty(); }

    double latest_price() const {
        return prices_.empty() ? 0.0 : prices_.back().second;
    }

private:
    double window_s_;
    std::deque<std::pair<double, double>> prices_;  // (unix_sec, price)
};
