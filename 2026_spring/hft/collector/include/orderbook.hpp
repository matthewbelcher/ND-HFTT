#pragma once
#include <array>
#include <string>
#include <cmath>
#include <algorithm>
#include <boost/json.hpp>

namespace bj = boost::json;

// ---------------------------------------------------------------------------
// KalshiBook
//
// Reconstructs the YES/NO order book from Kalshi WebSocket orderbook_snapshot
// and orderbook_delta messages.
//
// Price encoding: Kalshi prices tick in $0.01 from $0.01–$0.99 (99 levels).
// Internal array index = round(price * 100) - 1  →  price 0.01→0, 0.99→98.
//
// Kalshi binary contract relationship:
//   YES ask (implied) = 1.0 - best NO bid
//   This is because YES and NO are complementary: YES + NO pays $1.00.
// ---------------------------------------------------------------------------

namespace kalshi {

inline int price_idx(double price) {
    int idx = static_cast<int>(std::round(price * 100)) - 1;
    return std::max(0, std::min(98, idx));
}

// Parse a boost::json value that may be a string or a number.
inline double parse_num(const bj::value& v) {
    if (v.is_string()) {
        try { return std::stod(std::string(v.as_string())); }
        catch (...) { return 0.0; }
    }
    if (v.is_double()) return v.as_double();
    if (v.is_int64())  return static_cast<double>(v.as_int64());
    if (v.is_uint64()) return static_cast<double>(v.as_uint64());
    return 0.0;
}

inline std::string parse_str(const bj::value& v) {
    if (v.is_string()) return std::string(v.as_string());
    return {};
}

class KalshiBook {
public:
    std::array<double, 99> yes_qty_{};
    std::array<double, 99> no_qty_{};
    bool ready_ = false;

    // Apply a full snapshot message (the "msg" sub-object).
    // Keys: yes_dollars_fp, no_dollars_fp — each is [[price, qty], ...]
    void apply_snapshot(const bj::object& msg) {
        yes_qty_.fill(0.0);
        no_qty_.fill(0.0);

        auto load = [&](const char* key, std::array<double, 99>& arr) {
            auto* v = msg.if_contains(key);
            if (!v || !v->is_array()) return;
            for (auto& pair : v->as_array()) {
                if (!pair.is_array() || pair.as_array().size() < 2) continue;
                double price = parse_num(pair.as_array()[0]);
                double qty   = parse_num(pair.as_array()[1]);
                if (price > 0.0 && qty > 0.0)
                    arr[price_idx(price)] = qty;
            }
        };
        load("yes_dollars_fp", yes_qty_);
        load("no_dollars_fp",  no_qty_);
        ready_ = true;
    }

    // Apply a delta message (the "msg" sub-object).
    // Keys: side ("yes"/"no"), price_dollars, delta_fp.
    void apply_delta(const bj::object& msg) {
        auto* sv = msg.if_contains("side");
        auto* pv = msg.if_contains("price_dollars");
        auto* dv = msg.if_contains("delta_fp");
        if (!sv || !pv || !dv) return;

        std::string side  = parse_str(*sv);
        double      price = parse_num(*pv);
        double      delta = parse_num(*dv);

        if (price <= 0.0 || price >= 1.0) return;
        auto& arr = (side == "yes") ? yes_qty_ : no_qty_;
        int idx = price_idx(price);
        arr[idx] = std::max(0.0, arr[idx] + delta);
    }

    // Same as apply_delta but pre-extracted fields (for fill-check path).
    void apply_delta(const std::string& side, double price, double delta_fp) {
        if (price <= 0.0 || price >= 1.0) return;
        auto& arr = (side == "yes") ? yes_qty_ : no_qty_;
        int idx = price_idx(price);
        arr[idx] = std::max(0.0, arr[idx] + delta_fp);
    }

    bool ready() const { return ready_; }

    double yes_qty_at(double price) const {
        if (price <= 0.0 || price >= 1.0) return 0.0;
        return yes_qty_[price_idx(price)];
    }

    double no_qty_at(double price) const {
        if (price <= 0.0 || price >= 1.0) return 0.0;
        return no_qty_[price_idx(price)];
    }

    double best_yes_bid() const {
        for (int i = 98; i >= 0; --i)
            if (yes_qty_[i] > 0.0) return (i + 1) / 100.0;
        return 0.0;
    }

    double best_no_bid() const {
        for (int i = 98; i >= 0; --i)
            if (no_qty_[i] > 0.0) return (i + 1) / 100.0;
        return 0.0;
    }

    // Implied YES ask = 1.0 - best NO bid (Kalshi complementary contract).
    double best_yes_ask() const {
        double nb = best_no_bid();
        if (nb <= 0.0) return 1.0;
        return std::round((1.0 - nb) * 100.0) / 100.0;
    }

    double mid() const {
        double bid = best_yes_bid();
        double ask = best_yes_ask();
        return (bid > 0.0 && ask < 1.0) ? (bid + ask) / 2.0 : 0.5;
    }

    // OBI at depth 1: (best_yes_bid_qty - best_no_bid_qty) / total.
    // Matches the training signal: IC = +0.20 at 5s horizon.
    double obi1() const {
        double yq = 0.0, nq = 0.0;
        for (int i = 98; i >= 0; --i) {
            if (yq == 0.0 && yes_qty_[i] > 0.0) yq = yes_qty_[i];
            if (nq == 0.0 && no_qty_[i]  > 0.0) nq = no_qty_[i];
            if (yq > 0.0 && nq > 0.0) break;
        }
        double total = yq + nq;
        return (total > 0.0) ? (yq - nq) / total : 0.0;
    }
};

} // namespace kalshi
