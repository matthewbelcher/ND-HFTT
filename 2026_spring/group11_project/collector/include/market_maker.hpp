#pragma once
#include <optional>
#include <vector>
#include <mutex>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <boost/json.hpp>

#include "orderbook.hpp"
#include "btc_tracker.hpp"

// Forward-declare so market_maker.hpp doesn't pull in all of Boost.Beast.
// session.hpp includes kalshi_rest.hpp and passes a pointer at construction.
class KalshiRestClient;

namespace bj = boost::json;

// ---------------------------------------------------------------------------
// MarketMaker
//
// Live paper simulation of the OBI + BTC momentum market making strategy,
// ported directly from analysis/market_maker_sim.py (the one that showed
// 80% win rate, +$20.88 on 54 sessions).
//
// Strategy:
//   Signal: OBI level 1 (IC +0.20) + BTC momentum 10s (IC +0.14).
//   When both signals agree on direction, post a limit order 1 tick inside
//   the current spread.  When signals conflict, skip.
//
//   OBI > +obi_thr AND btc_mom >= 0  →  post YES bid at (best_yes_bid + 1¢)
//   OBI < -obi_thr AND btc_mom <= 0  →  post YES ask at (best_yes_ask - 1¢)
//
//   Cancel entry quote if:
//     (a) OBI flips sign past threshold (original signal gone)
//     (b) mid crosses our quote price (price moved against us)
//     (c) BTC moves strongly against our quote direction (adverse selection)
//
//   After fill, immediately post exit on the opposite side.
//   If exit doesn't fill within max_hold_s → forced taker exit.
//   If BTC spikes against open position → forced taker exit immediately.
//
// Fee model:
//   Maker entry/exit: $0  (Kalshi charges no maker fees on KXBTC15M)
//   Forced taker exit: ceil(0.07 × qty × P × (1-P) × 100) / 100
//
// Thread-safety:
//   on_delta()     — called from KalshiWorker thread.
//   add_btc_price()— called from CoinbaseWorker thread.
//   Both acquire mtx_ so all state is protected.
//   print_summary()— may be called from any thread (acquires mtx_).
// ---------------------------------------------------------------------------

static constexpr double TICK             = 0.01;
static constexpr double CANCEL_THRESHOLD = 300.0;  // delta_fp above = probable cancel, not trade

inline double taker_fee_calc(double price, int qty) {
    return std::ceil(0.07 * qty * price * (1.0 - price) * 100.0) / 100.0;
}

inline double now_sec() {
    return std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// ── Data structures ───────────────────────────────────────────────────────────

struct MMQuote {
    int         direction;   // +1 = YES bid, -1 = YES ask
    double      price;
    int         qty;
    double      posted_at;   // unix seconds
    std::string phase;       // "entry" or "exit"
};

struct MMFill {
    int         direction;
    double      price;
    int         qty;
    double      ts;
    std::string phase;
    double      fee = 0.0;
};

struct MMTrade {
    MMFill              entry;
    std::optional<MMFill> exit;

    double gross_pnl()    const {
        if (!exit) return 0.0;
        return entry.direction * (exit->price - entry.price) * entry.qty;
    }
    double net_pnl()      const {
        if (!exit) return 0.0;
        return gross_pnl() - entry.fee - exit->fee;
    }
    double hold_seconds() const {
        if (!exit) return 0.0;
        return exit->ts - entry.ts;
    }
};

// ── MarketMaker ───────────────────────────────────────────────────────────────

class MarketMaker {
public:
    // rest == nullptr → pure paper mode; rest != nullptr → live trading
    // ticker is required when rest is non-null (used in REST order body)
    // resolution_ts: unix epoch of market resolution (0 = unknown, no cutoff).
    // Stops posting new entry quotes 60s before resolution and cancels any
    // resting entry quote at that point.
    MarketMaker(int    qty           = 10,
                double obi_thr       = 0.05,
                double max_hold_s    = 30.0,
                double btc_cancel    = 0.0001,
                KalshiRestClient* rest = nullptr,
                const std::string& ticker = "",
                time_t resolution_ts = 0)
        : qty_(qty), obi_thr_(obi_thr),
          max_hold_s_(max_hold_s), btc_cancel_thr_(btc_cancel),
          rest_(rest), live_ticker_(ticker),
          deadline_ts_(resolution_ts > 0 ? static_cast<double>(resolution_ts) - 60.0 : 0.0),
          btc_(10.0)
    {}

    // ── Thread-safe public API ────────────────────────────────────────────────

    // Called from CoinbaseWorker thread: feed a BTC mid price tick.
    void add_btc_price(double ts, double price) {
        std::lock_guard<std::mutex> lk(mtx_);
        btc_.add_price(ts, price);
    }

    // Called from KalshiWorker thread: process an orderbook_snapshot.
    void on_snapshot(const bj::object& msg) {
        std::lock_guard<std::mutex> lk(mtx_);
        book_.apply_snapshot(msg);
    }

    // Called from KalshiWorker thread: process a fill channel message.
    // Only meaningful in live mode (rest_ != nullptr); paper mode ignores it.
    void on_fill(const bj::object& msg) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!rest_ || !quote_) return;

        fill_pending_ = false;  // unfreeze state machine regardless of match result

        // Match fill to our resting order.
        auto* ov = msg.if_contains("order_id");
        if (!ov) return;
        std::string fill_order_id = kalshi::parse_str(*ov);
        if (fill_order_id != open_order_id_) {
            std::cerr << "[MM:fill] unexpected order_id=" << fill_order_id
                      << " (expected=" << open_order_id_ << ") — ignoring\n";
            return;
        }

        // Price and count arrive as strings in the fill message ("0.750", "278.00").
        double yes_price = get_str_as_double(msg, "yes_price_dollars");
        if (yes_price <= 0.0) {
            // NO-side fill: convert no_price_dollars to YES equivalent.
            double no_price = get_str_as_double(msg, "no_price_dollars");
            if (no_price > 0.0) yes_price = 1.0 - no_price;
        }
        if (yes_price <= 0.0) return;

        int count = static_cast<int>(get_str_as_double(msg, "count_fp"));
        if (count <= 0) count = 1;

        order_filled_ += count;  // accumulate partial fills for this order
        double ts = now_sec();

        if (quote_->phase == "entry") {
            int expected = quote_->qty;
            if (order_filled_ < expected) {
                std::cout << "[MM:fill] PARTIAL ENTRY " << count << "/" << expected
                          << " (acc=" << order_filled_ << ") @ " << fmt(yes_price, 2) << "\n";
                return;  // wait for remaining fills
            }
            int filled = order_filled_;
            order_filled_ = 0;
            int    log_dir   = quote_->direction;
            double log_price = yes_price;
            MMFill fill{quote_->direction, yes_price, filled, ts, "entry", 0.0};
            open_order_id_ = {};
            open_trade_ = MMTrade{fill};
            quote_ = std::nullopt;
            post_exit_quote(ts);
            std::cout << "[MM:fill] ENTRY dir=" << log_dir
                      << " price=" << fmt(log_price, 2)
                      << " qty="   << filled << "\n";
        } else {
            if (!open_trade_) return;
            int expected = open_trade_->entry.qty;
            if (order_filled_ < expected) {
                std::cout << "[MM:fill] PARTIAL EXIT " << count << "/" << expected
                          << " (acc=" << order_filled_ << ") @ " << fmt(yes_price, 2) << "\n";
                return;  // wait for remaining fills
            }
            int filled = order_filled_;
            order_filled_ = 0;
            MMFill fill{quote_->direction, yes_price, filled, ts, quote_->phase, 0.0};
            open_trade_->exit = fill;
            double net = open_trade_->net_pnl();
            std::cout << "[MM:fill] EXIT price=" << fmt(yes_price, 2)
                      << " net=$"  << fmt(net, 2)
                      << " hold="  << fmt(open_trade_->hold_seconds(), 1) << "s\n";
            open_order_id_ = {};
            completed_.push_back(std::move(*open_trade_));
            open_trade_ = std::nullopt;
            quote_      = std::nullopt;
        }
    }

    // Called from KalshiWorker thread: process an orderbook_delta.
    void on_delta(const bj::object& msg, double ts) {
        std::lock_guard<std::mutex> lk(mtx_);

        // Extract fields once.
        std::string side    = get_str(msg, "side");
        double      price   = get_num(msg, "price_dollars");
        double      delta   = get_num(msg, "delta_fp");

        // 1. Fill check BEFORE applying the delta to the book.
        //    In live mode we use the fill WS channel (on_fill) instead of
        //    this heuristic, which can fire spuriously during rapid order churn.
        if (quote_ && delta < 0.0 && !rest_)
            check_fill(side, price, delta, ts);

        // If a cancel returned 404 (order filled but fill channel hasn't
        // confirmed yet), freeze the state machine.  on_fill() will unfreeze.
        if (fill_pending_) { maybe_print(ts); return; }

        // 2. Apply delta (may change best bid/ask/obi).
        book_.apply_delta(side, price, delta);

        if (!book_.ready()) return;

        double bid = book_.best_yes_bid();
        double ask = book_.best_yes_ask();
        if (bid <= 0.0 || ask >= 1.0 || ask <= bid) return;

        double obi     = book_.obi1();
        auto   btc_mom = btc_.momentum(ts);

        // 3. Position timeout: if we've held too long, exit via taker.
        if (open_trade_ && ts - open_trade_->entry.ts > max_hold_s_) {
            force_taker_exit(bid, ask, ts);
            maybe_print(ts);
            return;
        }

        // 3b. Deadline: cancel resting entry quote and stop posting new ones.
        if (deadline_ts_ > 0.0 && ts >= deadline_ts_) {
            if (quote_ && quote_->phase == "entry") {
                std::cout << "[MM] deadline: cancelling entry quote\n";
                bool fill_detected = live_cancel();
                if (!fill_detected) {
                    if (!finalize_cancel_entry(ts))
                        quote_ = std::nullopt;
                }
            }
            // Allow exit management to continue below if we hold a position.
        }

        // 4. State machine.
        if (!open_trade_ && !quote_)
            maybe_quote(bid, ask, obi, btc_mom, ts);
        else if (open_trade_ && quote_)
            manage_exit_quote(bid, ask, btc_mom, ts);
        else if (!open_trade_ && quote_)
            manage_entry_quote(bid, ask, obi, btc_mom, ts);

        maybe_print(ts);
    }

    // Cancel any resting order and force-close any open position via taker.
    // Call from Session::stop() before print_summary/save_results.
    void cleanup(const std::string& ticker) {
        std::lock_guard<std::mutex> lk(mtx_);
        double bid = book_.best_yes_bid();
        double ask = book_.best_yes_ask();
        double ts  = now_sec();

        // If fill_pending_ is still set (from a prior 404 cancel before workers stopped),
        // synthesize the fill at quote price before cancelling.
        if (fill_pending_ && quote_ && !open_trade_) {
            std::cout << "[MM:" << ticker << "] cleanup: synthesising fill (fill_pending_)\n";
            int filled = order_filled_ > 0 ? order_filled_ : quote_->qty;
            MMFill fill{quote_->direction, quote_->price, filled, ts, "entry", 0.0};
            open_trade_ = MMTrade{fill};
            order_filled_ = 0;
            fill_pending_ = false;
            open_order_id_ = {};
            quote_ = std::nullopt;
        }
        fill_pending_ = false;

        if (quote_) {
            std::cout << "[MM:" << ticker << "] cleanup: cancelling resting "
                      << quote_->phase << " order\n";
            bool fill_detected = live_cancel();
            if (!fill_detected) {
                // 200 cancel: check for partial entry fills
                if (quote_ && !finalize_cancel_entry(ts))
                    quote_ = std::nullopt;
            } else {
                // 404: order filled but fill channel won't arrive (WebSocket closing).
                if (fill_pending_ && quote_) {
                    int filled = std::max(1, order_filled_ > 0 ? order_filled_ : quote_->qty);
                    if (quote_->phase == "entry" && !open_trade_) {
                        std::cout << "[MM:" << ticker << "] cleanup: synthesising entry fill"
                                     " at quote price (fill channel timeout)\n";
                        MMFill fill{quote_->direction, quote_->price, filled, ts, "entry", 0.0};
                        open_trade_ = MMTrade{fill};
                    } else if (quote_->phase == "exit" && open_trade_) {
                        std::cout << "[MM:" << ticker << "] cleanup: synthesising exit fill"
                                     " at quote price (fill channel timeout)\n";
                        MMFill fill{quote_->direction, quote_->price, filled, ts, "exit", 0.0};
                        open_trade_->exit = fill;
                        completed_.push_back(std::move(*open_trade_));
                        open_trade_ = std::nullopt;
                    }
                    order_filled_ = 0;
                    fill_pending_ = false;
                    quote_ = std::nullopt;
                }
            }
        }
        order_filled_ = 0;
        if (open_trade_) {
            std::cout << "[MM:" << ticker << "] cleanup: force-closing open position\n";
            if (bid > 0.0 && ask < 1.0)
                force_taker_exit(bid, ask, ts);
            else
                open_trade_ = std::nullopt;
        }
    }

    // Print end-of-session P&L summary. Safe to call from any thread.
    void print_summary(const std::string& ticker) const {
        std::lock_guard<std::mutex> lk(mtx_);
        int n = static_cast<int>(completed_.size());
        std::cout << "\n[MM:" << ticker << "] ── SESSION SUMMARY ──────────────────────\n";
        if (n == 0) {
            std::cout << "[MM:" << ticker << "] No completed trades.\n";
        } else {
            double gross = 0.0, net = 0.0;
            int wins = 0;
            for (auto& t : completed_) {
                gross += t.gross_pnl();
                net   += t.net_pnl();
                if (t.net_pnl() > 0) ++wins;
            }
            double win_pct = 100.0 * wins / n;
            std::cout << "[MM:" << ticker << "]"
                      << "  trades="      << n
                      << "  win="         << fmt(win_pct, 1) << "%"
                      << "  gross=$"      << fmt(gross, 2)
                      << "  fees=$"       << fmt(total_fees_, 2)
                      << "  net=$"        << fmt(net, 2) << "\n"
                      << "[MM:" << ticker << "]"
                      << "  cancelled="   << n_cancelled_
                      << "  btc_skip="    << n_no_post_btc_
                      << "  btc_cancel="  << n_btc_cancelled_
                      << "  taker_exits=" << n_taker_exits_
                      << "  btc_exits="   << n_btc_exits_ << "\n";
        }
        std::cout << "[MM:" << ticker << "] ──────────────────────────────────────────\n";
    }

    // Write a JSON result file to results_dir/mm_{ticker}_{timestamp}.json.
    // Called once per session from Session::stop(). Safe to call from any thread.
    void save_results(const std::string& ticker, const std::string& results_dir) const {
        std::lock_guard<std::mutex> lk(mtx_);
        try {
            std::filesystem::create_directories(results_dir);

            // Timestamp suffix for filename
            auto now = std::chrono::system_clock::now();
            std::time_t t = std::chrono::system_clock::to_time_t(now);
            struct tm utc; gmtime_r(&t, &utc);
            char tsbuf[32];
            strftime(tsbuf, sizeof(tsbuf), "%Y%m%d_%H%M%S", &utc);

            std::string path = results_dir + "/mm_" + ticker + "_" + tsbuf + ".json";

            int n = static_cast<int>(completed_.size());
            double gross = 0.0, net = 0.0; int wins = 0;
            for (auto& tr : completed_) {
                gross += tr.gross_pnl();
                net   += tr.net_pnl();
                if (tr.net_pnl() > 0) ++wins;
            }

            bj::object summary;
            summary["n_trades"]      = n;
            summary["win_rate"]      = (n > 0) ? static_cast<double>(wins) / n : 0.0;
            summary["gross_pnl"]     = gross;
            summary["total_fees"]    = total_fees_;
            summary["net_pnl"]       = net;
            summary["n_cancelled"]   = n_cancelled_;
            summary["n_btc_skip"]    = n_no_post_btc_;
            summary["n_btc_cancel"]  = n_btc_cancelled_;
            summary["n_taker_exits"] = n_taker_exits_;
            summary["n_btc_exits"]   = n_btc_exits_;
            summary["qty"]           = qty_;
            summary["obi_thr"]       = obi_thr_;
            summary["max_hold_s"]    = max_hold_s_;
            summary["btc_cancel_thr"]= btc_cancel_thr_;

            bj::array trades_arr;
            for (auto& tr : completed_) {
                bj::object t_obj;
                t_obj["entry_price"] = tr.entry.price;
                t_obj["entry_dir"]   = tr.entry.direction;
                t_obj["entry_ts"]    = tr.entry.ts;
                t_obj["entry_phase"] = tr.entry.phase;
                t_obj["qty"]         = tr.entry.qty;
                if (tr.exit) {
                    t_obj["exit_price"] = tr.exit->price;
                    t_obj["exit_phase"] = tr.exit->phase;
                    t_obj["exit_ts"]    = tr.exit->ts;
                    t_obj["exit_fee"]   = tr.exit->fee;
                }
                t_obj["gross_pnl"] = tr.gross_pnl();
                t_obj["net_pnl"]   = tr.net_pnl();
                t_obj["hold_s"]    = tr.hold_seconds();
                trades_arr.push_back(t_obj);
            }

            bj::object root;
            root["ticker"]    = ticker;
            root["timestamp"] = std::string(tsbuf);
            root["summary"]   = summary;
            root["trades"]    = trades_arr;

            std::ofstream f(path);
            f << bj::serialize(root) << "\n";
            std::cout << "[MM:" << ticker << "] results → " << path << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[MM:" << ticker << "] save_results error: " << e.what() << "\n";
        }
    }

private:
    // ── Parameters ────────────────────────────────────────────────────────────
    int    qty_;
    double obi_thr_;
    double max_hold_s_;
    double btc_cancel_thr_;

    // ── Live trading (nullptr = paper mode) ───────────────────────────────────
    KalshiRestClient* rest_        = nullptr;
    std::string       live_ticker_;
    std::string       open_order_id_;   // current resting order; empty if none
    bool              fill_pending_ = false;  // cancel returned 404 → order filled, awaiting on_fill
    double            deadline_ts_ = 0.0;     // stop entry quoting at this unix time (0 = none)

    // ── State (all protected by mtx_) ─────────────────────────────────────────
    mutable std::mutex     mtx_;
    kalshi::KalshiBook     book_;
    BtcTracker             btc_;
    std::optional<MMQuote> quote_;
    std::optional<MMTrade> open_trade_;
    std::vector<MMTrade>   completed_;

    // ── Counters ──────────────────────────────────────────────────────────────
    double last_order_ts_  = 0.0;   // wall time of last live_place or live_cancel
    static constexpr double ORDER_COOLDOWN = 1.0;  // min seconds between entry attempts
    int    order_filled_   = 0;    // qty accumulated for current open_order_id_ (partial fill tracking)
    int    n_cancelled_    = 0;
    int    n_btc_cancelled_= 0;
    int    n_no_post_btc_  = 0;
    int    n_taker_exits_  = 0;
    int    n_btc_exits_    = 0;
    double total_fees_     = 0.0;
    double last_print_     = 0.0;

    // ── Quoting ───────────────────────────────────────────────────────────────

    void maybe_quote(double bid, double ask, double obi,
                     const std::optional<double>& btc_mom, double ts) {
        // Stop posting new entry quotes past the deadline.
        if (deadline_ts_ > 0.0 && ts >= deadline_ts_) return;

        // In live mode, enforce a cooldown so we don't hammer the REST API.
        if (rest_ && (ts - last_order_ts_ < ORDER_COOLDOWN)) return;

        // btc_mom == nullopt → still warming up → use OBI alone (don't block).
        auto btc_agrees = [&](int dir) -> bool {
            if (!btc_mom) return true;
            return (dir == +1) ? (*btc_mom >= 0.0) : (*btc_mom <= 0.0);
        };

        if (obi > obi_thr_) {
            if (!btc_agrees(+1)) { ++n_no_post_btc_; return; }
            double qp = round2(bid + TICK);
            if (qp >= ask) qp = bid;
            if (bid <= 0.01) return;
            quote_ = MMQuote{+1, qp, qty_, ts, "entry"};
            if (!live_place("buy", "yes", to_yes_cents(qp), qty_)) {
                quote_ = std::nullopt;
                return;
            }
        } else if (obi < -obi_thr_) {
            if (!btc_agrees(-1)) { ++n_no_post_btc_; return; }
            double qp = round2(ask - TICK);
            if (qp <= bid) qp = ask;
            if (ask >= 0.99) return;
            quote_ = MMQuote{-1, qp, qty_, ts, "entry"};
            if (!live_place("buy", "no", to_no_cents(qp), qty_)) {
                quote_ = std::nullopt;
                return;
            }
        }
    }

    void manage_entry_quote(double bid, double ask, double obi,
                            const std::optional<double>& btc_mom, double ts) {
        if (!quote_) return;
        MMQuote& q = *quote_;

        // Cancel conditions.
        bool signal_flipped =
            (q.direction == +1 && obi < -obi_thr_) ||
            (q.direction == -1 && obi >  obi_thr_);

        double mid = (bid + ask) / 2.0;
        bool mid_crossed =
            (q.direction == +1 && mid < q.price) ||
            (q.direction == -1 && mid > q.price);

        bool btc_adverse = btc_mom &&
            std::abs(*btc_mom) > btc_cancel_thr_ &&
            ((q.direction == +1 && *btc_mom < 0.0) ||
             (q.direction == -1 && *btc_mom > 0.0));

        if (signal_flipped || mid_crossed || btc_adverse) {
            ++n_cancelled_;
            if (btc_adverse) ++n_btc_cancelled_;
            bool fill_detected = live_cancel();
            if (!fill_detected) {  // 200 cancel: check for partial fills
                if (!finalize_cancel_entry(ts))
                    quote_ = std::nullopt;
            }  // 404 → fill_pending_ set; quote_ kept; on_fill will handle
            return;
        }

        // In live mode don't reprice — cancel+repost would spam the API and
        // the posted price is already in the book at a valid level.
        if (rest_) return;

        // Paper mode: update price in-place to stay 1 tick inside the spread.
        double target = (q.direction == +1)
            ? round2(bid + TICK)
            : round2(ask - TICK);
        if (q.direction == +1 && target >= ask) target = bid;
        if (q.direction == -1 && target <= bid) target = ask;
        if (std::abs(target - q.price) >= TICK)
            q.price = target;
    }

    void manage_exit_quote(double bid, double ask,
                           const std::optional<double>& btc_mom, double ts) {
        if (!quote_ || !open_trade_) return;
        const MMFill& ent = open_trade_->entry;

        // BTC adverse: cancel exit quote then exit via taker immediately.
        if (btc_mom &&
            std::abs(*btc_mom) > btc_cancel_thr_ &&
            ((ent.direction == +1 && *btc_mom < 0.0) ||
             (ent.direction == -1 && *btc_mom > 0.0))) {
            ++n_btc_exits_;
            bool fill_detected = live_cancel();
            if (fill_detected) return;  // exit order already filled — on_fill handles it
            force_taker_exit(bid, ask, ts);
            return;
        }

        // Optimal exit price: 1 tick inside current spread.
        double target;
        if (quote_->direction == +1) {
            target = round2(bid + TICK);
            if (target >= ask) target = bid;
        } else {
            target = round2(ask - TICK);
            if (target <= bid) target = ask;
        }

        if (std::abs(target - quote_->price) < TICK) return;  // price unchanged

        if (rest_) {
            // Live mode: cancel and repost only when cooldown has elapsed.
            if (ts - last_order_ts_ < ORDER_COOLDOWN) return;
            bool fill_detected = live_cancel();
            if (fill_detected) return;  // exit already filled — on_fill handles it
            // Repost for remaining qty (partial fills already confirmed reduce it).
            int remaining = ent.qty - order_filled_;
            if (remaining <= 0) {
                // All qty filled via partial fills; wait for on_fill to close trade.
                quote_ = std::nullopt;
                return;
            }
            quote_->price = target;
            quote_->qty   = remaining;
            bool ok = (ent.direction == +1)
                ? live_place("sell", "yes", to_yes_cents(target), remaining)
                : live_place("sell", "no",  to_no_cents(target),  remaining);
            if (!ok) {
                quote_ = std::nullopt;
                force_taker_exit(bid, ask, ts);
            }
        } else {
            // Paper mode: update price in-place.
            quote_->price = target;
        }
    }

    // ── Fill detection ────────────────────────────────────────────────────────

    // True if a negative delta at (side, price) would consume our resting quote.
    // YES bid (direction=+1): consumed when someone sells YES → negative YES delta at our price.
    // YES ask (direction=-1): equivalent to NO bid at (1-price); consumed by negative NO delta.
    bool delta_hits_quote(const std::string& side, double price,
                          const MMQuote& q) const {
        if (q.direction == +1)
            return side == "yes" && std::abs(price - q.price) < 0.001;
        else {
            double no_equiv = round2(1.0 - q.price);
            return side == "no" && std::abs(price - no_equiv) < 0.001;
        }
    }

    void check_fill(const std::string& side, double price,
                    double delta_fp, double ts) {
        if (!quote_) return;
        const MMQuote& q = *quote_;
        if (!delta_hits_quote(side, price, q)) return;

        double abs_delta = std::abs(delta_fp);
        if (abs_delta > CANCEL_THRESHOLD) return;   // probable cancel, not a trade

        // Level quantity BEFORE this delta (book not yet updated).
        double level_qty = (q.direction == +1)
            ? book_.yes_qty_at(price)
            : book_.no_qty_at(round2(1.0 - q.price));
        if (level_qty <= 0.0) return;

        double our_share = std::min(1.0, static_cast<double>(q.qty) / level_qty);
        int filled = std::max(1, static_cast<int>(std::round(our_share * abs_delta)));
        filled = std::min(filled, q.qty);

        // Kalshi charges $0 maker fees on KXBTC15M.
        MMFill fill{q.direction, q.price, filled, ts, q.phase, 0.0};

        if (q.phase == "entry") {
            int    log_dir   = q.direction;
            double log_price = q.price;
            open_order_id_ = {};  // entry order filled; no cancel needed
            open_trade_ = MMTrade{fill};
            quote_ = std::nullopt;
            post_exit_quote(ts);  // may set open_order_id_ via live_place
            std::cout << "[MM] FILL ENTRY dir=" << log_dir
                      << " price=" << fmt(log_price, 2)
                      << " qty="   << filled << "\n";
        } else {
            // Round trip complete.
            open_trade_->exit = fill;
            double net = open_trade_->net_pnl();
            std::cout << "[MM] FILL EXIT  dir=" << q.direction
                      << " price=" << fmt(q.price, 2)
                      << " net=$"  << fmt(net, 2)
                      << " hold="  << fmt(open_trade_->hold_seconds(), 1) << "s\n";
            open_order_id_ = {};  // exit order filled
            completed_.push_back(std::move(*open_trade_));
            open_trade_ = std::nullopt;
            quote_      = std::nullopt;
        }
    }

    void post_exit_quote(double ts) {
        if (!open_trade_) return;
        const MMFill& ent = open_trade_->entry;
        double bid = book_.best_yes_bid();
        double ask = book_.best_yes_ask();
        if (bid <= 0.0 || ask >= 1.0) return;

        if (ent.direction == +1) {
            // Long YES → sell YES to close
            double ep = round2(ask - TICK);
            if (ep <= bid) ep = ask;
            quote_ = MMQuote{-1, ep, ent.qty, ts, "exit"};
            if (!live_place("sell", "yes", to_yes_cents(ep), ent.qty)) {
                quote_ = std::nullopt;
                force_taker_exit(bid, ask, ts);  // fallback to taker if limit fails
            }
        } else {
            // Long NO → sell NO to close
            double ep = round2(bid + TICK);
            if (ep >= ask) ep = bid;
            quote_ = MMQuote{+1, ep, ent.qty, ts, "exit"};
            if (!live_place("sell", "no", to_no_cents(ep), ent.qty)) {
                quote_ = std::nullopt;
                force_taker_exit(bid, ask, ts);
            }
        }
    }

    void force_taker_exit(double bid, double ask, double ts) {
        if (!open_trade_) return;

        // Cancel any resting exit quote. If it returned 404 (already filled),
        // let on_fill() handle the exit accounting — don't double-exit.
        bool fill_detected = live_cancel();
        if (fill_detected) return;

        const MMFill& ent = open_trade_->entry;

        // Qty remaining after any partial limit exits already confirmed.
        int remaining = ent.qty - order_filled_;
        order_filled_ = 0;
        if (remaining <= 0) return;  // all qty already exited via limit fills

        double exit_price = (ent.direction == +1) ? bid : ask;
        double fee = taker_fee_calc(exit_price, remaining);
        total_fees_ += fee;
        ++n_taker_exits_;

        if (rest_) {
            if (ent.direction == +1)
                live_place("sell", "yes", to_yes_cents(bid), remaining,
                           false, "immediate_or_cancel");
            else
                live_place("sell", "no", to_no_cents(ask), remaining,
                           false, "immediate_or_cancel");
            open_order_id_ = {};  // IOC doesn't persist
        }

        MMFill ef{-ent.direction, exit_price, remaining, ts, "taker_exit", fee};
        open_trade_->exit = ef;
        double net = open_trade_->net_pnl();
        std::cout << "[MM] TAKER EXIT price=" << fmt(exit_price, 2)
                  << " qty=" << remaining
                  << " fee=$" << fmt(fee, 4)
                  << " net=$" << fmt(net, 2) << "\n";
        completed_.push_back(std::move(*open_trade_));
        open_trade_ = std::nullopt;
        quote_      = std::nullopt;
    }

    // ── Status display ────────────────────────────────────────────────────────

    void maybe_print(double ts) {
        if (ts - last_print_ < 5.0) return;
        last_print_ = ts;
        print_status_locked(ts);
    }

    void print_status_locked(double ts) const {
        int n = static_cast<int>(completed_.size());
        double net = 0.0; int wins = 0;
        for (auto& t : completed_) {
            net += t.net_pnl();
            if (t.net_pnl() > 0) ++wins;
        }

        double bid  = book_.best_yes_bid();
        double ask  = book_.best_yes_ask();
        double mid  = (bid + ask) / 2.0;
        double obi  = book_.obi1();
        auto   mom  = btc_.momentum(ts);
        double btcp = btc_.latest_price();

        std::string state;
        if (!open_trade_ && !quote_)
            state = "FLAT";
        else if (!open_trade_ && quote_)
            state = (quote_->direction == +1)
                ? "POSTING BID " + fmt(quote_->price, 2)
                : "POSTING ASK " + fmt(quote_->price, 2);
        else
            state = (open_trade_->entry.direction == +1)
                ? "LONG  @ " + fmt(open_trade_->entry.price, 2)
                : "SHORT @ " + fmt(open_trade_->entry.price, 2);

        std::string mom_s = mom ? (fmt(*mom * 100.0, 4) + "%") : "warming";

        std::cout
            << "[MM] bid=" << fmt(bid, 2)
            << " ask=" << fmt(ask, 2)
            << " mid=" << fmt(mid, 2)
            << " OBI=" << fmt(obi, 3)
            << " btc_mom=" << mom_s
            << " BTC=$" << fmt(btcp, 0)
            << " | " << state
            << " | trades=" << n;
        if (n > 0)
            std::cout << " win=" << fmt(100.0 * wins / n, 1) << "% net=$" << fmt(net, 2);
        std::cout << " | skip=" << n_no_post_btc_
                  << " cancel=" << n_cancelled_ << "\n";
    }

    // ── Live order helpers ────────────────────────────────────────────────────

    // Place a live order and store the returned order_id.
    // side_mapping:
    //   entry  direction=+1 → buy  yes  yes_price=cents
    //   entry  direction=-1 → buy  no   no_price=(100-cents)
    //   exit   entry_dir=+1 → sell yes  yes_price=cents   (closing long YES)
    //   exit   entry_dir=-1 → sell no   no_price=(100-cents) (closing long NO)
    // Returns true on success (paper mode always returns true).
    // On failure, caller must clear quote_ to avoid phantom state.
    bool live_place(const std::string& action, const std::string& side,
                    int price_cents, int qty,
                    bool post_only = true,
                    const std::string& tif = "good_till_canceled") {
        if (!rest_) return true;
        open_order_id_ = rest_->place_order(
            live_ticker_, action, side, price_cents, qty, post_only, tif);
        last_order_ts_ = now_sec();
        order_filled_ = 0;  // reset partial-fill accumulator for new order
        if (open_order_id_.empty()) {
            std::cerr << "[MM:live] place_order failed\n";
            return false;
        }
        std::cout << "[MM:live] placed " << action << " " << side
                  << " " << price_cents << "¢ x" << qty
                  << " order_id=" << open_order_id_ << "\n";
        return true;
    }

    // Returns true if the cancel revealed the order was already filled (HTTP 404).
    // In that case open_order_id_ and quote_ are NOT cleared — on_fill() will handle them.
    // Returns false on clean cancel (200) or other error.
    bool live_cancel() {
        if (!rest_ || open_order_id_.empty()) return false;
        bool was_filled = false;
        bool ok = rest_->cancel_order(open_order_id_, was_filled);
        last_order_ts_ = now_sec();
        if (ok) {
            std::cout << "[MM:live] cancelled " << open_order_id_ << "\n";
            open_order_id_ = {};
            return false;
        }
        if (was_filled) {
            std::cout << "[MM:live] cancel 404: " << open_order_id_
                      << " was filled — waiting for fill channel\n";
            fill_pending_ = true;  // freeze state machine; on_fill will unfreeze
            return true;
        }
        std::cerr << "[MM:live] cancel error for " << open_order_id_ << "\n";
        open_order_id_ = {};
        return false;
    }

    // Call after live_cancel() returns false (200) while in entry phase.
    // If partial fills accumulated, opens a trade with that partial qty and posts exit.
    // Returns true if a partial entry was finalized (quote_ already cleared).
    // Returns false if no partial fills; caller should clear quote_.
    bool finalize_cancel_entry(double ts) {
        if (order_filled_ <= 0 || !quote_ || quote_->phase != "entry") {
            order_filled_ = 0;
            return false;
        }
        int partial_qty = order_filled_;
        order_filled_ = 0;
        std::cout << "[MM:live] partial entry fill: " << partial_qty
                  << "/" << qty_ << " @ " << fmt(quote_->price, 2) << " — entering\n";
        MMFill fill{quote_->direction, quote_->price, partial_qty, ts, "entry", 0.0};
        open_trade_ = MMTrade{fill};
        quote_ = std::nullopt;
        post_exit_quote(ts);
        return true;
    }

    // Map direction + phase to REST action/side.
    // phase="entry": direction=+1 → buy yes; direction=-1 → buy no
    // phase="exit":  entry_dir=+1 → sell yes; entry_dir=-1 → sell no
    //   (pass entry direction as dir when phase=exit)
    static void order_params(int dir, const std::string& phase,
                             std::string& action, std::string& side) {
        if (phase == "entry") {
            action = "buy";
            side   = (dir == +1) ? "yes" : "no";
        } else {
            action = "sell";
            side   = (dir == +1) ? "yes" : "no";
        }
    }

    static int to_yes_cents(double yes_price) {
        return static_cast<int>(std::lround(yes_price * 100.0));
    }
    static int to_no_cents(double yes_price) {
        return static_cast<int>(std::lround((1.0 - yes_price) * 100.0));
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    static double round2(double v) { return std::round(v * 100.0) / 100.0; }

    static std::string fmt(double v, int decimals) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(decimals) << v;
        return ss.str();
    }

    static std::string get_str(const bj::object& obj, const char* key) {
        auto* v = obj.if_contains(key);
        if (!v) return {};
        return kalshi::parse_str(*v);
    }

    static double get_num(const bj::object& obj, const char* key) {
        auto* v = obj.if_contains(key);
        if (!v) return 0.0;
        return kalshi::parse_num(*v);
    }

    // Kalshi fill messages encode numbers as JSON strings ("0.750", "278.00").
    static double get_str_as_double(const bj::object& obj, const char* key) {
        auto* v = obj.if_contains(key);
        if (!v) return 0.0;
        if (v->is_string()) {
            try { return std::stod(std::string(v->as_string())); } catch (...) {}
        }
        return kalshi::parse_num(*v);  // fallback if server sends a number
    }
};
