#pragma once

#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <iostream>

#include "ticker.hpp"
#include "csv_writer.hpp"
#include "kalshi_rest.hpp"
#include "market_maker.hpp"
#include "kalshi_worker.hpp"
#include "coinbase_worker.hpp"

struct KalshiConfig {
    std::string key_id;
    std::string pem_path;
};

struct CoinbaseConfig {
    std::string api_key;
    std::string private_key_pem;  // PEM string from JSON
};

struct MMConfig {
    int    qty           = 10;
    double obi_thr       = 0.05;
    double max_hold_s    = 30.0;
    double btc_cancel    = 0.0001;
    std::string results_dir  = "../results";
    bool        live_trading = false;   // true = place real Kalshi orders
    time_t      resolution_ts = 0;     // unix epoch of market resolution
};

// ---------------------------------------------------------------------------
// Session state machine
//
//   RUNNING ──► STOPPING ──► STOPPED
//       │
//       └──► (future) ORDER_PENDING ──► FILLED / EXPIRED
//
// The Scheduler checks state before spawning the next session so it can
// block on an overlapping ORDER_PENDING if needed.
// ---------------------------------------------------------------------------
enum class SessionState {
    RUNNING,        // workers active, collecting data
    ORDER_PENDING,  // signal fired, order submitted, awaiting fill (future)
    STOPPING,       // stop() called, joins in progress
    STOPPED,        // fully done, safe to destroy
};

// One session = one 15-min market window.
// It owns a KalshiWorker + CoinbaseWorker, each in its own thread.
class Session {
public:
    Session(const std::string& ticker,
            const KalshiConfig& kalshi_cfg,
            const CoinbaseConfig& cb_cfg,
            const std::string& rawdata_dir,
            const MMConfig& mm_cfg = MMConfig{})
        : ticker_(ticker), state_(SessionState::RUNNING), mm_cfg_(mm_cfg)
    {
        std::string kalshi_csv = rawdata_dir + "/" + ticker + ".csv";
        std::string btc_csv    = rawdata_dir + "/BTC-" + ticker + ".csv";

        auto kalshi_csv_w = std::make_shared<CsvWriter>(kalshi_csv);
        auto btc_csv_w    = std::make_shared<CsvWriter>(btc_csv);

        // Create REST client if live trading is enabled.
        if (mm_cfg.live_trading) {
            rest_client_ = std::make_unique<KalshiRestClient>(
                kalshi_cfg.key_id, kalshi_cfg.pem_path);
            std::cout << "[Session:" << ticker_ << "] LIVE TRADING enabled\n";
        }

        mm_ = std::make_shared<MarketMaker>(
            mm_cfg.qty, mm_cfg.obi_thr, mm_cfg.max_hold_s, mm_cfg.btc_cancel,
            rest_client_.get(), ticker_, mm_cfg.resolution_ts);

        kalshi_worker_ = std::make_unique<KalshiWorker>(
            kalshi_cfg.key_id, kalshi_cfg.pem_path, ticker, kalshi_csv_w, mm_);
        cb_worker_ = std::make_unique<CoinbaseWorker>(
            cb_cfg.api_key, cb_cfg.private_key_pem, btc_csv_w, mm_);

        std::cout << "[Session:" << ticker_ << "] starting\n";

        kalshi_thread_ = std::thread([this]{ kalshi_worker_->run(); });
        cb_thread_     = std::thread([this]{ cb_worker_->run(); });
    }

    // Non-blocking: signals workers to exit, then joins. Call from the
    // session's owning thread (i.e. the Scheduler's watchdog thread).
    void stop() {
        SessionState expected = SessionState::RUNNING;
        if (!state_.compare_exchange_strong(expected, SessionState::STOPPING)) {
            // Already stopping/stopped, or order pending — don't double-stop.
            // If ORDER_PENDING, the trading layer is responsible for calling
            // stop() once the order resolves.
            return;
        }
        std::cout << "[Session:" << ticker_ << "] stopping\n";
        kalshi_worker_->stop();
        cb_worker_->stop();
        if (kalshi_thread_.joinable()) kalshi_thread_.join();
        if (cb_thread_.joinable())     cb_thread_.join();
        state_ = SessionState::STOPPED;
        std::cout << "[Session:" << ticker_ << "] stopped\n";
        if (mm_) {
            mm_->cleanup(ticker_);  // cancel resting orders, close open position
            mm_->print_summary(ticker_);
            mm_->save_results(ticker_, mm_cfg_.results_dir);
        }
    }

    // ── State accessors ──────────────────────────────────────────────────────

    SessionState state() const { return state_.load(); }

    bool is_stopped()       const { return state_ == SessionState::STOPPED; }
    bool is_order_pending() const { return state_ == SessionState::ORDER_PENDING; }

    // Called by your future trading layer when a signal fires.
    // Returns false if the session is no longer RUNNING (e.g. already stopping).
    bool set_order_pending() {
        SessionState expected = SessionState::RUNNING;
        return state_.compare_exchange_strong(expected, SessionState::ORDER_PENDING);
    }

    // Called by your future trading layer when the order fills or expires.
    // After this, the Scheduler's watchdog will call stop() on the next cycle.
    void clear_order_pending() {
        SessionState expected = SessionState::ORDER_PENDING;
        state_.compare_exchange_strong(expected, SessionState::RUNNING);
    }

    const std::string& ticker() const { return ticker_; }

private:
    std::string ticker_;
    std::atomic<SessionState> state_;
    MMConfig                          mm_cfg_;
    std::unique_ptr<KalshiRestClient> rest_client_;  // null in paper mode
    std::shared_ptr<MarketMaker>      mm_;
    std::unique_ptr<KalshiWorker>     kalshi_worker_;
    std::unique_ptr<CoinbaseWorker>   cb_worker_;
    std::thread kalshi_thread_;
    std::thread cb_thread_;
};