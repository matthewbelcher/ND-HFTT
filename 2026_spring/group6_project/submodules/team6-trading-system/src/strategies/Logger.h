#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <vector>

#include <hft/ExchangeClient.h>
#include <hft/PositionTracker.h>
#include <hft/msg/common.h>

namespace hft::strategy::log {

/* Tick alignment helpers — used by all strategies */

inline msg::price_t round_dn(msg::price_t p, int tick = 5) {
    return (p / tick) * tick;
}

inline msg::price_t round_up(msg::price_t p, int tick = 5) {
    return ((p + tick - 1) / tick) * tick;
}

/* Timestamp string: HH:MM:SS.mmm */

inline std::string timestamp() {
    using namespace std::chrono;
    const auto now   = system_clock::now();
    const auto ms    = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    const std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream ss;
    ss << std::setfill('0')
       << std::setw(2) << tm.tm_hour << ':'
       << std::setw(2) << tm.tm_min  << ':'
       << std::setw(2) << tm.tm_sec  << '.'
       << std::setw(3) << ms.count();
    return ss.str();
}

/* Log a single timestamped action for a symbol */

inline void log_action(msg::symbol_id_t sym, std::string_view action) {
    std::cout << '[' << timestamp() << "] [SYM:" << sym << "] " << action << '\n';
}

/* Compact single-line position summary */

inline void log_positions(const risk::PositionTracker &tracker,
                           const std::vector<msg::symbol_id_t> &syms) {
    std::cout << '[' << timestamp() << "] POSITIONS:";
    for (const auto sym : syms) {
        const int64_t pos = tracker.get_position(sym);
        std::cout << "  SYM" << sym << '=' << std::showpos << pos << std::noshowpos;
    }
    std::cout << '\n';
}

/* Full P&L table */

inline void log_pnl(const risk::PositionTracker &tracker,
                    const std::vector<msg::symbol_id_t> &syms,
                    client::ExchangeClient &client) {
    std::cout << '[' << timestamp() << "] P&L REPORT\n";
    std::cout << std::setw(5)  << "SYM"
              << std::setw(10) << "POS"
              << std::setw(14) << "REALIZED"
              << std::setw(14) << "UNREALIZED"
              << std::setw(14) << "TOTAL"
              << '\n';
    std::cout << std::string(57, '-') << '\n';
    for (const auto sym : syms) {
        const auto bbo = client.get_bbo(sym);
        const msg::price_t mid =
            (bbo.bid_price > 0 && bbo.ask_price > 0)
                ? (bbo.bid_price + bbo.ask_price) / 2
                : 0;
        const double realized   = tracker.get_realized_pnl(sym);
        const double unrealized = (mid > 0) ? tracker.get_unrealized_pnl(sym, mid) : 0.0;
        const double total      = realized + unrealized;
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(5)  << sym
                  << std::setw(10) << std::showpos << tracker.get_position(sym) << std::noshowpos
                  << std::setw(13) << realized   << '$'
                  << std::setw(13) << unrealized << '$'
                  << std::setw(13) << total      << '$'
                  << '\n';
    }
    std::cout << std::flush;
}

} // namespace hft::strategy::log
