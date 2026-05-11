// UNIT TESTS FOR WHEN ITEMS ARE MODIFIED TO MAKE SURE THE TIME BOUNDARIES STILL WORK
#include <iostream> 
#include <cassert>
#include <string>
#include <ctime>
#include "ticker.hpp"

// Parse "YYYY-MM-DD HH:MM:SS" as UTC → time_t
static time_t parse_utc(const char* s) {
    struct tm tm{};
    sscanf(s, "%d-%d-%d %d:%d:%d",
        &tm.tm_year, &tm.tm_mon, &tm.tm_mday,
        &tm.tm_hour, &tm.tm_min, &tm.tm_sec);
    tm.tm_year -= 1900;
    tm.tm_mon  -= 1;
    return timegm(&tm);
}

struct Case {
    const char* label;
    const char* utc_time;       // moment we call ticker_at()
    const char* expected_ticker;
};

int main() {
    Case cases[] = {
        // Mid-window: 21:05 UTC is inside the 21:00→21:15 window
        { "mid :00-:15 window",
          "2026-03-04 21:05:00",
          "KXBTC15M-26MAR042115-15" },

        // Right at resolution: 21:15 UTC itself — now in :15→:30 window
        { "exactly at :15",
          "2026-03-04 21:15:00",
          "KXBTC15M-26MAR042130-30" },

        // :59 anticipation — 1 second before :00, still in :45→:00 window
        { "anticipation :59",
          "2026-03-04 21:59:00",
          "KXBTC15M-26MAR042200-00" },

        // :00 on the dot — now in :00→:15 window
        { "exactly :00",
          "2026-03-04 22:00:00",
          "KXBTC15M-26MAR042215-15" },

        // Day rollover: 23:55 → resolves at 00:00 next day
        { "day rollover 23:55",
          "2026-03-04 23:55:00",
          "KXBTC15M-26MAR050000-00" },

        // Month rollover: Mar 31 23:55 → Apr 1 00:00
        { "month rollover Mar31→Apr1",
          "2026-03-31 23:55:00",
          "KXBTC15M-26APR010000-00" },

        // :30 window mid-point
        { "mid :30-:45 window",
          "2026-03-04 21:37:00",
          "KXBTC15M-26MAR042145-45" },

        // :14 anticipation — 1 min before :15, still resolves at :15
        { "anticipation :14",
          "2026-03-04 21:14:00",
          "KXBTC15M-26MAR042115-15" },

        // 1 second before :30 — still in :15→:30 window
        { "1s before :30",
          "2026-03-04 21:29:59",
          "KXBTC15M-26MAR042130-30" },
    };

    int pass = 0, fail = 0;
    for (auto& c : cases) {
        time_t t    = parse_utc(c.utc_time);
        std::string got = ticker_at(t);
        bool ok = (got == c.expected_ticker);
        std::cout << (ok ? "  PASS" : "  FAIL")
                  << "  [" << c.label << "]\n"
                  << "       time    : " << c.utc_time << "\n"
                  << "       expected: " << c.expected_ticker << "\n"
                  << "       got     : " << got << "\n";
        ok ? pass++ : fail++;
    }

    std::cout << "\n" << pass << "/" << (pass+fail) << " passed\n";
    return fail == 0 ? 0 : 1;
}
