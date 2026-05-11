#pragma once
#include <string>
#include <ctime>

// Kalshi BTC 15M ticker format:
//   KXBTC15M-YYMONDDHHMMM-MM
//
// All times in Kalshi tickers are US/Eastern (EDT = UTC-4 in summer, EST = UTC-5 in winter).
// Markets resolve at :00, :15, :30, :45 EASTERN TIME — these are NOT UTC boundaries.
//
// Examples (current date Mar 16 2026 = EDT, UTC-4):
//   Resolves 14:15 EDT = 18:15 UTC  -> KXBTC15M-26MAR161415-15
//   Resolves 14:30 EDT = 18:30 UTC  -> KXBTC15M-26MAR161430-30
//   Resolves 15:00 EDT = 19:00 UTC  -> KXBTC15M-26MAR161500-00
//
// Internal scheduling always works in UTC epochs.

static const char* MONTH_NAMES[] = {
    "JAN","FEB","MAR","APR","MAY","JUN",
    "JUL","AUG","SEP","OCT","NOV","DEC"
};

// Returns the UTC offset for US/Eastern at time t (handles EDT/EST automatically).
// EDT: second Sunday March 2:00am → first Sunday November 2:00am  = UTC-4
// EST: otherwise                                                    = UTC-5
inline int eastern_utc_offset(time_t t) {
    struct tm utc; gmtime_r(&t, &utc);

    // Second Sunday in March (DST start): find first Sunday, add 7
    struct tm mar1 = {}; mar1.tm_year=utc.tm_year; mar1.tm_mon=2; mar1.tm_mday=1;
    time_t mar1_t = timegm(&mar1);
    struct tm mar1_s; gmtime_r(&mar1_t, &mar1_s);
    int dow = mar1_s.tm_wday; // 0=Sun
    int second_sun_mar = (dow==0 ? 1 : 8-dow) + 7;

    // First Sunday in November (DST end)
    struct tm nov1 = {}; nov1.tm_year=utc.tm_year; nov1.tm_mon=10; nov1.tm_mday=1;
    time_t nov1_t = timegm(&nov1);
    struct tm nov1_s; gmtime_r(&nov1_t, &nov1_s);
    dow = nov1_s.tm_wday;
    int first_sun_nov = (dow==0 ? 1 : 8-dow);

    // DST starts at 2am EST = 7am UTC on second Sunday of March
    struct tm ds = {}; ds.tm_year=utc.tm_year; ds.tm_mon=2; ds.tm_mday=second_sun_mar; ds.tm_hour=7;
    time_t dst_start = timegm(&ds);

    // DST ends at 2am EDT = 6am UTC on first Sunday of November
    struct tm de = {}; de.tm_year=utc.tm_year; de.tm_mon=10; de.tm_mday=first_sun_nov; de.tm_hour=6;
    time_t dst_end = timegm(&de);

    return (t >= dst_start && t < dst_end) ? -4*3600 : -5*3600;
}

// Convert a UTC epoch to Eastern wall-clock struct tm.
inline struct tm utc_to_eastern(time_t t) {
    time_t eastern_epoch = t + eastern_utc_offset(t);
    struct tm est; gmtime_r(&eastern_epoch, &est);
    return est;
}

// Convert an Eastern wall-clock struct tm back to UTC epoch.
inline time_t eastern_to_utc(struct tm est, int offset_seconds) {
    time_t eastern_epoch = timegm(&est);
    return eastern_epoch - offset_seconds; // subtract offset (offset is negative, so this adds)
}

// Returns the UTC epoch of the next :00/:15/:30/:45 Eastern-time boundary
// that is >= t + min_seconds_ahead.
inline time_t next_eastern_boundary(time_t t, int min_seconds_ahead = 0) {
    int offset = eastern_utc_offset(t);
    // Convert to Eastern time
    time_t et = t + offset;
    struct tm e; gmtime_r(&et, &e);
    e.tm_sec = 0;

    for (int h = 0; h <= 3; h++) {
        for (int m : {0, 15, 30, 45}) {
            struct tm c = e; c.tm_hour += h; c.tm_min = m; c.tm_sec = 0;
            time_t eastern_candidate = timegm(&c);
            // Convert back to UTC
            time_t utc_candidate = eastern_candidate - offset;
            if (utc_candidate > t + min_seconds_ahead) return utc_candidate;
        }
    }
    return t + 15*60; // fallback
}

// Build the Kalshi ticker string from a UTC resolution epoch.
inline std::string make_ticker(time_t resolution_utc) {
    struct tm e = utc_to_eastern(resolution_utc);
    int yy  = e.tm_year % 100;
    int mon = e.tm_mon;
    int day = e.tm_mday;
    int hr  = e.tm_hour;
    int min = e.tm_min;
    char buf[64];
    snprintf(buf, sizeof(buf),
        "KXBTC15M-%02d%s%02d%02d%02d-%02d",
        yy, MONTH_NAMES[mon], day, hr, min, min);
    return std::string(buf);
}

// Convenience: ticker for market active at UTC time t
inline std::string ticker_at(time_t t) {
    return make_ticker(next_eastern_boundary(t));
}

// Market open (UTC) = resolution - 15 minutes
inline time_t market_open(time_t resolution_utc) {
    return resolution_utc - 15*60;
}