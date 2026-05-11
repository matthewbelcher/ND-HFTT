#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <thread>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <boost/json.hpp>
#include "ticker.hpp"
#include "session.hpp"

namespace bj = boost::json;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void sleep_until_epoch(time_t target) {
    while (true) {
        time_t now = time(nullptr);
        if (now >= target) return;
        long nap = std::min(target - now, (long)1);
        std::this_thread::sleep_for(std::chrono::seconds(nap));
    }
}

static std::string fmt_time(time_t t) {
    struct tm utc;
    gmtime_r(&t, &utc);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S UTC", &utc);
    return buf;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct Config {
    KalshiConfig   kalshi;
    CoinbaseConfig coinbase;
    MMConfig       mm;
    std::string    rawdata_dir;
};

static Config load_config(const std::string& kalshi_key_id,
                           const std::string& kalshi_pem_path,
                           const std::string& coinbase_json_path,
                           const std::string& rawdata_dir,
                           const MMConfig&    mm_cfg) {
    Config cfg;
    cfg.kalshi.key_id   = kalshi_key_id;
    cfg.kalshi.pem_path = kalshi_pem_path;
    cfg.rawdata_dir     = rawdata_dir;
    cfg.mm              = mm_cfg;

    std::ifstream f(coinbase_json_path);
    if (!f.is_open()) throw std::runtime_error("Cannot open " + coinbase_json_path);
    std::string contents((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
    auto jv  = bj::parse(contents);
    auto& jo = jv.as_object();
    cfg.coinbase.api_key         = std::string(jo.at("name").as_string());
    cfg.coinbase.private_key_pem = std::string(jo.at("privateKey").as_string());
    return cfg;
}

// ---------------------------------------------------------------------------
// Scheduler
//
// Markets resolve at :00, :15, :30, :45 Eastern time.
// Each session covers one 15-min market window and lives for exactly 17 min:
//   starts at  resolution - 16 min  (:44 / :59 / :14 / :29)
//   stops  at  resolution +  1 min  (:01 / :16 / :31 / :46)
//
// Each session runs in a detached thread that owns its Session object,
// sleeps for its lifetime, then calls stop() and exits. The thread cleans
// up everything on exit — no handles, no vector, no leaks.
//
// During the intentional 2-min overlap two sessions run simultaneously,
// which is fine — they write to separate CSV files.
//
// Example (started at 15:56 UTC):
//   15:59  spawn session for ...1600-00  (runs 15:59 -> 16:16)
//   16:14  spawn session for ...1615-15  (runs 16:14 -> 16:31)
//   16:29  spawn session for ...1630-30  ...
// ---------------------------------------------------------------------------

static constexpr int START_BEFORE = 16*60;  // spawn 16 min before resolution
static constexpr int STOP_AFTER   = 60;     //  stop  1 min after  resolution
static constexpr int LIFETIME     = START_BEFORE + STOP_AFTER;  // 17 min

static time_t first_valid_resolution(time_t now) {
    return next_eastern_boundary(now, START_BEFORE - 1);
}

static void launch_detached(const std::string& ticker,
                             Config cfg,           // by value so we can mutate
                             int lifetime_sec,
                             time_t resolution_ts) {
    cfg.mm.resolution_ts = resolution_ts;
    std::thread([ticker, cfg, lifetime_sec]() {
        try {
            auto session = std::make_unique<Session>(
                ticker, cfg.kalshi, cfg.coinbase, cfg.rawdata_dir, cfg.mm);
            std::this_thread::sleep_for(std::chrono::seconds(lifetime_sec));
            session->stop();
        } catch (const std::exception& e) {
            std::cerr << "[Session:" << ticker << "] fatal: " << e.what() << "\n";
        }
    }).detach();
}

class Scheduler {
public:
    explicit Scheduler(Config cfg) : cfg_(std::move(cfg)) {}

    void run() {
        std::filesystem::create_directories(cfg_.rawdata_dir);
        std::cout << "[Scheduler] starting. rawdata -> " << cfg_.rawdata_dir << "\n";

        time_t res = first_valid_resolution(time(nullptr));

        std::cout << "[Scheduler] first session anticipation at "
                  << fmt_time(res - START_BEFORE)
                  << " (market resolves " << fmt_time(res) << ")\n";

        while (true) {
            time_t spawn_at = res - START_BEFORE;
            sleep_until_epoch(spawn_at);
            time_t now = time(nullptr);

            // Guard: missed the window entirely (e.g. system suspended)
            if (now >= res) {
                std::cout << "[Scheduler] missed window for " << make_ticker(res)
                          << " (resolution " << fmt_time(res)
                          << " already passed), finding next...\n";
                res = first_valid_resolution(now);
                continue;
            }

            // Guard: too close to resolution to be useful
            if (res - now < 30) {
                std::cout << "[Scheduler] too close to resolution for " << make_ticker(res)
                          << " (" << (res - now) << "s remaining), skipping\n";
                res = next_eastern_boundary(res, 1);
                continue;
            }

            std::string ticker    = make_ticker(res);
            int         remaining = (int)(spawn_at + LIFETIME - now);

            std::cout << "[Scheduler] spawning " << ticker
                      << " | now="      << fmt_time(now)
                      << " | resolves=" << fmt_time(res)
                      << " | stops="    << fmt_time(now + remaining) << "\n";

            launch_detached(ticker, cfg_, remaining, res);

            // Advance to next Eastern-time boundary — avoids drift across DST
            res = next_eastern_boundary(res, 1);
        }
    }

private:
    Config cfg_;
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <kalshi_key_id> <kalshi_pem_path> <coinbase_json_path>"
                     " [rawdata_dir] [qty] [obi_thr] [max_hold_s] [btc_cancel] [live]\n"
                  << "\n"
                  << "  kalshi_key_id      : UUID from Kalshi API keys page\n"
                  << "  kalshi_pem_path    : path to your .pem private key file\n"
                  << "  coinbase_json_path : path to cdp_api_key.json\n"
                  << "  rawdata_dir        : CSV output directory (default: rawdata)\n"
                  << "\n"
                  << "  Market maker parameters:\n"
                  << "  qty         contracts per quote       (default: 10)\n"
                  << "  obi_thr     OBI threshold to post     (default: 0.05)\n"
                  << "  max_hold_s  seconds before taker exit (default: 30)\n"
                  << "  btc_cancel  |btc_mom_10s| cancel thr  (default: 0.0001)\n"
                  << "  live        1 = place real Kalshi orders (default: 0 = paper)\n"
                  << "\n"
                  << "Example (paper):\n"
                  << "  " << argv[0]
                  << " 773b19f6-...-f99 ../secrets/TestExample1.pem misc/cdp_api_key.json rawdata\n"
                  << "Example (live, qty=1):\n"
                  << "  " << argv[0]
                  << " 773b19f6-...-f99 ../secrets/TestExample1.pem misc/cdp_api_key.json rawdata 1 0.05 30 0.0001 1\n";
        return 1;
    }

    std::string kalshi_key_id   = argv[1];
    std::string kalshi_pem_path = argv[2];
    std::string coinbase_json   = argv[3];
    std::string rawdata_dir     = (argc >= 5) ? argv[4] : "rawdata";

    MMConfig mm_cfg;
    if (argc >= 6)  mm_cfg.qty          = std::stoi(argv[5]);
    if (argc >= 7)  mm_cfg.obi_thr      = std::stod(argv[6]);
    if (argc >= 8)  mm_cfg.max_hold_s   = std::stod(argv[7]);
    if (argc >= 9)  mm_cfg.btc_cancel   = std::stod(argv[8]);
    if (argc >= 10) mm_cfg.live_trading = std::stoi(argv[9]) != 0;

    std::cout << "[Config] MM: qty=" << mm_cfg.qty
              << " obi_thr=" << mm_cfg.obi_thr
              << " max_hold=" << mm_cfg.max_hold_s << "s"
              << " btc_cancel=" << mm_cfg.btc_cancel
              << " live=" << (mm_cfg.live_trading ? "YES" : "no") << "\n";

    try {
        Config cfg = load_config(kalshi_key_id, kalshi_pem_path, coinbase_json,
                                 rawdata_dir, mm_cfg);
        Scheduler scheduler(std::move(cfg));
        scheduler.run();
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << "\n";
        return 1;
    }

    return 0;
}