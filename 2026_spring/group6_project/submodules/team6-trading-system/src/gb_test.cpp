#include <arpa/inet.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <iostream>
#include <memory>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <unistd.h>

#include <hft/ExchangeClient.h>
#include <hft/ExchangeOrder.h>
#include <hft/PositionTracker.h>
#include <hft/RiskSystem.h>
#include <hft/Strategy.h>
#include <hft/multicast.h>
#include <hft/orderbook.h>

#include "hft/msg/common.h"
#include "strategies/ETFArbitrage.h"
#include "strategies/Liquidate.h"
#include "strategies/Logger.h"
#include "strategies/MarketMaker.h"
#include "strategies/MomentumScalper.h"
#include "strategies/SpreadScalper.h"

constexpr const uint32_t NDFEX_CLIENT_ID = (6);
constexpr const char *NDFEX_USERNAME = ("team6");
constexpr const char *NDFEX_PASSWORD = ("56vjGN5S");

const hft::client::ConnectionSpec GHOSTBOOK_SPEC{
    inet_addr("192.168.13.14"),
    {inet_addr("239.255.255.1"), 12345},
    {inet_addr("239.255.255.2"), 12345},
    {inet_addr("127.0.0.1"), 10000}};

const hft::risk::RiskLimits RISK_LIMITS{.max_qty_per_order = 10,
                                        .max_qty_per_side = 30,
                                        .max_exposure_per_side = 2000,
                                        .max_position = 10,
                                        .max_abs_position_shutdown = 10,
                                        .min_pnl_shutdown = -4500,
                                        .max_orders_per_second = 10,
                                        .max_orders_per_md_update = 10,
                                        .max_inflight_orders = 50,
                                        .min_valid_price = 2,
                                        .max_valid_price = 100000};

constexpr const int CPU_ID = 5;

std::atomic<bool> interrupted(false);

static void print_usage(const char *prog) {
    std::cerr << "Usage: " << prog << " <local_interface> <strategy> [params...]\n\n";
    std::cerr << "Strategies:\n";
    std::cerr << "  MarketMaker     [symbol=1] [qty=4] [half_spread=5] [max_skew=3] [stop_loss=-1000]\n";
    std::cerr << "  MomentumScalper [symbol=4] [qty=2] [window=100] [threshold=0.003] [take_profit=10] [stop_loss=10] [cooldown=200]\n";
    std::cerr << "  ETFArbitrage    [threshold=15] [qty=1] [revert_threshold=5]\n";
    std::cerr << "  SpreadScalper   [symbol=9] [qty=2] [min_spread=10] [target_profit=10] [stop_loss=15]\n";
}

bool setup_hot_thread(int cpu_id) {
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(cpu_id, &cpu_set);
    if (sched_setaffinity(0, sizeof(cpu_set), &cpu_set) == -1) {
        std::cerr << "sched_setaffinity failed: " << strerror(errno) << std::endl;
        return false;
    }
    return true;
}

void interruptionHandler(int sig) {
    if (sig == SIGINT) {
        std::cout << "[INTERRUPT] Beginning cleanup..." << std::endl;
        interrupted = true;
    }
}

int main(const int argc, const char *argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const in_addr_t iface_ip = inet_addr(argv[1]);
    const std::string strategy_name = argv[2];

    hft::client::ConnectionSpec connection = GHOSTBOOK_SPEC;
    connection.iface_ip = iface_ip;

    if (!setup_hot_thread(CPU_ID)) {
        return EXIT_FAILURE;
    }

    signal(SIGINT, interruptionHandler);

    auto client = hft::client::ExchangeClient(connection, RISK_LIMITS);
    client.login(NDFEX_CLIENT_ID, NDFEX_USERNAME, NDFEX_PASSWORD);

    auto &position_tracker = client.get_position_tracker();

    std::unique_ptr<hft::strategy::Strategy> strategy;

    if (strategy_name == "MarketMaker") {
        const auto symbol      = static_cast<hft::msg::symbol_id_t>(argc > 3 ? std::atoi(argv[3]) : 1);
        const auto qty         = static_cast<hft::msg::quantity_t> (argc > 4 ? std::atoi(argv[4]) : 4);
        const auto half_spread = static_cast<hft::msg::price_t>    (argc > 5 ? std::atoi(argv[5]) : 5);
        const auto max_skew    = static_cast<int64_t>               (argc > 6 ? std::atoi(argv[6]) : 3);
        const double stop_loss =                                      argc > 7 ? std::atof(argv[7]) : -1000.0;
        strategy = std::make_unique<hft::strategy::impl::MarketMaker>(
            client, symbol, qty, half_spread, max_skew, stop_loss);

    } else if (strategy_name == "MomentumScalper") {
        const auto symbol       = static_cast<hft::msg::symbol_id_t>(argc > 3 ? std::atoi(argv[3]) : 4);
        const auto qty          = static_cast<hft::msg::quantity_t> (argc > 4 ? std::atoi(argv[4]) : 2);
        const auto window       = static_cast<size_t>               (argc > 5 ? std::atoi(argv[5]) : 100);
        const double threshold  =                                     argc > 6 ? std::atof(argv[6]) : 0.003;
        const auto take_profit  = static_cast<hft::msg::price_t>    (argc > 7 ? std::atoi(argv[7]) : 10);
        const auto stop_loss    = static_cast<hft::msg::price_t>    (argc > 8 ? std::atoi(argv[8]) : 10);
        const auto cooldown     = static_cast<uint32_t>             (argc > 9 ? std::atoi(argv[9]) : 200);
        strategy = std::make_unique<hft::strategy::impl::MomentumScalper>(
            client, symbol, qty, window, threshold, take_profit, stop_loss, cooldown);

    } else if (strategy_name == "ETFArbitrage") {
        const auto threshold       = static_cast<hft::msg::price_t>   (argc > 3 ? std::atoi(argv[3]) : 15);
        const auto qty             = static_cast<hft::msg::quantity_t>(argc > 4 ? std::atoi(argv[4]) : 1);
        const auto revert_threshold= static_cast<hft::msg::price_t>   (argc > 5 ? std::atoi(argv[5]) : 5);
        strategy = std::make_unique<hft::strategy::impl::ETFArbitrage>(
            client, threshold, qty, revert_threshold);

    } else if (strategy_name == "SpreadScalper") {
        const auto symbol      = static_cast<hft::msg::symbol_id_t>(argc > 3 ? std::atoi(argv[3]) : 9);
        const auto qty         = static_cast<hft::msg::quantity_t> (argc > 4 ? std::atoi(argv[4]) : 2);
        const auto min_spread  = static_cast<hft::msg::price_t>    (argc > 5 ? std::atoi(argv[5]) : 10);
        const auto target      = static_cast<hft::msg::price_t>    (argc > 6 ? std::atoi(argv[6]) : 10);
        const auto stop_loss   = static_cast<hft::msg::price_t>    (argc > 7 ? std::atoi(argv[7]) : 15);
        strategy = std::make_unique<hft::strategy::impl::SpreadScalper>(
            client, symbol, qty, min_spread, target, stop_loss);

    } else {
        std::cerr << "[ERROR] Unknown strategy: " << strategy_name << "\n\n";
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::cout << "[INFO] Running strategy: " << strategy_name << std::endl;

    static const std::vector<hft::msg::symbol_id_t> LOG_SYMS = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    uint64_t tick_count = 0;

    while (!interrupted) {
        client.update();

        if (++tick_count % 10000 == 0) {
            hft::strategy::log::log_pnl(position_tracker, LOG_SYMS, client);
        }
    }

    // Destroy the strategy before liquidating so it cancels its own orders.
    strategy.reset();

    client.cancel_all_orders();
    hft::strategy::impl::Liquidate liquidate(
        client, client.get_orderbook(), position_tracker,
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
        RISK_LIMITS.max_qty_per_order);

    while (true) {
        client.update();
        liquidate.on_tick();

        if (liquidate.is_done()) {
            std::cout << "[INFO] All positions flat... exiting" << std::endl;
            break;
        }
    }

    return EXIT_SUCCESS;
}
