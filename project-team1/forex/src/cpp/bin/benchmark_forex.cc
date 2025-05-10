#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "forex_arbitrage.h"

bool parse_line(std::ifstream &csv, std::string &base_currency,
                std::string &quote_currency, double &ask_price,
                double &bid_price, unsigned long int &timestamp);

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <path to csv> [currency1 currency2 ...]" << std::endl;
        return 1;
    }

    // Grab CSV.
    std::ifstream csv(argv[1]);
    std::string _;
    std::getline(csv, _);

    // Grab list of currencies.
    std::vector<std::string> currencies;
    for (int i = 2; i < argc; ++i)
        currencies.emplace_back(argv[i]);

    // Initialize benchmark parameters.
    ForexArbitrage forex_arbitrage(currencies);
    unsigned long long int noarb_time_ns = 0;
    unsigned long long int noarb_n = 0;
    unsigned long long int arb_time_ns = 0;
    unsigned long long int arb_n = 0;
    unsigned long long int find_noarb_time_ns = 0;
    unsigned long long int find_noarb_n = 0;
    unsigned long long int find_arb_time_ns = 0;
    unsigned long long int find_arb_n = 0;

    // Run benchmark
    std::string base_currency;
    std::string quote_currency;
    double ask_price;
    double bid_price;
    unsigned long int timestamp;
    unsigned long int last_timestamp = 0;
    struct arbitrage_opportunity __;
    while (parse_line(csv, base_currency, quote_currency, ask_price, bid_price,
                      timestamp)) {
        if (timestamp != last_timestamp) {
            last_timestamp = timestamp;

            // ##################################################
            // ACTUAL TIMING
            auto start = std::chrono::steady_clock::now();
            bool arb = forex_arbitrage.IsArbitragePossible();
            auto end = std::chrono::steady_clock::now();
            if (arb) {
                arb_n += 1;
                arb_time_ns += static_cast<unsigned long long int>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                         start)
                        .count());
            } else {
                noarb_n += 1;
                noarb_time_ns += static_cast<unsigned long long int>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                         start)
                        .count());
            }

            start = std::chrono::steady_clock::now();
            (void)forex_arbitrage.FindArbitrageOpportunity(__);
            end = std::chrono::steady_clock::now();
            if (arb) {
                find_arb_n += 1;
                find_arb_time_ns += static_cast<unsigned long long int>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                         start)
                        .count());
            } else {
                find_noarb_n += 1;
                find_noarb_time_ns += static_cast<unsigned long long int>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                         start)
                        .count());
            }
            // ##################################################
        }
        forex_arbitrage.Update(base_currency, quote_currency, bid_price);
    }

    std::cout << "No arbitrage time (ns): " << noarb_time_ns << std::endl;
    std::cout << "No arbitrage count:     " << noarb_n << std::endl;
    std::cout << "Arbitrage time (ns):    " << arb_time_ns << std::endl;
    std::cout << "Arbitrage count:        " << arb_n << std::endl;
    std::cout << "Find no arbitrage time (ns):  " << find_noarb_time_ns
              << std::endl;
    std::cout << "Find no arbitrage count:      " << find_noarb_n << std::endl;
    std::cout << "Find arbitrage time (ns):     " << find_arb_time_ns
              << std::endl;
    std::cout << "Find arbitrage count:         " << find_arb_n << std::endl;

    return 0;
}

bool parse_line(std::ifstream &csv, std::string &base_currency,
                std::string &quote_currency, double &ask_price,
                double &bid_price, unsigned long int &timestamp) {
    std::string line;
    std::string field;
    std::vector<std::string> fields;

    if (!std::getline(csv, line))
        return false;

    std::stringstream ss(line);
    while (std::getline(ss, field, ','))
        fields.push_back(field);
    if (fields.size() != 5) {
        std::cerr << "Error: The CSV does not have the expected number of "
                     "columns."
                  << std::endl;
        return false;
    }

    std::string ticker;
    try {
        ticker = fields[1];
        ask_price = std::stod(fields[2]);
        bid_price = std::stod(fields[3]);
        timestamp = std::stoul(fields[4]);
    } catch (const std::exception &e) {
        std::cerr << "Error: Failed to convert to numeric types." << std::endl;
        return false;
    }

    if (ticker.size() != 9) {
        std::cerr << "Error: Ticker symbol is malformed." << std::endl;
        return false;
    }

    base_currency = ticker.substr(2, 3);
    quote_currency = ticker.substr(6, 3);

    return true;
}
