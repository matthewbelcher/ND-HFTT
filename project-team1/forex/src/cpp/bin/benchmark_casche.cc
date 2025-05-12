#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include "casche_arbitrage.h"

bool parse_line_polygon(std::ifstream &csv, std::string &base_currency,
                        std::string &quote_currency, double &ask_price,
                        double &bid_price, unsigned long int &timestamp);

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fmt::print(std::cerr,
                   "Usage: {} <path to file with line-delimited cycles to "
                   "track> <path to Polygon.io CSV forex data>\n",
                   argv[0]);
        return 1;
    }

    // Load the CascheArbitrage object.
    std::ifstream cycles_file(argv[1]);
    std::string string_cycle;
    std::vector<std::string> string_cycles;
    while (std::getline(cycles_file, string_cycle))
        string_cycles.push_back(string_cycle);
    CascheArbitrage casche_arbitrage(string_cycles);
    auto &string_to_pair = casche_arbitrage.string_to_pair();

    // Load the Polygon CSV.
    std::ifstream polygon_csv(argv[2]);
    std::string _;
    std::getline(polygon_csv, _);

    // Initialize benchmark parameters.
    unsigned long long int noarb_time_ns = 0;
    unsigned long long int noarb_n = 0;
    unsigned long long int arb_time_ns = 0;
    unsigned long long int arb_n = 0;
    unsigned long long int allarb_time_ns = 0;
    unsigned long long int allarb_n = 0;

    // Run benchmark
    std::string base_currency;
    std::string quote_currency;
    double ask_price;
    double bid_price;
    unsigned long int timestamp;
    while (parse_line_polygon(polygon_csv, base_currency, quote_currency,
                              ask_price, bid_price, timestamp)) {
        if (string_to_pair.count(base_currency + "/" + quote_currency) == 0)
            continue;
        uint32_t pair = string_to_pair.at(base_currency + "/" + quote_currency);
        uint32_t cycle;
        // ##################################################
        // ACTUAL TIMING
        auto start = std::chrono::steady_clock::now();
        bool arb = casche_arbitrage.NewArbitrage(pair, bid_price, cycle);
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
        (void)casche_arbitrage.FindAllArbitrageCycles();
        end = std::chrono::steady_clock::now();
        allarb_n += 1;
        allarb_time_ns += static_cast<unsigned long long int>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count());
        // ##################################################
        casche_arbitrage.Update(pair, bid_price);
    }

    std::cout << "No arbitrage time (ns): " << noarb_time_ns << std::endl;
    std::cout << "No arbitrage count:     " << noarb_n << std::endl;
    std::cout << "Arbitrage time (ns):    " << arb_time_ns << std::endl;
    std::cout << "Arbitrage count:        " << arb_n << std::endl;
    std::cout << "All arbitrage time (ns):  " << allarb_time_ns << std::endl;
    std::cout << "All arbitrage count:      " << allarb_n << std::endl;

    return 0;
}

bool parse_line_polygon(std::ifstream &csv, std::string &base_currency,
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
        fmt::print(std::cerr,
                   "Error: The CSV does not have the expected number of "
                   "columns.\n");
        return false;
    }

    std::string ticker;
    try {
        ticker = fields[1];
        ask_price = std::stod(fields[2]);
        bid_price = std::stod(fields[3]);
        timestamp = std::stoul(fields[4]);
    } catch (const std::exception &e) {
        fmt::print(std::cerr, "Error: Failed to convert to numeric types.\n");
        return false;
    }

    if (ticker.size() != 9) {
        fmt::print(std::cerr, "Error: Ticker symbol is malformed.\n");
        return false;
    }

    base_currency = ticker.substr(2, 3);
    quote_currency = ticker.substr(6, 3);

    return true;
}
