#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include "casche_arbitrage.h"

bool parse_line_polygon(std::ifstream &csv, std::string &base_currency,
                        std::string &quote_currency, double &ask_price,
                        double &bid_price, unsigned long int &timestamp);

int main(int argc, char *argv[]) {
    if (argc != 3 && argc != 4) {
        fmt::print(std::cerr,
                   "Usage: {} <path to file with line-delimited cycles to "
                   "track> <path to Polygon.io CSV forex data> [<path to "
                   "output CSV>]\n",
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
    auto &cycle_to_string = casche_arbitrage.cycle_to_string();
    auto &string_to_pair = casche_arbitrage.string_to_pair();
    auto &cycle_to_profit = casche_arbitrage.cycle_to_profit();

    // Load the Polygon CSV.
    std::ifstream polygon_csv(argv[2]);
    std::string _;
    std::getline(polygon_csv, _);

    // Load the output CSV.
    std::ofstream output(argc == 4 ? argv[3] : "casche_all_arbitrages.csv");
    fmt::print(output, "timestamp,cycle,profit\n");

    // Loop through all timestamps in the dataset and output all arbitrage
    // cycles at each timestep.
    std::string base_currency;
    std::string quote_currency;
    double ask_price;
    double bid_price;
    unsigned long int timestamp;
    unsigned long int last_timestamp = 0;
    while (parse_line_polygon(polygon_csv, base_currency, quote_currency,
                              ask_price, bid_price, timestamp)) {
        // Resolves off-by-one error.
        if (last_timestamp == 0)
            last_timestamp = timestamp;

        // If we finish updating a timestamp, find all arbitrage opportunities.
        if (timestamp != last_timestamp) {
            auto arbitrage_cycles = casche_arbitrage.FindAllArbitrageCycles();
            for (uint32_t arbitrage_cycle : arbitrage_cycles) {
                std::string string_cycle = cycle_to_string[arbitrage_cycle];
                double profit = cycle_to_profit[arbitrage_cycle];
                fmt::print(output, "{},{},{}\n", timestamp, string_cycle,
                           profit);
            }
            last_timestamp = timestamp;
        }

        // Update the currency pair's exchange rate. Only perform the update if
        // the pair is tracked.
        std::string string_pair = base_currency + "/" + quote_currency;
        if (string_to_pair.count(string_pair) == 1)
            casche_arbitrage.Update(base_currency + "/" + quote_currency,
                                    bid_price);
    }

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
