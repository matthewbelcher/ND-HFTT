#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "forex_arbitrage.h"

const std::vector<std::string> CURRENCIES = {"USD", "EUR", "JPY", "GBP", "CNH",
                                             "AUD", "CAD", "CHF", "HKD", "SGD"};

bool parse_line(std::ifstream &csv, std::string &base_currency,
                std::string &quote_currency, double &ask_price,
                double &bid_price, unsigned long int &timestamp);

int main(int argc, char *argv[]) {
    if (argc != 3 && argc != 4) {
        std::cerr
            << "Usage: " << argv[0]
            << " <path to Polygon.io CSV forex data> <number of "
               "timestamps to iterate through> <anything to indicate verbose>"
            << std::endl;
        return 1;
    }

    std::ifstream csv(argv[1]);
    std::string _;
    std::getline(csv, _);

    unsigned long int n_timestamps = 1 + std::stoul(argv[2]);

    bool verbose = false;
    if (argc == 4)
        verbose = true;

    ForexArbitrage forex_arbitrage(CURRENCIES);
    const auto &id_to_currency = forex_arbitrage.id_to_currency();

    // Update the graph.
    std::string base_currency;
    std::string quote_currency;
    double ask_price;
    double bid_price;
    unsigned long int timestamp;
    unsigned long int last_timestamp = 0;
    while (parse_line(csv, base_currency, quote_currency, ask_price, bid_price,
                      timestamp)) {
        if (timestamp != last_timestamp) {
            last_timestamp = timestamp;
            --n_timestamps;
            if (n_timestamps == 0)
                break;
        }

        if (verbose) {
            std::cout << base_currency << " -> " << quote_currency << " "
                      << bid_price << std::endl;
        }

        forex_arbitrage.Update(base_currency, quote_currency, bid_price);
    }

    // Find the arbitrage opportunities.
    struct arbitrage_opportunity arbitrage_opportunity;
    bool arbitrage_possible =
        forex_arbitrage.FindArbitrageOpportunity(arbitrage_opportunity);
    auto graph = forex_arbitrage.graph();

    std::cin.get();
    std::cout << std::setprecision(8);
    std::cout << "##################################################"
              << std::endl;
    std::cout << "FOREX ARBITRAGE" << std::endl;
    std::cout << "  arbitrage?=" << (arbitrage_possible ? "yes" : "no")
              << std::endl;
    for (const auto &c : arbitrage_opportunity.currencies)
        std::cout << " -> " << id_to_currency.at(c) << std::endl;
    double p = 1;
    for (size_t i = 0; i < arbitrage_opportunity.currencies.size(); ++i) {
        uint32_t c1id = arbitrage_opportunity.currencies.at(i);
        uint32_t c2id = arbitrage_opportunity.currencies.at(
            (i + 1) % arbitrage_opportunity.currencies.size());
        auto c1 = id_to_currency.at(c1id);
        auto c2 = id_to_currency.at(c2id);
        std::cout << "sell " << p << " units of " << c1 << "/" << c2
                  << std::endl;
        p *= exp(-graph[c1id][c2id]);
    }
    std::cout << "profit=" << arbitrage_opportunity.profit << std::endl;
    std::cout << "##################################################"
              << std::endl;

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
