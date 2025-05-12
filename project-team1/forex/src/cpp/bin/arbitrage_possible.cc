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
    if (argc != 2 && argc != 4) {
        std::cerr
            << "Usage: " << argv[0]
            << " <path to Polygon.io CSV forex data> [-s <path to output csv>]"
            << std::endl;
        std::cerr
            << "    -s: prints each arbitrage opportunity to specified csv"
            << std::endl;
        return 1;
    }

    std::ifstream csv(argv[1]);
    std::string _;
    std::getline(csv, _);

    unsigned long int n_arbitrages = 0;
    std::ofstream output;
    bool summary = false;
    if (argc == 4 && strcmp(argv[2], "-s") == 0) {
        summary = true;
        output.open(argv[3]);
        output << std::fixed << std::setprecision(16);
        output << "Profit,Cycle,Timestamp" << std::endl;
    }

    ForexArbitrage forex_arbitrage(CURRENCIES);
    const auto &id_to_currency = forex_arbitrage.id_to_currency();
    struct arbitrage_opportunity arbitrage_opportunity;

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
            if (forex_arbitrage.FindArbitrageOpportunity(
                    arbitrage_opportunity)) {
                if (summary) {
                    output << arbitrage_opportunity.profit << ",";
                    for (auto &currency : arbitrage_opportunity.currencies)
                        output << " " << id_to_currency.at(currency);
                    output << "," << timestamp << std::endl;
                    n_arbitrages++;
                } else {
                    std::cout << arbitrage_opportunity.profit;
                    for (auto &currency : arbitrage_opportunity.currencies)
                        std::cout << " " << id_to_currency.at(currency);
                    output << "," << timestamp << std::endl;
                }
            }
        }
        forex_arbitrage.Update(base_currency, quote_currency, bid_price);
    }
    forex_arbitrage.FindArbitrageOpportunity(arbitrage_opportunity);

    if (summary) {
        std::cout << n_arbitrages << " total arbitrage opportunities found."
                  << std::endl;
        output.close();
    }

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
