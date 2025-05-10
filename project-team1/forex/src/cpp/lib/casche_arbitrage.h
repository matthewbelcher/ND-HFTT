#ifndef CASCHE_ARBITRAGE_H
#define CASCHE_ARBITRAGE_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct arbitrage_table_entry {
    uint32_t pair;
    double threshold;
    uint32_t cycle;
};

class CascheArbitrage {
  public:
    // CascheArbitrage accepts a list of currency pair cycles to track. The
    // object tracks the profitability of these cycles such that it can quickly
    // determine if an arbitrage opportunity arises given new market data.
    // Further, it can output a table of currency pairs, rate thresholds, and
    // cycles such that if the exchange rate for a currency pair exceeds a
    // threshold, the corresponding cycle should be traded immediately. This
    // paradigm pairs well with an FPGA.
    CascheArbitrage(std::vector<std::string> string_cycles);

    // Destructor.
    ~CascheArbitrage();

    // Update the state to reflect a new exchange rate for a currency pair.
    void Update(uint32_t pair, double rate);
    void Update(std::string string_pair, double rate);

    // Determine if an arbitrage opportunity arises given a new exchange rate
    // for a currency pair. This will always return false if the previous
    // exchange rate is NAN.
    bool NewArbitrage(uint32_t pair, double rate);
    bool NewArbitrage(std::string string_pair, double rate);
    bool NewArbitrage(uint32_t pair, double rate, uint32_t &cycle);
    bool NewArbitrage(std::string string_pair, double rate, uint32_t &cycle);

    // Find all cycles that present an arbitrage opportunity.
    std::vector<uint32_t> FindAllArbitrageCycles();

    // Generate a table of currency pairs, thresholds, and cycles. If a new
    // market data message has an exchange rate that exceeds the threshold for
    // some currency pair, then the cycle should be traded immediately. This
    // pairs well with an FPGA. Currently, this function outputs a table for
    // all currency pairs with the corresponding most profitable cycle.
    std::vector<struct arbitrage_table_entry> GenerateTable();

    const std::unordered_map<std::string, uint32_t> &string_to_cycle() const;
    const std::vector<std::string> &cycle_to_string() const;
    const std::unordered_map<std::string, uint32_t> &string_to_pair() const;
    const std::vector<std::string> &pair_to_string() const;
    const std::vector<std::vector<uint32_t>> &pair_to_cycles() const;
    const std::vector<std::vector<uint32_t>> &cycle_to_pairs() const;
    const std::vector<double> &pair_to_rate() const;
    const std::vector<double> &cycle_to_profit() const;

  private:
    std::vector<std::string> StringCycleToStringPairs(std::string string_cycle);

    // The number of cycles to track.
    uint32_t n_cycles_;
    // The number of pairs within tracked cycles.
    uint32_t n_pairs_;

    std::unordered_map<std::string, uint32_t> string_to_cycle_;
    std::vector<std::string> cycle_to_string_;

    std::unordered_map<std::string, uint32_t> string_to_pair_;
    std::vector<std::string> pair_to_string_;

    std::vector<std::vector<uint32_t>> pair_to_cycles_;
    std::vector<std::vector<uint32_t>> cycle_to_pairs_;

    std::vector<double> pair_to_rate_;
    std::vector<double> cycle_to_profit_;
};

#endif
