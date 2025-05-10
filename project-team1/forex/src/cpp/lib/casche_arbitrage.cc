#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "casche_arbitrage.h"

CascheArbitrage::CascheArbitrage(std::vector<std::string> string_cycles) {
    // Bijectively map the string pairs to numeric IDs in the range [0,
    // n_pairs). The pairs to track are all pairs present in the cycles.
    uint32_t pair = 0;
    for (const auto &string_cycle : string_cycles) {
        auto string_pairs = StringCycleToStringPairs(string_cycle);
        for (const auto &string_pair : string_pairs) {
            if (string_to_pair_.count(string_pair) == 0) {
                string_to_pair_[string_pair] = pair;
                pair_to_string_.push_back(string_pair);
                ++pair;
            }
        }
    }

    // Bijectively map the string cycles to numeric IDs in the range [0,
    // n_cycles).
    uint32_t cycle = 0;
    for (const auto &string_cycle : string_cycles) {
        string_to_cycle_[string_cycle] = cycle;
        cycle_to_string_.push_back(string_cycle);
        ++cycle;
    }

    // Set the number of pairs and number of cycles. It is reasonable to limit
    // these to 32-bit unsigned integers because 4 billion pairs/cycles seems
    // excessive.
    n_pairs_ = pair;
    n_cycles_ = cycle;

    // Construct two maps: pair-to-cycles and cycle-to-pairs. The
    // pair-to-cycles map tracks to which cycles a given pair belongs. The
    // cycle-to-pairs map tracks the ordered sequence of pairs that belong to a
    // given cycle.
    for (uint32_t _ = 0; _ < n_pairs_; ++_)
        pair_to_cycles_.push_back({});
    for (uint32_t _ = 0; _ < n_cycles_; ++_)
        cycle_to_pairs_.push_back({});
    for (const auto &string_cycle : string_cycles) {
        uint32_t cycle = string_to_cycle_[string_cycle];
        auto string_pairs = StringCycleToStringPairs(string_cycle);
        for (const auto &string_pair : string_pairs) {
            uint32_t pair = string_to_pair_[string_pair];
            pair_to_cycles_[pair].push_back(cycle);
            cycle_to_pairs_[cycle].push_back(pair);
        }
    }

    // Initialize NAN exchange rate for every pair.
    for (uint32_t _ = 0; _ < n_pairs_; ++_)
        pair_to_rate_.push_back(NAN);

    // Initialize NAN profit for every cycle.
    for (uint32_t _ = 0; _ < n_cycles_; ++_)
        cycle_to_profit_.push_back(NAN);
}

CascheArbitrage::~CascheArbitrage() {
}

std::vector<std::string>
CascheArbitrage::StringCycleToStringPairs(std::string string_cycle) {
    std::vector<std::string> currencies;
    std::stringstream ss(string_cycle);
    std::string currency;

    while (std::getline(ss, currency, '/'))
        currencies.push_back(currency);

    std::vector<std::string> string_pairs;
    for (size_t i = 0; i + 1 < currencies.size(); ++i)
        string_pairs.push_back(currencies[i] + "/" + currencies[i + 1]);

    return string_pairs;
}

const std::unordered_map<std::string, uint32_t> &
CascheArbitrage::string_to_cycle() const {
    return string_to_cycle_;
}

const std::vector<std::string> &CascheArbitrage::cycle_to_string() const {
    return cycle_to_string_;
}

const std::unordered_map<std::string, uint32_t> &
CascheArbitrage::string_to_pair() const {
    return string_to_pair_;
}

const std::vector<std::string> &CascheArbitrage::pair_to_string() const {
    return pair_to_string_;
}

const std::vector<std::vector<uint32_t>> &
CascheArbitrage::pair_to_cycles() const {
    return pair_to_cycles_;
}

const std::vector<std::vector<uint32_t>> &
CascheArbitrage::cycle_to_pairs() const {
    return cycle_to_pairs_;
}

const std::vector<double> &CascheArbitrage::pair_to_rate() const {
    return pair_to_rate_;
}

const std::vector<double> &CascheArbitrage::cycle_to_profit() const {
    return cycle_to_profit_;
}

void CascheArbitrage::Update(uint32_t pair, double rate) {
    // Update the exchange rate for the pair.
    pair_to_rate_[pair] = rate;

    // For each cycle to which the pair belongs, update the cycle profit. This
    // involves multiplying the exchange rates of all pairs in each cycle. The
    // profit must be NAN if any exchange rate is NAN.
    for (uint32_t cycle : pair_to_cycles_[pair]) {
        double profit = 1;
        for (uint32_t pair : cycle_to_pairs_[cycle]) {
            double rate = pair_to_rate_[pair];
            if (isnan(rate)) {
                profit = NAN;
                break;
            }
            profit *= rate;
        }
        cycle_to_profit_[cycle] = profit;
    }
}

void CascheArbitrage::Update(std::string string_pair, double rate) {
    uint32_t pair = string_to_pair_[string_pair];
    Update(pair, rate);
}

bool CascheArbitrage::NewArbitrage(uint32_t pair, double rate) {
    double old_rate = pair_to_rate_[pair];
    if (isnan(old_rate))
        return false;

    // If a cycle has profit p, then an arbitrage opportunity exists iff p > 1.
    // If we update a currency pair's exchange rate in the cycle, then the
    // cyclesing profit equals p * new_rate / old_rate.
    double new_old_ratio = rate / old_rate;

    for (uint32_t cycle : pair_to_cycles_[pair]) {
        double profit = cycle_to_profit_[cycle];
        if (isnan(profit))
            continue;
        if (profit * new_old_ratio > 1)
            return true;
    }

    return false;
}

bool CascheArbitrage::NewArbitrage(std::string string_pair, double rate) {
    uint32_t pair = string_to_pair_[string_pair];
    return NewArbitrage(pair, rate);
}

bool CascheArbitrage::NewArbitrage(uint32_t pair, double rate,
                                   uint32_t &cycle) {
    double old_rate = pair_to_rate_[pair];
    if (isnan(old_rate))
        return false;

    // If a cycle has profit p, then an arbitrage opportunity exists iff p > 1.
    // If we update a currency pair's exchange rate in the cycle, then the
    // cyclesing profit equals p * new_rate / old_rate.
    double new_old_ratio = rate / old_rate;

    for (uint32_t cycle_ : pair_to_cycles_[pair]) {
        double profit = cycle_to_profit_[cycle_];
        if (isnan(profit))
            continue;
        if (profit * new_old_ratio > 1) {
            cycle = cycle_;
            return true;
        }
    }

    return false;
}

bool CascheArbitrage::NewArbitrage(std::string string_pair, double rate,
                                   uint32_t &cycle) {
    uint32_t pair = string_to_pair_[string_pair];
    return NewArbitrage(pair, rate, cycle);
}

std::vector<uint32_t> CascheArbitrage::FindAllArbitrageCycles() {
    std::vector<uint32_t> arbitrage_cycles;
    for (uint32_t cycle = 0; cycle < n_cycles_; ++cycle) {
        double profit = cycle_to_profit_[cycle];
        if (isnan(profit))
            continue;
        if (profit > 1)
            arbitrage_cycles.push_back(cycle);
    }
    return arbitrage_cycles;
}


std::vector<struct arbitrage_table_entry> CascheArbitrage::GenerateTable() {
    std::vector<struct arbitrage_table_entry> result;

    // For each pair, compute the minimum threshold the rate must exceed to
    // have an arbitrage opportunity. Always favor cycles that are closest to
    // being profitable.
    for (uint32_t pair = 0; pair < n_pairs_; ++pair) {
        uint32_t cycle_max = pair_to_cycles_[pair][0];
        double profit_max = cycle_to_profit_[cycle_max];
        for (uint32_t cycle : pair_to_cycles_[pair]) {
            double profit = cycle_to_profit_[cycle];
            if (profit > profit_max || isnan(profit_max))
                cycle_max = cycle;
                profit_max = profit;
        }
        if (isnan(profit_max))
            continue;
        double rate = pair_to_rate_[pair];
        double threshold = rate / profit_max;
        result.push_back((struct arbitrage_table_entry){pair, threshold, cycle_max});
    }

    return result;
}
