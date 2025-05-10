#ifndef FOREX_ARBITRAGE_H
#define FOREX_ARBITRAGE_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct arbitrage_opportunity {
    std::vector<uint32_t> currencies;
    double profit;
};

class ForexArbitrage {
  public:
    ForexArbitrage(std::vector<std::string> currencies);
    ~ForexArbitrage();

    void Update(std::string currency1, std::string currency2,
                double exchange_rate);
    bool IsArbitragePossible();
    bool FindArbitrageOpportunity(
        struct arbitrage_opportunity &arbitrage_opportunity);

    double **graph() const;
    std::unordered_map<std::string, uint32_t> currency_to_id() const;
    std::unordered_map<uint32_t, std::string> id_to_currency() const;

  private:
    uint32_t n_currencies_;
    uint32_t n_nodes_;
    double **graph_;

    double *distance_;
    uint32_t *predecessor_;
    bool *visited_;

    std::unordered_map<std::string, uint32_t> currency_to_id_;
    std::unordered_map<uint32_t, std::string> id_to_currency_;
};

#endif
