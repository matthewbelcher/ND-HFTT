#include <string>
#include <vector>

#include <fmt/core.h>

#include "casche_arbitrage.h"

std::vector<std::string> CYCLES = {"USD/GBP/USD"};

int main(int argc, char *argv[]) {
    bool arbitrage;
    CascheArbitrage casche_arbitrage(CYCLES);

    // No arbitrage opportunity exists.
    casche_arbitrage.Update("USD/GBP", 0.75);
    casche_arbitrage.Update("GBP/USD", 1.32);

    // These updates do not create an arbitrage opportunity.
    arbitrage = casche_arbitrage.NewArbitrage("USD/GBP", 0.755);
    fmt::print("arbitrage={}\n", arbitrage);
    casche_arbitrage.NewArbitrage("GBP/USD", 1.33);
    fmt::print("arbitrage={}\n", arbitrage);

    // These update do create an arbitrage opportunity.
    arbitrage = casche_arbitrage.NewArbitrage("USD/GBP", 0.76);
    fmt::print("arbitrage={}\n", arbitrage);
    casche_arbitrage.NewArbitrage("GBP/USD", 1.34);
    fmt::print("arbitrage={}\n", arbitrage);

    return 0;
}
