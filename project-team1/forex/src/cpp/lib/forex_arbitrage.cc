#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "forex_arbitrage.h"

ForexArbitrage::ForexArbitrage(std::vector<std::string> currencies) {
    n_currencies_ = static_cast<uint32_t>(currencies.size());
    n_nodes_ = n_currencies_ + 1;

    // Instantiate the graph such that graph_[currency1][currency2] is the
    // negative logarithm of the exchange rate between currency1 and currency2.
    graph_ = new double *[n_nodes_];

    // Include a sentinel node in the graph with a weight 0 edge to every other
    // node. We do this because the Bellman-Ford algorithm finds the shortest
    // distance from a source node to every other connected node, but we only
    // care about finding a negative cycle, so the source node should have
    // access to all other nodes.
    graph_[0] = new double[n_nodes_];
    std::fill_n(graph_[0], n_nodes_, 0);

    // Set all edge weights to NAN (no edge) and track the currency ID maps.
    for (uint32_t i = 0; i < n_currencies_; ++i) {
        graph_[i + 1] = new double[n_nodes_];
        std::fill_n(graph_[i + 1], n_nodes_, NAN);

        currency_to_id_[currencies.at(i)] = i + 1;
        id_to_currency_[i + 1] = currencies.at(i);
    }

    // These data structures are used by the Bellman-Ford algorithm.
    distance_ = new double[n_nodes_];
    predecessor_ = new uint32_t[n_nodes_];
    visited_ = new bool[n_nodes_];
}

ForexArbitrage::~ForexArbitrage() {
    for (uint32_t i = 0; i < n_currencies_; ++i)
        delete[] graph_[i];
    delete[] graph_;
}

void ForexArbitrage::Update(std::string currency1, std::string currency2,
                            double exchange_rate) {
    uint32_t currency1_id = currency_to_id_.at(currency1);
    uint32_t currency2_id = currency_to_id_.at(currency2);

    // The weight of the edge equals the negative logarithm of the exchange
    // rate. An arbitrage opportunity exists if and only if the product of the
    // exchange rates is greater than 1. We need e1 * e2 * ... * en > 1. This is
    // equivalent to 1/e1 * 1/e2 * ... * 1/en < 1. This is equivalent to
    // -log(e1) + -log(e2) + ... + -log(en) < 0. Thus, the problem is reduced to
    // finding a negative cycle in a directed, weighted graph.
    double weight = -log(exchange_rate);

    graph_[currency1_id][currency2_id] = weight;
}

bool ForexArbitrage::IsArbitragePossible() {
    // The distance to the source node equals 0. All other distances start at
    // infinity.
    distance_[0] = 0;
    std::fill_n(distance_ + 1, n_currencies_, INFINITY);

    // Relax the edges n_nodes_ - 1 = n_currencies_ times. This ensures that all
    // edges are completely relaxed (unless a negative cycle exists).
    double weight;
    bool relaxed;
    for (uint32_t _ = 0; _ < n_currencies_; ++_) {
        relaxed = false;
        for (uint32_t i = 0; i < n_nodes_; ++i) {
            for (uint32_t j = 0; j < n_nodes_; ++j) {
                if (i == j)
                    continue;

                weight = graph_[i][j];
                if (isnan(weight))
                    continue;

                if (distance_[i] + weight < distance_[j]) {
                    distance_[j] = distance_[i] + weight;
                    relaxed = true;
                }
            }
        }

        // If no edges are relaxed, then there must not be a negative cycle,
        // meaning there is no arbitrage opportunity.
        if (!relaxed)
            return false;
    }

    // Try to relax the edges one more time. If any edge relaxes, then there
    // must be a negative cycle.
    for (uint32_t i = 0; i < n_nodes_; ++i) {
        for (uint32_t j = 0; j < n_nodes_; ++j) {
            if (i == j)
                continue;

            weight = graph_[i][j];
            if (isnan(weight))
                continue;

            if (distance_[i] + weight < distance_[j])
                return true;
        }
    }

    return false;
}

bool ForexArbitrage::FindArbitrageOpportunity(
    struct arbitrage_opportunity &arbitrage_opportunity) {
    // The distance to the source node equals 0. All other distances start at
    // infinity. In the beginning, there are no predecessors, and it is not
    // necessary to initialize the predecessors at all. They will be filled when
    // relaxing the edges. The visited array is initialized to no nodes visited.
    distance_[0] = 0;
    std::fill_n(distance_ + 1, n_currencies_, INFINITY);
    std::fill_n(visited_ + 1, n_currencies_, false);

    // Relax the edges n_nodes_ - 1 = n_currencies_ times. This ensures that all
    // edges are completely relaxed (unless a negative cycle exists).
    double weight;
    bool relaxed;
    for (uint32_t _ = 0; _ < n_currencies_; ++_) {
        relaxed = false;
        for (uint32_t i = 0; i < n_nodes_; ++i) {
            for (uint32_t j = 0; j < n_nodes_; ++j) {
                if (i == j)
                    continue;

                weight = graph_[i][j];
                if (isnan(weight))
                    continue;

                if (distance_[i] + weight < distance_[j]) {
                    distance_[j] = distance_[i] + weight;
                    predecessor_[j] = i;
                    relaxed = true;
                }
            }
        }

        // If no edges are relaxed, then there must not be a negative cycle,
        // meaning there is no arbitrage opportunity.
        if (!relaxed)
            return false;
    }

    // Try to relax the edges one more time. If any edge relaxes, then there
    // must be a negative cycle.
    for (uint32_t i = 0; i < n_nodes_; ++i) {
        for (uint32_t j = 0; j < n_nodes_; ++j) {
            if (i == j)
                continue;

            weight = graph_[i][j];
            if (isnan(weight))
                continue;

            if (distance_[i] + weight < distance_[j]) {
                predecessor_[j] = i;

                // Although node j is reachable from the negative cycle, it is
                // not necessarily part of the cycle. We can use the
                // predecessors structure to find the cycle.
                while (!visited_[j]) {
                    visited_[j] = true;
                    j = predecessor_[j];
                }

                // Reconstruct the negative cycle and calculate the profit.
                arbitrage_opportunity.currencies = {j};
                arbitrage_opportunity.profit = 0;
                i = predecessor_[j];
                while (i != j) {
                    arbitrage_opportunity.currencies.insert(
                        arbitrage_opportunity.currencies.begin(), i);
                    arbitrage_opportunity.profit +=
                        graph_[arbitrage_opportunity.currencies[0]]
                              [arbitrage_opportunity.currencies[1]];
                    i = predecessor_[i];
                }
                arbitrage_opportunity.profit +=
                    graph_[arbitrage_opportunity.currencies.back()]
                          [arbitrage_opportunity.currencies.front()];
                arbitrage_opportunity.profit =
                    exp(-arbitrage_opportunity.profit);

                return true;
            }
        }
    }

    return false;
}

double **ForexArbitrage::graph() const {
    return graph_;
}

std::unordered_map<std::string, uint32_t>
ForexArbitrage::currency_to_id() const {
    return currency_to_id_;
}

std::unordered_map<uint32_t, std::string>
ForexArbitrage::id_to_currency() const {
    return id_to_currency_;
}
