#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

#include "catch2/catch_test_macros.hpp"
#include "statement_downloader/statement_downloader.hpp"

std::vector<std::vector<std::string>> readCSV(const std::string& filename) {
    std::vector<std::vector<std::string>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> row;
        while (std::getline(lineStream, cell, ',')) {
            row.push_back(cell);
        }
        data.push_back(row);
    }
    file.close();
    return data;
}

TEST_CASE("Test Statement Parsing", "[statement_downloader]") {
    auto fed_rates = readCSV("/home/zbrown2/fed-market-impact/src/data/target_rates/target_rates.csv");

    for (const auto& row : std::vector<std::vector<std::string>>(fed_rates.begin() + 20, fed_rates.end())) {
        if (row.size() < 3) {
            continue; // Skip rows that don't have enough data
        }
        std::string date = row[0];
        std::string lower_bound = row[1];
        std::string upper_bound = row[2];

        printf("Testing date: %s", date.c_str());

        std::string url = "https://www.federalreserve.gov/newsevents/pressreleases/monetary" + date + "a.htm";
        StatementDownloader downloader = StatementDownloader(url, std::chrono::system_clock::now() + std::chrono::seconds(30), 80000);
    
        int result = downloader.download();
        REQUIRE(result == 0); 

        result = downloader.parse();
        REQUIRE(result == 0);

        auto bounds = downloader.getRateBounds();
        REQUIRE(bounds.first == std::stod(lower_bound));
        REQUIRE(bounds.second == std::stod(upper_bound));

        downloader.reset();
    }
}