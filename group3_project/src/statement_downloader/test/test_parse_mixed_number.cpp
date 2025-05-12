#include "catch2/catch_test_macros.hpp"
#include "statement_downloader/statement_downloader.hpp"

TEST_CASE("Basic Mixed Number", "[parse_mixed_number]") {
    char str[] = "3-1/2";
    double result = parseMixedNumber(str);
    REQUIRE(result == 3.5);
}

TEST_CASE("Mixed Number with Extra Characters", "[parse_mixed_number]") {
    char str[] = "3-1/2 cnasovneaw";
    double result = parseMixedNumber(str);
    REQUIRE(result == 3.5);
}

TEST_CASE("Mixed Number with Extra Characters at the End", "[parse_mixed_number]") {
    char str[] = "3-1/2 to 3-1/2";
    double result = parseMixedNumber(str);
    REQUIRE(result == 3.5);
}

TEST_CASE("Whole Number", "[parse_mixed_number]") {
    char str[] = "4";
    double result = parseMixedNumber(str);
    REQUIRE(result == 4.0);
}

TEST_CASE("Invalid Mixed Number", "[parse_mixed_number]") {
    char str[] = "invalid";
    double result = parseMixedNumber(str);
    REQUIRE(result == -1);
}

TEST_CASE("Empty String", "[parse_mixed_number]") {
    char str[] = "";
    double result = parseMixedNumber(str);
    REQUIRE(result == -1);
}

TEST_CASE("Whole number with chars at the beginning", "[parse_mixed_number]") {
    char str[] = "abc 4";
    double result = parseMixedNumber(str);
    REQUIRE(result == -1);
}

TEST_CASE("Fraction only", "[parse_mixed_number]") {
    char str[] = "1/2";
    double result = parseMixedNumber(str);
    REQUIRE(result == 0.5);
}

TEST_CASE("Fraction Again", "[parse_mixed_number]") {
    char str[] = "3/4";
    double result = parseMixedNumber(str);
    REQUIRE(result == 0.75);
}

TEST_CASE("Fraction with another number", "[parse_mixed_number]") {
    char str[] = "3/4 5";
    double result = parseMixedNumber(str);
    REQUIRE(result == 0.75);
}

TEST_CASE("Two Fractions", "[parse_mixed_number]") {
    char str[] = "3/4 to 5/6";
    double result = parseMixedNumber(str);
    REQUIRE(result == 0.75);
}