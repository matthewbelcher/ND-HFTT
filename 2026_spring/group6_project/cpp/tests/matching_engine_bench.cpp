#include "matching_main.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>

using ghostbook::matching::MatchingEngine;
using ghostbook::matching::NewOrderCommand;
using ghostbook::matching::OrderType;
using ghostbook::matching::Side;
using ghostbook::matching::TimeInForce;

namespace {

using clock = std::chrono::steady_clock;

template <typename Fn>
std::chrono::nanoseconds measure(Fn&& fn) {
	auto start = clock::now();
	fn();
	auto end = clock::now();
	return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
}

void run_passive_insert_benchmark(std::size_t iterations) {
	MatchingEngine engine;
	for (std::size_t i = 0; i < iterations; ++i) {
		engine.submit(NewOrderCommand{
		    .client_order_id = static_cast<std::uint64_t>(i + 1),
		    .instrument_id = 1,
		    .side = (i % 2 == 0) ? Side::Buy : Side::Sell,
		    .order_type = OrderType::Limit,
		    .tif = TimeInForce::Day,
		    .price_tick = (i % 2 == 0) ? 100 : 101,
		    .quantity = 1,
		});
	}
	(void)engine.drain_events();
}

void run_cancel_benchmark(std::size_t iterations) {
	MatchingEngine engine;
	for (std::size_t i = 0; i < iterations; ++i) {
		engine.submit(NewOrderCommand{
		    .client_order_id = static_cast<std::uint64_t>(10'000 + i),
		    .instrument_id = 2,
		    .side = Side::Buy,
		    .order_type = OrderType::Limit,
		    .tif = TimeInForce::Day,
		    .price_tick = 100 + static_cast<ghostbook::matching::price_tick_t>(i % 4),
		    .quantity = 1,
		});
	}
	(void)engine.drain_events();

	for (std::size_t i = 0; i < iterations; ++i) {
		engine.cancel({
		    .client_order_id = static_cast<std::uint64_t>(20'000 + i),
		    .target_order_id = static_cast<std::uint64_t>(10'000 + i),
		    .cancel_quantity = 0,
		});
	}
	(void)engine.drain_events();
}

void run_sweep_benchmark(std::size_t iterations) {
	MatchingEngine engine;
	for (std::size_t i = 0; i < iterations; ++i) {
		engine.submit(NewOrderCommand{
		    .client_order_id = static_cast<std::uint64_t>(30'000 + i),
		    .instrument_id = 3,
		    .side = Side::Sell,
		    .order_type = OrderType::Limit,
		    .tif = TimeInForce::Day,
		    .price_tick = 100 + static_cast<ghostbook::matching::price_tick_t>(i % 8),
		    .quantity = 1,
		});
	}
	(void)engine.drain_events();

	engine.submit(NewOrderCommand{
	    .client_order_id = 40'000,
	    .instrument_id = 3,
	    .side = Side::Buy,
	    .order_type = OrderType::Market,
	    .tif = TimeInForce::IOC,
	    .price_tick = 0,
	    .quantity = static_cast<ghostbook::matching::quantity_t>(iterations),
	});
	(void)engine.drain_events();
}

void print_result(const std::string& name, std::size_t iterations, std::chrono::nanoseconds elapsed) {
	const double per_op = static_cast<double>(elapsed.count()) / static_cast<double>(iterations);
	std::cout << name << ": " << iterations << " ops in " << elapsed.count() << " ns"
	          << " (" << per_op << " ns/op)" << std::endl;
}

}  // namespace

int main() {
	constexpr std::size_t iterations = 50'000;
	constexpr std::size_t large_iterations = 200'000;

	print_result("passive_insert", iterations, measure([&] { run_passive_insert_benchmark(iterations); }));
	print_result("cancel", iterations, measure([&] { run_cancel_benchmark(iterations); }));
	print_result("sweep", iterations, measure([&] { run_sweep_benchmark(iterations); }));

	print_result("passive_insert_large", large_iterations,
	             measure([&] { run_passive_insert_benchmark(large_iterations); }));
	print_result("cancel_large", large_iterations, measure([&] { run_cancel_benchmark(large_iterations); }));
	print_result("sweep_large", large_iterations, measure([&] { run_sweep_benchmark(large_iterations); }));

	return 0;
}