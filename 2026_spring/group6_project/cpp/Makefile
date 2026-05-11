CMAKE ?= cmake
CTEST ?= ctest
BUILD_DIR ?= build
BUILD_TYPE ?= Debug

.PHONY: all configure build test bench run clean

all: build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: configure
	$(CMAKE) --build $(BUILD_DIR) --config $(BUILD_TYPE)

test: build
	$(CTEST) --test-dir $(BUILD_DIR)/tests/ --build-config $(BUILD_TYPE) --output-on-failure

bench: build
	$(CMAKE) --build $(BUILD_DIR) --config $(BUILD_TYPE) --target matching_engine_bench
	./$(BUILD_DIR)/src/matching_engine_bench

run: build
	./$(BUILD_DIR)/src/matching_engine

clean:
	$(CMAKE) -E rm -rf $(BUILD_DIR)
