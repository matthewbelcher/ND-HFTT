# Toolchain
CXX      := clang++
CC       := clang

# Directories
SRC_DIR   := src
INC_DIR   := include
BIN_DIR   := bin
TEST_DIR  := test
BENCH_DIR := bench
OBJ_DIR   := build/obj
DEP_DIR   := build/dep
DATA_DIR  := data

# Main binary
TARGET   := $(BIN_DIR)/hft-tcpstack

# Sources: exclude dpdk_port.cpp unless BACKEND_DPDK=1
SRCS_ALL := $(shell find $(SRC_DIR) -name "*.cpp")
ifdef BACKEND_DPDK
  SRCS := $(SRCS_ALL)
else
  SRCS := $(filter-out $(SRC_DIR)/io/dpdk_port.cpp, $(SRCS_ALL))
endif
OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

# Test sources: each file has its own main(); exclude dpdk test without DPDK
TEST_SRCS_ALL := $(shell find $(TEST_DIR) -name "test_*.cpp")
ifdef BACKEND_DPDK
  TEST_SRCS := $(TEST_SRCS_ALL)
else
  TEST_SRCS := $(filter-out $(TEST_DIR)/test_dpdk_port.cpp, $(TEST_SRCS_ALL))
endif
TEST_OBJS := $(patsubst $(TEST_DIR)/%.cpp, $(OBJ_DIR)/test/%.o, $(TEST_SRCS))
TEST_BINS := $(patsubst $(TEST_DIR)/%.cpp, $(BIN_DIR)/%, $(TEST_SRCS))

# Benchmark binary
BENCH_SRC := $(BENCH_DIR)/benchmark.cpp
BENCH_OBJ := $(OBJ_DIR)/bench/benchmark.o
BENCH_BIN := $(BIN_DIR)/benchmark

# Kernel TCP comparison benchmark (standalone, no project libs needed)
BENCH_KERN_SRC := $(BENCH_DIR)/bench_kernel_tcp.cpp
BENCH_KERN_OBJ := $(OBJ_DIR)/bench/bench_kernel_tcp.o
BENCH_KERN_BIN := $(BIN_DIR)/bench_kernel_tcp

# Src objects without main.o (used when linking test and bench binaries)
SRC_OBJS_NO_MAIN := $(filter-out $(OBJ_DIR)/main.o, $(OBJS))

# Dependency files
DEPS           := $(patsubst $(OBJ_DIR)/%.o, $(DEP_DIR)/%.d, $(OBJS))
TEST_DEPS      := $(patsubst $(OBJ_DIR)/test/%.o, $(DEP_DIR)/test/%.d, $(TEST_OBJS))
BENCH_DEPS     := $(DEP_DIR)/bench/benchmark.d
BENCH_KERN_DEP := $(DEP_DIR)/bench/bench_kernel_tcp.d

# Flags
CXXFLAGS := -std=c++17 \
            -Wall -Wextra -Wpedantic \
            -I$(INC_DIR)

LDFLAGS  :=
LIBS     :=

# DPDK backend
ifdef BACKEND_DPDK
  CXXFLAGS += $(shell pkg-config --cflags libdpdk) -DBACKEND_DPDK
  LIBS     += $(shell pkg-config --libs libdpdk)
endif

# Debug / Release modes: make DEBUG=1 or make RELEASE=1
ifdef DEBUG
  CXXFLAGS += -O0 -g3 -fsanitize=address,undefined -DDEBUG
  LDFLAGS  += -fsanitize=address,undefined
else ifdef RELEASE
  CXXFLAGS += -O3 -march=native -flto -DNDEBUG
  LDFLAGS  += -flto
else
  CXXFLAGS += -O2 -g
endif

# Rules
.PHONY: all clean test bench bench-kernel bench-compare capture dirs

all: dirs $(TARGET)

# Link main binary
$(TARGET): $(OBJS)
	@echo "  LINK  $@"
	@$(CXX) $(LDFLAGS) $^ -o $@ $(LIBS)

# Compile src/**/*.cpp
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@) $(dir $(DEP_DIR)/$*.d)
	@echo "  CXX   $<"
	@$(CXX) $(CXXFLAGS) -MT $@ -MMD -MP -MF $(DEP_DIR)/$*.d -c $< -o $@

# Compile test/test_*.cpp
$(OBJ_DIR)/test/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@) $(dir $(DEP_DIR)/test/$*.d)
	@echo "  CXX   $<"
	@$(CXX) $(CXXFLAGS) -MT $@ -MMD -MP -MF $(DEP_DIR)/test/$*.d -c $< -o $@

# Compile bench/benchmark.cpp
$(BENCH_OBJ): $(BENCH_SRC)
	@mkdir -p $(dir $@) $(dir $(BENCH_DEPS))
	@echo "  CXX   $<"
	@$(CXX) $(CXXFLAGS) -MT $@ -MMD -MP -MF $(BENCH_DEPS) -c $< -o $@

# Compile bench/bench_kernel_tcp.cpp (no project includes needed)
$(BENCH_KERN_OBJ): $(BENCH_KERN_SRC)
	@mkdir -p $(dir $@) $(dir $(BENCH_KERN_DEP))
	@echo "  CXX   $<"
	@$(CXX) $(CXXFLAGS) -MT $@ -MMD -MP -MF $(BENCH_KERN_DEP) -c $< -o $@

# Link each test binary
$(BIN_DIR)/test_%: $(OBJ_DIR)/test/test_%.o $(SRC_OBJS_NO_MAIN)
	@mkdir -p $(BIN_DIR)
	@echo "  LINK  $@"
	@$(CXX) $(LDFLAGS) $^ -o $@ $(LIBS)

# Link benchmark binary
$(BENCH_BIN): $(BENCH_OBJ) $(SRC_OBJS_NO_MAIN)
	@mkdir -p $(BIN_DIR)
	@echo "  LINK  $@"
	@$(CXX) $(LDFLAGS) $^ -o $@ $(LIBS)

# Link kernel TCP benchmark (only needs pthreads from the system)
$(BENCH_KERN_BIN): $(BENCH_KERN_OBJ)
	@mkdir -p $(BIN_DIR)
	@echo "  LINK  $@"
	@$(CXX) $(LDFLAGS) $^ -o $@ -lpthread

# Run all test binaries
test: dirs all $(TEST_BINS)
	@echo ""
	@total_pass=0; total_fail=0; \
	for t in $(TEST_BINS); do \
		echo "--- $$t ---"; \
		./$$t; \
		rc=$$?; \
		if [ $$rc -ne 0 ]; then total_fail=$$((total_fail+1)); \
		else total_pass=$$((total_pass+1)); fi; \
		echo ""; \
	done; \
	echo "=== $$total_pass suite(s) passed, $$total_fail suite(s) failed ==="

# Per-phase convenience targets
test-phase1: dirs all $(BIN_DIR)/test_raw_socket
	@./$(BIN_DIR)/test_raw_socket

test-phase2: dirs all $(BIN_DIR)/test_checksum
	@./$(BIN_DIR)/test_checksum

test-phase3: dirs all $(BIN_DIR)/test_retransmit $(BIN_DIR)/test_tcp_state
	@echo "--- test_retransmit ---"
	@./$(BIN_DIR)/test_retransmit
	@echo ""
	@echo "--- test_tcp_state ---"
	@./$(BIN_DIR)/test_tcp_state

test-phase4: dirs all $(BIN_DIR)/test_message
	@./$(BIN_DIR)/test_message

ifdef BACKEND_DPDK
test-phase5: dirs all $(BIN_DIR)/test_dpdk_port
	@./$(BIN_DIR)/test_dpdk_port
else
test-phase5:
	@echo "Build with BACKEND_DPDK=1 to run Phase 5 tests"
	@echo "  make BACKEND_DPDK=1 test-phase5"
endif

# Benchmark: in-process stack latency (always) + DPDK I/O latency (BACKEND_DPDK=1)
bench: dirs all $(BENCH_BIN)
	@./$(BENCH_BIN) | tee $(DATA_DIR)/benchmark_results.txt

# Kernel TCP loopback baseline benchmark
bench-kernel: dirs $(BENCH_KERN_BIN)
	@./$(BENCH_KERN_BIN) | tee $(DATA_DIR)/kernel_tcp_results.txt

# Side-by-side comparison: run both benchmarks and print results together
bench-compare: dirs all $(BENCH_BIN) $(BENCH_KERN_BIN)
	@echo "=========================================="
	@echo " USERSPACE STACK (in-process, no syscalls)"
	@echo "=========================================="
	@./$(BENCH_BIN) | tee $(DATA_DIR)/benchmark_results.txt
	@echo "=========================================="
	@echo " KERNEL TCP BASELINE (loopback, 4 syscalls per RTT)"
	@echo "=========================================="
	@./$(BENCH_KERN_BIN) | tee $(DATA_DIR)/kernel_tcp_results.txt

# Phase 2 wire validation
capture: dirs $(TARGET)
	@sudo ./scripts/capture_syn.sh

# Create required directories
dirs:
	@mkdir -p $(BIN_DIR) $(OBJ_DIR) $(DEP_DIR) $(OBJ_DIR)/bench $(DEP_DIR)/bench $(DATA_DIR)

clean:
	rm -rf build/ $(BIN_DIR)/

# Pull in generated dependency files
-include $(DEPS) $(TEST_DEPS) $(BENCH_DEPS) $(BENCH_KERN_DEP)
