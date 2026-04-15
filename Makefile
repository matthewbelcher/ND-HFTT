# Toolchain
CXX      := clang++
CC       := clang

# Directories
SRC_DIR  := src
INC_DIR  := include
LIB_DIR  := lib
BIN_DIR  := bin
TEST_DIR := test
OBJ_DIR  := build/obj
DEP_DIR  := build/dep

# Target
TARGET   := $(BIN_DIR)/hft-tcpstack

# Sources & Objects
SRCS     := $(shell find $(SRC_DIR) -name "*.cpp")
OBJS     := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

TEST_SRCS := $(shell find $(TEST_DIR) -name "*.cpp")
TEST_OBJS := $(patsubst $(TEST_DIR)/%.cpp, $(OBJ_DIR)/test/%.o, $(TEST_SRCS))
TEST_BINS := $(patsubst $(TEST_DIR)/%.cpp, $(BIN_DIR)/%, $(TEST_SRCS))

# Dependency files (auto-generated)
DEPS     := $(patsubst $(OBJ_DIR)/%.o, $(DEP_DIR)/%.d, $(OBJS))
TEST_DEPS := $(patsubst $(OBJ_DIR)/test/%.o, $(DEP_DIR)/test/%.d, $(TEST_OBJS))

# Shared src objects (no main.o) linked into every test binary
SRC_LINK_OBJS := $(filter-out $(OBJ_DIR)/main.o, $(OBJS))

# Flags
CXXFLAGS := -std=c++17 \
            -Wall -Wextra -Wpedantic \
            -I$(INC_DIR)

LDFLAGS  :=
LIBS     :=

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
.PHONY: all clean test capture dirs

all: dirs $(TARGET)

# Link main binary (exclude test objects)
$(TARGET): $(OBJS)
	@echo "  LINK  $@"
	@$(CXX) $(LDFLAGS) $^ -o $@ $(LIBS)

# Compile src/*.cpp → build/obj/*.o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@) $(dir $(DEP_DIR)/$*.d)
	@echo "  CXX   $<"
	@$(CXX) $(CXXFLAGS) -MT $@ -MMD -MP -MF $(DEP_DIR)/$*.d -c $< -o $@

# Compile test/*.cpp → build/obj/test/*.o
$(OBJ_DIR)/test/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@) $(dir $(DEP_DIR)/test/$*.d)
	@echo "  CXX   $<"
	@$(CXX) $(CXXFLAGS) -MT $@ -MMD -MP -MF $(DEP_DIR)/test/$*.d -c $< -o $@

# One binary per test file: bin/test_foo from test/test_foo.cpp
$(BIN_DIR)/%: $(OBJ_DIR)/test/%.o $(SRC_LINK_OBJS)
	@echo "  LINK  $@"
	@$(CXX) $(LDFLAGS) $^ -o $@ $(LIBS)

# Build & run every test binary in sequence
test: dirs $(TEST_BINS)
	@for t in $(TEST_BINS); do \
		echo ""; \
		echo "--- $$t ---"; \
		./$$t; \
	done

# Phase 2 wire validation: build main binary then run tcpdump capture script
capture: dirs $(TARGET)
	@sudo ./scripts/capture_syn.sh

# Create required directories
dirs:
	@mkdir -p $(BIN_DIR) $(OBJ_DIR) $(DEP_DIR)

clean:
	rm -rf build/ $(BIN_DIR)/

# Pull in generated dependency files so headers trigger recompilation
-include $(DEPS) $(TEST_DEPS)
