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
TEST_BIN := $(BIN_DIR)/hft-tcpstack-test

# Sources & Objects
SRCS     := $(wildcard $(SRC_DIR)/**/*.cpp $(SRC_DIR)/*.cpp)
OBJS     := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

TEST_SRCS := $(wildcard $(TEST_DIR)/**/*.cpp $(TEST_DIR)/*.cpp)
TEST_OBJS := $(patsubst $(TEST_DIR)/%.cpp, $(OBJ_DIR)/test/%.o, $(TEST_SRCS))

# Dependency files (auto-generated)
DEPS     := $(patsubst $(OBJ_DIR)/%.o, $(DEP_DIR)/%.d, $(OBJS))
TEST_DEPS := $(patsubst $(OBJ_DIR)/test/%.o, $(DEP_DIR)/test/%.d, $(TEST_OBJS))

# Flags
CXXFLAGS := -std=c++17 \
            -Wall -Wextra -Wpedantic \
            -I$(INC_DIR) \
            -MMD -MP                   # emit .d dependency files

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
.PHONY: all clean test dirs

all: dirs $(TARGET)

# Link main binary (exclude test objects)
$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo "  LINK  $@"

# Compile src/*.cpp → build/obj/*.o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@) $(dir $(DEP_DIR)/$*.d)
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@mv -f $(OBJ_DIR)/$*.d $(DEP_DIR)/$*.d 2>/dev/null || true
	@echo "  CXX   $<"

# Compile test/*.cpp → build/obj/test/*.o
$(OBJ_DIR)/test/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@) $(dir $(DEP_DIR)/test/$*.d)
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@mv -f $(OBJ_DIR)/test/$*.d $(DEP_DIR)/test/$*.d 2>/dev/null || true
	@echo "  CXX   $<"

# Build & run tests (link all src objs except main.o + test objs)
test: dirs $(filter-out $(OBJ_DIR)/main.o, $(OBJS)) $(TEST_OBJS)
	$(CXX) $(LDFLAGS) $^ -o $(TEST_BIN) $(LIBS)
	@echo "  LINK  $(TEST_BIN)"
	./$(TEST_BIN)

# Create required directories
dirs:
	@mkdir -p $(BIN_DIR) $(OBJ_DIR) $(DEP_DIR)

clean:
	rm -rf build/ $(BIN_DIR)/

# Pull in generated dependency files so headers trigger recompilation
-include $(DEPS) $(TEST_DEPS)
