# Compiler and flags
CXX = clang++
CXXFLAGS = -Wall -Wextra -Werror=format -std=c++17 -Iinclude -I/usr/include/etherfabric
LD_LIBRARY_PATH=$(HOME)/usr/lib/x86_64-linux-gnu
LDFLAGS = -L$(LD_LIBRARY_PATH) -lciul1
NIC = enp1s0f1

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
INC_DIR = include

# Source files and objects
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
TARGET = $(BIN_DIR)/program

# Default target
all: ./$(TARGET)

gdb-run: all
	sudo gdb -ex "set environment LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)" -ex "run $(NIC)" $(TARGET)

run: all
	sudo LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$(TARGET) $(NIC)

# Link object files to create executable
$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

# Compile source files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Phony targets (targets that don't represent files)
.PHONY: all clean run
