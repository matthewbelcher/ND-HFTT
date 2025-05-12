CC=clang++
CFLAGS=-Wall -Wconversion -Werror -std=c++11

SRC_DIR = src/cpp
SRC_LIB_DIR = $(SRC_DIR)/lib
SRC_BIN_DIR = $(SRC_DIR)/bin
BUILD_DIR = build
BUILD_OBJ_DIR = $(BUILD_DIR)/obj
BUILD_BIN_DIR = $(BUILD_DIR)/bin

LIB_NAME = forex
SRC_LIB_CC = $(wildcard $(SRC_LIB_DIR)/*.cc)
SRC_LIB_H = $(wildcard $(SRC_LIB_DIR)/*.h)
OBJ_FILES = $(patsubst $(SRC_LIB_DIR)/%.cc, $(BUILD_OBJ_DIR)/%.o, $(SRC_LIB_CC))

SRC_BIN_CC = $(wildcard $(SRC_BIN_DIR)/*.cc)
BIN_LIBS = -lfmt
BINARIES = $(patsubst $(SRC_BIN_DIR)/%.cc, $(BUILD_BIN_DIR)/%, $(SRC_BIN_CC))

.PHONY: all
all: $(BUILD_DIR)/lib$(LIB_NAME).a $(BINARIES)

# Archive the library.
$(BUILD_DIR)/lib$(LIB_NAME).a: $(OBJ_FILES)
	@mkdir -p $(BUILD_DIR)
	ar rcs $@ $^

# Compile the object files.
$(BUILD_OBJ_DIR)/%.o: $(SRC_LIB_DIR)/%.cc
	@mkdir -p $(BUILD_OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Build the binaries.
$(BUILD_BIN_DIR)/%: $(SRC_BIN_DIR)/%.cc $(BUILD_DIR)/lib$(LIB_NAME).a
	@mkdir -p $(BUILD_BIN_DIR)
	$(CC) $(CFLAGS) $< -o $@ -I$(SRC_LIB_DIR) -L$(BUILD_DIR) -l$(LIB_NAME) $(BIN_LIBS)

.PHONY: format
format:
	clang-format -i $(SRC_LIB_CC) $(SRC_LIB_H) $(SRC_BIN_CC)

.PHONY: clean
clean:
	rm -fr $(BUILD_DIR)
