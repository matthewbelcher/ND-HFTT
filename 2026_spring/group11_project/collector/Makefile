# ─────────────────────────────────────────────────────────────────
#  Kalshi Collector — Makefile (no CMake required)
# ─────────────────────────────────────────────────────────────────
#
#  Works on macOS and Linux. If your conda env is active, Boost and
#  OpenSSL are pulled from there automatically. Otherwise it falls
#  back to whatever is installed system-wide.
#
#  Usage:
#    make              - build
#    make clean        - remove build/
#    make help         - show run instructions
# ─────────────────────────────────────────────────────────────────

BUILD_DIR := build
BINARY    := $(BUILD_DIR)/collector
SRC       := src/main.cpp

# ── Compiler: prefer clang++ on mac, g++ on linux ────────────────
CXX := $(shell command -v clang++ 2>/dev/null || echo g++)

# ── Flags ─────────────────────────────────────────────────────────
CXXFLAGS := -std=c++17 -O2 -Wall \
             -DBOOST_ASIO_NO_DEPRECATED

# ── Include paths ─────────────────────────────────────────────────
# If CONDA_PREFIX is set (conda env active), use its headers first.
ifdef CONDA_PREFIX
  INC := -I include \
          -I $(CONDA_PREFIX)/include
else
  INC := -I include
endif

# ── Library paths & libs ──────────────────────────────────────────
ifdef CONDA_PREFIX
  LDFLAGS := -L $(CONDA_PREFIX)/lib \
              -Wl,-rpath,$(CONDA_PREFIX)/lib
else
  LDFLAGS :=
endif

LIBS := -lboost_system \
        -lboost_thread \
        -lboost_json \
        -lssl \
        -lcrypto \
        -lpthread

# ─────────────────────────────────────────────────────────────────

.PHONY: all clean help

all: $(BINARY)

$(BINARY): $(SRC) $(wildcard include/*.hpp)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INC) $(SRC) -o $(BINARY) $(LDFLAGS) $(LIBS)
	@echo ""
	@echo "  Built: $(BINARY)"
	@echo "  Run 'make help' for usage."

clean:
	rm -rf $(BUILD_DIR)
	@echo "  cleaned"

help:
	@echo ""
	@echo "  Project layout (run 'make' from collector/):"
	@echo "    Kalshi-Orderbook-Predictor/"
	@echo "    ├── collector/          ← you are here"
	@echo "    │   ├── include/"
	@echo "    │   ├── src/main.cpp"
	@echo "    │   ├── misc/           ← cdp_api_key.json, .pem key"
	@echo "    │   └── Makefile"
	@echo "    ├── analysis/           ← Python signals + dashboard"
	@echo "    ├── data/               ← all CSV output goes here"
	@echo "    └── results/            ← parquet cache, charts"
	@echo ""
	@echo "  Usage:"
	@echo "    ./build/collector <kalshi_key_id> <kalshi_pem> <coinbase_json> [data_dir]"
	@echo ""
	@echo "  Arguments:"
	@echo "    kalshi_key_id   UUID from Kalshi API keys page"
	@echo "    kalshi_pem      path to your .pem private key file"
	@echo "    coinbase_json   path to your cdp_api_key.json"
	@echo "    data_dir        (optional) CSV output directory, default: ../data"
	@echo ""
	@echo "  Standard run (from collector/):"
	@echo "    ./build/collector 773b19f6-...-f99 misc/key.pem misc/cdp_api_key.json ../data"
	@echo ""
	@echo "  Background run over SSH:"
	@echo "    nohup ./build/collector 773b19f6-...-f99 misc/key.pem misc/cdp_api_key.json ../data \\"
	@echo "      > collector.log 2>&1 & echo \$$! > collector.pid"
	@echo ""
	@echo "  Stop background run:"
	@echo "    kill \$$(cat collector.pid)"
	@echo ""