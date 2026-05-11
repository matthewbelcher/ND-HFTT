CWD = $(shell pwd -P)

BUILD_TYPE	:= Debug

INCLUDE 	:= -I$(CWD)/include
CC 			:= clang++
CFLAGS 		:= -std=c++20 -Wall -Wextra -Werror $(INCLUDE)
LD 			:= clang++
LDFLAGS 	:= -Llib
AR 			:= ar
ARFLAGS 	:= rcs

HFT_LIB 	:= -lhft
MACHINE 	:= hft0

TARGETS 	:= bin/app bin/liquidate bin/gb_test lib/libhft.a

# Utility variables
_NULL :=
_SPACE := $(_NULL) $(_NULL)
_COMMA := ,

ifeq ($(BUILD_TYPE),Debug)
	CFLAGS += -c
else
	CFLAGS += -O3
endif

all:			$(TARGETS)
	@echo "✅ Build successful"

bin/app: 		bin/main.o lib/libhft.a
	@echo "⛓️ Linking executable '$@'..."
	$(LD) $(LDFLAGS) -o $@ $^

bin/liquidate:	bin/liquidate.o lib/libhft.a
	@echo "⛓️ Linking executable '$@'..."
	$(LD) $(LDFLAGS) -o $@ $^

bin/gb_test:	bin/gb_test.o lib/libhft.a
	@echo "⛓️ Linking executable '$@'..."
	$(LD) $(LDFLAGS) -o $@ $^

bin/probe:	bin/probe.o lib/libhft.a
	@echo "⛓️ Linking executable '$@'..."
	$(LD) $(LDFLAGS) -o $@ $^

lib/libhft.a:	bin/multicast.o bin/market_data.o bin/orderbook.o bin/ExchangeClient.o bin/order_entry.o bin/ExchangeOrder.o bin/PositionTracker.o bin/RiskSystem.o
	@echo "📚 Building library archive '$@'..."
	$(AR) $(ARFLAGS) $@ $^

bin/%.o:		src/%.cpp include/hft/*.h include/hft/msg/*.h
	@echo "🔨 Compiling object file '$@'..."
	$(CC) $(CFLAGS) -c -o $@ $<

bin/test_risk:	test/test_risk.cpp lib/libhft.a
	@echo "🔨 Compiling and linking test '$@'..."
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(HFT_LIB)

test:			bin/test_risk
	@echo "🧪 Running tests..."
	./bin/test_risk

.clangd:
	@echo "Building $@"
	echo "CompileFlags:" > $@
	echo "	Add: [$(subst $(_SPACE),$(_COMMA),$(strip $(CFLAGS)))]" >> $@

clean:
	@echo "🗑️ Cleaning object files..."
	rm -f bin/*.o
	@echo "🗑️ Cleaning targets..."
	rm -f $(TARGETS) bin/test_risk bin/liquidate

# Send files to remote machine
sync:
	@echo "🛜 Syncing files to $(MACHINE) machine..."
	@rsync \
		-azvP \
		--exclude "bin/**" \
		--exclude "lib/**" \
		--exclude ".idea/" \
		--exclude ".git/" \
		./ $(MACHINE):~/trading_project/
	@echo "✅ Sync successful"
