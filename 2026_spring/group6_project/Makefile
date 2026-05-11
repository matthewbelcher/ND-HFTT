MACHINE=brabidea@129.74.160.70
REMOTE_DIR ?= ~/ghostbook
CPP_BUILD_DIR ?= build
CPP_BUILD_TYPE ?= Debug

.PHONY: sync remote-cpp-configure remote-cpp-build remote-cpp-test

sync:
	@echo "🛜 Syncing files to $(MACHINE) machine..."
	@rsync \
		-azvP \
		--exclude ".idea/" \
		--exclude ".git/" \
		./ $(MACHINE):~/ghostbook/
	@echo "✅ Sync successful"

remote-cpp-configure: sync
	@ssh $(MACHINE) "cd $(REMOTE_DIR)/cpp && cmake -S . -B $(CPP_BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CPP_BUILD_TYPE)"

remote-cpp-build: remote-cpp-configure
	@ssh $(MACHINE) "cd $(REMOTE_DIR)/cpp && cmake --build $(CPP_BUILD_DIR) --config $(CPP_BUILD_TYPE)"

remote-cpp-test: remote-cpp-build
	@ssh $(MACHINE) "cd $(REMOTE_DIR)/cpp && ctest --test-dir $(CPP_BUILD_DIR) --build-config $(CPP_BUILD_TYPE) --output-on-failure"
