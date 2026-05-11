#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <memory>

#include "ghostbook/gateway.h"
#include "ghostbook/protocol.h"

namespace ghostbook::gateway {

class TcpGateway : public Gateway {
public:
    explicit TcpGateway(logical_clock_t start_clock = 0);
    ~TcpGateway() override;

    bool start(
        std::uint16_t port = 1234,
        const std::string& market_data_host = "239.255.0.1",
        std::uint16_t market_data_port = 12345,
        std::uint16_t snapshot_port = 1235
    );
    void shutdown();

    void process_frame(
        session_id_t session_id,
        const protocol::FrameHeader& header,
        const std::vector<std::uint8_t>& body
    ) override;

    // Test helper: create an authenticated session without socket logon.
    session_id_t create_test_session(
        std::uint16_t comp_id = 1,
        std::uint64_t app_instance = 1,
        std::uint32_t client_ip = 0,
        std::uint16_t heartbeat_interval_ms = 1000
    );

private:
    class Impl;
    friend class TcpGateway::Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ghostbook::gateway
