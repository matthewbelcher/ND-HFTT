#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ghostbook/gateway.h"
#include "ghostbook/ndfex/protocol.h"

namespace ghostbook::replay { class ReplayEngine; }

namespace ghostbook::gateway {

using NdfexMarketDataCallback = std::function<void(const ndfex::md::Message &)>;
using NdfexResponseCallback =
    std::function<void(session_id_t, const ndfex::oe::Response &)>;

class NdfexGateway : public Gateway {
public:
  explicit NdfexGateway(logical_clock_t start_clock = 0);
  explicit NdfexGateway(replay::ReplayEngine engine);
  ~NdfexGateway() override;

  bool start(std::uint16_t oe_port = 3000,
             const std::string &md_host = "239.255.0.1",
             std::uint16_t md_port = 12345,
             const std::string &snap_host = "239.255.0.2",
             std::uint16_t snap_port = 12346,
             const std::string &udp_interface = "127.0.0.1");
  void shutdown();

  // Ghostbook base class interface — not used for NDFEX; no-op.
  void process_frame(session_id_t session_id,
                     const protocol::FrameHeader &header,
                     const std::vector<std::uint8_t> &body) override;

  // NDFEX-specific inline processing for tests and the TCP receive path.
  void process_ndfex_request(session_id_t session_id,
                             const ndfex::oe::Request &request);

  // Test helper: create a pre-authenticated session without going through TCP
  // login.
  session_id_t create_test_session(ndfex::oe::client_id_t client_id = 1);

  void set_market_data_callback(NdfexMarketDataCallback cb);
  void set_ndfex_response_callback(NdfexResponseCallback cb);

  // Advance the replay feed by one message and flush any resulting events.
  // No-op when not backed by a ReplayEngine.
  void advance_feed();

private:
  class ImplBase;
  template <typename EngineT> class Impl;
  template <typename EngineT> friend class Impl;
  std::unique_ptr<ImplBase> impl_;
};

} // namespace ghostbook::gateway
