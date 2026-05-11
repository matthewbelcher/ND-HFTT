#ifndef HFT_TCPSTACK_FLOW_CONTROL_H
#define HFT_TCPSTACK_FLOW_CONTROL_H

#include <cstddef>
#include <cstdint>

// Sliding-window receive-side flow controller.
//
// Tracks how many bytes are currently buffered on the receive path and
// computes the advertised receive window to include in outgoing segments.
//
// When data arrives from the network, call on_receive(). When the application
// consumes data (via recv_fn_), call on_consume(). The window returned by
// recv_window() reflects the remaining buffer capacity.
//
// In the current immediate-delivery model recv_fn is called synchronously
// inside process_data, so on_consume follows on_receive immediately and the
// window stays near RECV_BUF_SIZE under normal operation. The class is still
// correct if an async delivery model is introduced later.
class FlowController {
public:
    // Total receive buffer capacity (bytes).
    static constexpr uint32_t RECV_BUF_SIZE = 65535;

    // Called when plen bytes arrive from the network (fill the receive buffer).
    void on_receive(size_t plen) noexcept;

    // Called when the application consumes plen bytes from the receive buffer.
    void on_consume(size_t plen) noexcept;

    // Advertised receive window: bytes available in the buffer for incoming data.
    uint16_t recv_window() const noexcept;

private:
    uint32_t bytes_in_buffer_{ 0 };
};

#endif // HFT_TCPSTACK_FLOW_CONTROL_H
