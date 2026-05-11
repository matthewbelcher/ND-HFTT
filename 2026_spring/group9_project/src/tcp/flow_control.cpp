#include "tcp/flow_control.h"

#include <algorithm>

void FlowController::on_receive(size_t plen) noexcept {
    bytes_in_buffer_ = std::min(bytes_in_buffer_ + static_cast<uint32_t>(plen),
                                RECV_BUF_SIZE);
}

void FlowController::on_consume(size_t plen) noexcept {
    const auto consumed = static_cast<uint32_t>(plen);
    bytes_in_buffer_ = (consumed <= bytes_in_buffer_) ? bytes_in_buffer_ - consumed : 0u;
}

uint16_t FlowController::recv_window() const noexcept {
    return static_cast<uint16_t>(RECV_BUF_SIZE - bytes_in_buffer_);
}
