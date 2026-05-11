#ifndef HFT_TCPSTACK_TCP_STATE_H
#define HFT_TCPSTACK_TCP_STATE_H

#include <string_view>

// TCP connection states per RFC 793 §3.2.
// The full teardown path (active close) is:
//   ESTABLISHED → FIN_WAIT_1 → FIN_WAIT_2 → TIME_WAIT → CLOSED
// The passive close path is:
//   ESTABLISHED → CLOSE_WAIT → LAST_ACK → CLOSED
// Simultaneous close (both sides send FIN at once) goes:
//   FIN_WAIT_1 → TIME_WAIT → CLOSED  (simplified; omits CLOSING state)
enum class TCPState {
    CLOSED,
    LISTEN,
    SYN_SENT,
    SYN_RECEIVED,
    ESTABLISHED,
    FIN_WAIT_1,
    FIN_WAIT_2,
    TIME_WAIT,
    CLOSE_WAIT,
    LAST_ACK,
};

// Human-readable name for use in logs and test output.
inline std::string_view state_name(TCPState s) {
    switch (s) {
    case TCPState::CLOSED:       return "CLOSED";
    case TCPState::LISTEN:       return "LISTEN";
    case TCPState::SYN_SENT:     return "SYN_SENT";
    case TCPState::SYN_RECEIVED: return "SYN_RECEIVED";
    case TCPState::ESTABLISHED:  return "ESTABLISHED";
    case TCPState::FIN_WAIT_1:   return "FIN_WAIT_1";
    case TCPState::FIN_WAIT_2:   return "FIN_WAIT_2";
    case TCPState::TIME_WAIT:    return "TIME_WAIT";
    case TCPState::CLOSE_WAIT:   return "CLOSE_WAIT";
    case TCPState::LAST_ACK:     return "LAST_ACK";
    }
    return "UNKNOWN";
}

#endif //HFT_TCPSTACK_TCP_STATE_H
