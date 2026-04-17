#ifndef HFT_TCPSTACK_FLOW_CONTROL_H
#define HFT_TCPSTACK_FLOW_CONTROL_H

// Phase 4 — Sliding-window flow control.
//
// Tasks (implemented in src/tcp/flow_control.cpp in Phase 4):
//   - Track snd_wnd (peer's advertised window) and rcv_wnd (our window).
//   - Clamp snd_nxt to snd_una + snd_wnd before sending.
//   - Update rcv_wnd in ACKs based on available receive-buffer space.
//   - Zero-window probe: send a 1-byte segment when snd_wnd == 0.
//
// In Phase 3, TCPConnection::send() performs a simple snd_wnd clamp directly.
// The dedicated FlowController class will be extracted here in Phase 4.

#endif //HFT_TCPSTACK_FLOW_CONTROL_H
