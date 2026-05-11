#ifndef HFT_TCPSTACK_MESSAGE_H
#define HFT_TCPSTACK_MESSAGE_H

#include <cstddef>
#include <cstdint>
#include <functional>

// Wire format
//
//  Byte 0    : message type (MsgType)
//  Bytes 1-2 : payload length in bytes (uint16_t, big-endian, 0-253)
//  Bytes 3+  : payload
//
//  Total frame = MSG_HDR_LEN + payload_len bytes.

inline constexpr size_t MSG_HDR_LEN     = 3;
// 253 keeps the total frame under 256 bytes, so a single stack buffer covers
// any valid message and the per-message overhead stays cache-friendly.
inline constexpr size_t MSG_MAX_PAYLOAD = 253;
inline constexpr size_t MSG_MAX_LEN     = MSG_HDR_LEN + MSG_MAX_PAYLOAD;

// Message types
// 0x01 is reserved for HEARTBEAT. Applications may define additional type
// codes in the range 0x02-0xFF by casting to MsgType.

enum class MsgType : uint8_t {
    HEARTBEAT = 0x01,  // keepalive, no payload
};

// Message
// type        : identifies the message kind
// payload_len : number of valid bytes in payload[]
// payload[]   : raw bytes; interpretation is application-defined

struct Message {
    MsgType  type;
    uint16_t payload_len;
    uint8_t  payload[MSG_MAX_PAYLOAD];
};

// Encode / decode

// Serialize msg into buf.
// Returns bytes written (MSG_HDR_LEN + msg.payload_len).
// Returns 0 if buf_len < MSG_HDR_LEN + msg.payload_len or payload_len > MSG_MAX_PAYLOAD.
size_t msg_encode(const Message& msg, uint8_t* buf, size_t buf_len);

// Deserialize one complete message from buf into msg.
// buf_len must be >= MSG_HDR_LEN + embedded payload_len.
// Returns true on success; false if buf is too short or payload_len > MSG_MAX_PAYLOAD.
bool msg_decode(const uint8_t* buf, size_t buf_len, Message& msg);

// MessageFramer
// Reassembles complete Messages from a raw TCP byte stream, handling
// partial reads and multiple messages per call transparently.
class MessageFramer {
public:
    using Callback = std::function<void(const Message&)>;

    // Feed raw bytes arriving from the TCP stream.
    // Calls cb once for each complete message decoded.
    void feed(const uint8_t* data, size_t len, const Callback& cb);

    // Discard buffered state (e.g. after a connection reset).
    void reset() noexcept;

private:
    uint8_t buf_[MSG_MAX_LEN];
    size_t  buf_len_{ 0 };
};

#endif // HFT_TCPSTACK_MESSAGE_H
