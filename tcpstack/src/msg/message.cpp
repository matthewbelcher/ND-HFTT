#include "msg/message.h"

#include <algorithm>
#include <cstring>

// Encode / decode

size_t msg_encode(const Message& msg, uint8_t* buf, size_t buf_len) {
    if (msg.payload_len > MSG_MAX_PAYLOAD) return 0;
    const size_t total = MSG_HDR_LEN + msg.payload_len;
    if (buf_len < total) return 0;

    buf[0] = static_cast<uint8_t>(msg.type);
    buf[1] = static_cast<uint8_t>(msg.payload_len >> 8);
    buf[2] = static_cast<uint8_t>(msg.payload_len & 0xFFu);
    std::memcpy(buf + MSG_HDR_LEN, msg.payload, msg.payload_len);
    return total;
}

bool msg_decode(const uint8_t* buf, size_t buf_len, Message& msg) {
    if (buf_len < MSG_HDR_LEN) return false;

    const uint16_t plen =
        (static_cast<uint16_t>(buf[1]) << 8) | static_cast<uint16_t>(buf[2]);
    if (plen > MSG_MAX_PAYLOAD) return false;
    if (buf_len < MSG_HDR_LEN + plen) return false;

    msg.type        = static_cast<MsgType>(buf[0]);
    msg.payload_len = plen;
    std::memcpy(msg.payload, buf + MSG_HDR_LEN, plen);
    return true;
}

// MessageFramer

void MessageFramer::feed(const uint8_t* data, size_t len, const Callback& cb) {
    size_t consumed = 0;

    while (consumed < len) {
        const size_t space = sizeof(buf_) - buf_len_;
        const size_t copy  = std::min(space, len - consumed);
        std::memcpy(buf_ + buf_len_, data + consumed, copy);
        buf_len_  += copy;
        consumed  += copy;

        while (buf_len_ >= MSG_HDR_LEN) {
            const uint16_t plen =
                (static_cast<uint16_t>(buf_[1]) << 8) | static_cast<uint16_t>(buf_[2]);

            // Corrupt length field: reset and stop processing this batch.
            if (plen > MSG_MAX_PAYLOAD) {
                reset();
                break;
            }

            const size_t msg_len = MSG_HDR_LEN + plen;
            if (buf_len_ < msg_len) break;

            Message msg{};
            msg_decode(buf_, msg_len, msg);
            cb(msg);

            buf_len_ -= msg_len;
            if (buf_len_ > 0)
                std::memmove(buf_, buf_ + msg_len, buf_len_);
        }
    }
}

void MessageFramer::reset() noexcept {
    buf_len_ = 0;
}
