#ifndef HFT_MSG_COMMON_H
#define HFT_MSG_COMMON_H

#include <cstdint>
#include <unistd.h>

namespace hft::msg {

    /* Utility Macros */

    #define side_string(_side)          ((_side == hft::msg::SIDE::BUY) ? "BUY" : "SELL")


    /* Type Aliases */

    using seq_num_t = uint32_t;
    using timestamp_t = uint64_t;
    using msg_type_t = uint8_t;
    using order_id_t = uint64_t;
    using symbol_id_t = uint32_t;
    using quantity_t = uint32_t;
    using price_t = int32_t;
    using order_flags_t = uint8_t;
    using status_t = uint8_t;


    /* Enums */

    enum class SIDE : uint8_t {
        BUY = 1,
        SELL = 2,
    };

}

#endif //HFT_MSG_COMMON_H
