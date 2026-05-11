#ifndef HFT_MSG_MARKET_DATA_H
#define HFT_MSG_MARKET_DATA_H

#include "common.h"

#include <cstdint>
#include <unistd.h>

namespace hft::msg::md {

    /* Utility Macros */

    #define md_msg_size(_MessageType)  (static_cast<ssize_t>(sizeof(hft::msg::md::MessageHeader) + sizeof(_MessageType)))


    /* Magic Numbers */

    #define MD_MAGIC_STRING             ("GOIRISH!")
    #define MD_SNAPSHOT_MAGIC_STRING    ("SNAPSHOT")

    static const uint64_t MAGIC_NUMBER = *reinterpret_cast<const uint64_t *>(MD_MAGIC_STRING);
    static const uint64_t SNAPSHOT_MAGIC_NUMBER = *reinterpret_cast<const uint64_t *>(MD_SNAPSHOT_MAGIC_STRING);

    /* Enums */

    enum class MSG_TYPE : msg_type_t {
        HEARTBEAT = 0,
        NEW_ORDER = 1,
        DELETE_ORDER = 2,
        MODIFY_ORDER = 3,
        TRADE = 4,
        TRADE_SUMMARY = 5,
        SNAPSHOT_INFO = 6,
    };

    /* Structures */

    typedef struct {
        uint64_t magic_number;
        uint16_t length;
        seq_num_t seq_num;
        timestamp_t timestamp;

        MSG_TYPE msg_type;
    } __attribute__((packed)) MessageHeader;

    typedef struct {
        order_id_t order_id;
        symbol_id_t symbol;
        SIDE side;
        quantity_t quantity;
        price_t price;
        order_flags_t flags;
    } __attribute__((packed)) NewOrder;

    typedef struct {
        order_id_t order_id;
    } __attribute__((packed)) DeleteOrder;

    typedef struct {
        order_id_t order_id;
        SIDE side;
        quantity_t quantity;
        price_t price;
    } __attribute__((packed)) ModifyOrder;

    typedef struct {
        order_id_t order_id;
        quantity_t quantity;
        price_t price;
    } __attribute__((packed)) Trade;

    typedef struct {
        symbol_id_t symbol;
        SIDE aggressor_side;
        quantity_t total_quantity;
        price_t last_price;
    } __attribute__((packed)) TradeSummary;

    typedef struct {
        symbol_id_t symbol;
        quantity_t bid_count;
        quantity_t ask_count;
        seq_num_t last_md_seq_num;
    } __attribute__((packed)) SnapshotInfo;

    typedef union {
        NewOrder new_order;
        DeleteOrder delete_order;
        ModifyOrder modify_order;
        Trade trade;
        TradeSummary trade_summary;
        SnapshotInfo snapshot_info;
    } MdBody;

    typedef struct {
        MessageHeader header;
        MdBody body;
    } __attribute__((packed)) MdMessage;

    /* Function Headers */
    ssize_t     parse_message(const uint8_t *buf, ssize_t length, MdMessage *m);
    void        dump_message(const MdMessage *m);

}

#endif //HFT_MSG_MARKET_DATA_H
