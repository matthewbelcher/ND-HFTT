#include "../include/hft/msg/common.h"
#include "../include/hft/msg/market_data.h"

#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <iostream>


namespace hft::msg::md {

    /**
     * Parse a market data update message from raw buffer
     *
     * @param buf Raw message buffer
     * @param length Length of message buffer in bytes
     * @param m Pointer to `MdMessage` object to write
     * @return Number of bytes parsed into message (-1 on error)
     */
    ssize_t parse_message(const uint8_t *buf, const ssize_t length, MdMessage *m) {

        if (length < static_cast<ssize_t>(sizeof(MessageHeader))) {
            return -1;
        }

        ssize_t bytes_read = 0;

        // Read header
        MessageHeader header = {};
        memcpy(&header, buf, sizeof(header));
        bytes_read += sizeof(header);

        if (header.magic_number != MAGIC_NUMBER && header.magic_number != SNAPSHOT_MAGIC_NUMBER) {
            return -1;
        }

        MdBody body = {};
        if (header.msg_type == MSG_TYPE::HEARTBEAT) {
            if (header.length != sizeof(MessageHeader)) { return -1; }
            // Do nothing?

        } else if (header.msg_type == MSG_TYPE::NEW_ORDER) {
            if (header.length != md_msg_size(NewOrder)) { return -1; }

            memcpy(&body.new_order, buf + bytes_read, sizeof(NewOrder));
            bytes_read += sizeof(NewOrder);

        } else if (header.msg_type == MSG_TYPE::DELETE_ORDER) {
            if (header.length != md_msg_size(DeleteOrder)) { return -1; }

            memcpy(&body.delete_order, buf + bytes_read, sizeof(DeleteOrder));
            bytes_read += sizeof(DeleteOrder);

        } else if (header.msg_type == MSG_TYPE::MODIFY_ORDER) {
            if (header.length != md_msg_size(ModifyOrder)) { return -1; }

            memcpy(&body.modify_order, buf + bytes_read, sizeof(ModifyOrder));
            bytes_read += sizeof(ModifyOrder);

        } else if (header.msg_type == MSG_TYPE::TRADE) {
            if (header.length != md_msg_size(Trade)) { return -1; }

            memcpy(&body.trade, buf + bytes_read, sizeof(Trade));
            bytes_read += sizeof(Trade);

        } else if (header.msg_type == MSG_TYPE::TRADE_SUMMARY) {
            if (header.length != md_msg_size(TradeSummary)) { return -1; }

            memcpy(&body.trade_summary, buf + bytes_read, sizeof(TradeSummary));
            bytes_read += sizeof(TradeSummary);

        } else if (header.msg_type == MSG_TYPE::SNAPSHOT_INFO) {
            if (header.length != md_msg_size(SnapshotInfo)) { return -1; }

            memcpy(&body.snapshot_info, buf + bytes_read, sizeof(SnapshotInfo));
            bytes_read += sizeof(SnapshotInfo);

        } else {
            // Message type does not match any valid option
            return -1;
        }

        m->header = header;
        m->body = body;

        return bytes_read;
    }


    /**
     * Dump message information to standard output
     *
     * @param m Pointer to MdMessage structure
     */
    void dump_message(const MdMessage *m) {

        printf("=== Begin message (%p) ===\n", static_cast<const void *>(m));

        char magic_number_string[9] = {};
        memcpy(magic_number_string, &m->header.magic_number, sizeof(m->header.magic_number));
        printf("-> Magic number: %s\n", magic_number_string);

        printf("-> Length: %" PRIu16 "\n", m->header.length);
        printf("-> Sequence number: %" PRIu16 "\n", m->header.seq_num);
        printf("-> Timestamp: %" PRIu64 "\n", m->header.timestamp);

        if (m->header.msg_type == MSG_TYPE::HEARTBEAT) {
            printf("-> Message type: HEARTBEAT\n");

        } else if (m->header.msg_type == MSG_TYPE::NEW_ORDER) {
            printf("-> Message type: NEW_ORDER\n");

            const auto [order_id, symbol, side, quantity, price, flags]= m->body.new_order;
            printf("-> Order ID: %" PRIu64 "\n", order_id);
            printf("-> Symbol: %" PRIu32 "\n", symbol);
            printf("-> Side: %s\n", side_string(side));
            printf("-> Quantity: %" PRIu32 "\n", quantity);
            printf("-> Price: %" PRId32 "\n", price);
            printf("-> Flags: %" PRIx8 "\n", flags);

        } else if (m->header.msg_type == MSG_TYPE::DELETE_ORDER) {
            printf("-> Message type: DELETE_ORDER\n");

            const auto [order_id] = m->body.delete_order;
            printf("-> Order ID: %" PRIu64 "\n", order_id);
        } else if (m->header.msg_type == MSG_TYPE::MODIFY_ORDER) {
            printf("-> Message type: MODIFY_ORDER\n");

            const auto [order_id, side, quantity, price] = m->body.modify_order;
            printf("-> Order ID: %" PRIu64 "\n", order_id);
            printf("-> Side: %s\n", side_string(side));
            printf("-> Quantity: %" PRIu32 "\n", quantity);
            printf("-> Price: %" PRId32 "\n", price);

        } else if (m->header.msg_type == MSG_TYPE::TRADE) {
            printf("-> Message type: TRADE\n");

            const auto [order_id, quantity, price] = m->body.trade;
            printf("-> Order ID: %" PRIu64 "\n", order_id);
            printf("-> Quantity: %" PRIu32 "\n", quantity);
            printf("-> Price: %" PRId32 "\n", price);

        } else if (m->header.msg_type == MSG_TYPE::TRADE_SUMMARY) {
            printf("-> Message type: TRADE_SUMMARY\n");

            const auto [symbol, aggressor_side, total_quantity, last_price] = m->body.trade_summary;
            printf("-> Symbol: %" PRIu32 "\n", symbol);
            printf("-> Aggressor side: %s\n", side_string(aggressor_side));
            printf("-> Total quantity: %" PRIu32 "\n", total_quantity);
            printf("-> Last price: %" PRId32 "\n", last_price);

        } else if (m->header.msg_type == MSG_TYPE::SNAPSHOT_INFO) {
            printf("-> Message type: SNAPSHOT_INFO\n");

            const auto [symbol, bid_count, ask_count, last_md_seq_num] = m->body.snapshot_info;
            printf("-> Symbol: %" PRIu32 "\n", symbol);
            printf("-> Bids: %" PRIu32 "\n", bid_count);
            printf("-> Asks: %" PRIu32 "\n", ask_count);
            printf("-> Last MD sequence number: %" PRIu32 "\n", last_md_seq_num);
        }

        printf("=== End message (%p) ===\n", static_cast<const void *>(m));
    }
}
