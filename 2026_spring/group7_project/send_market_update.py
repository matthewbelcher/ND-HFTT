#!/usr/bin/env python3
"""
Send a trade_summary market data packet over UDP.

Matches the packed C++ struct layout from dgram.hpp / types.hpp:

md_header (23 bytes):
  uint64_t  magic_number    8 bytes
  uint16_t  length          2 bytes
  uint32_t  seq_num         4 bytes  (SeqNum = StrongType<uint32_t>)
  uint64_t  timestamp       8 bytes
  uint8_t   msg_type        1 byte

trade_summary payload (13 bytes):
  uint32_t  symbol          4 bytes  (Symbol = StrongType<uint32_t>)
  uint8_t   aggressor_side  1 byte   (SIDE enum : uint8_t)
  uint32_t  total_quantity  4 bytes  (Quantity = StrongType<uint32_t>)
  int32_t   last_price      4 bytes  (Price = StrongType<int32_t>)

Total struct size: 36 bytes
"""

import socket
import struct
import time
import argparse

# ── Constants ────────────────────────────────────────────────────────────────

MAGIC_BYTES = b'GOIRISH!'          # uint64_t magic_number, little-endian
MAGIC_NUMBER = struct.unpack('<Q', MAGIC_BYTES)[0]

MSG_TYPE_TRADE_SUMMARY = 5

SIDE_BUY  = 1
SIDE_SELL = 2

# md_header: magic(8) + length(2) + seq_num(4) + timestamp(8) + msg_type(1) = 23 bytes
HEADER_FMT   = '<Q H I Q B'
HEADER_SIZE  = struct.calcsize(HEADER_FMT)  # 23

# trade_summary payload: symbol(4) + side(1) + quantity(4) + price(4) = 13 bytes
PAYLOAD_FMT  = '<I B I i'
PAYLOAD_SIZE = struct.calcsize(PAYLOAD_FMT)  # 13

TOTAL_SIZE   = HEADER_SIZE + PAYLOAD_SIZE    # 36


# ── Packet builder ───────────────────────────────────────────────────────────

def build_trade_summary(
    symbol: int,
    aggressor_side: int,
    total_quantity: int,
    last_price: int,
    seq_num: int = 1,
    timestamp: int | None = None,
) -> bytes:
    """
    Build a packed trade_summary datagram.

    Args:
        symbol:          uint32 symbol ID
        aggressor_side:  SIDE_BUY (1) or SIDE_SELL (2)
        total_quantity:  uint32 total quantity traded
        last_price:      int32 last trade price
        seq_num:         uint32 sequence number (default 1)
        timestamp:       uint64 nanosecond timestamp (default: current time)

    Returns:
        bytes of length 36 matching the packed C++ struct
    """
    if timestamp is None:
        timestamp = int(time.time_ns())

    # length field = total struct size (matches C++ convention)
    length = TOTAL_SIZE

    header = struct.pack(
        HEADER_FMT,
        MAGIC_NUMBER,       # uint64_t magic_number
        length,             # uint16_t length
        seq_num,            # uint32_t seq_num
        timestamp,          # uint64_t timestamp
        MSG_TYPE_TRADE_SUMMARY,  # uint8_t msg_type
    )

    payload = struct.pack(
        PAYLOAD_FMT,
        symbol,             # uint32_t symbol
        aggressor_side,     # uint8_t  aggressor_side
        total_quantity,     # uint32_t total_quantity
        last_price,         # int32_t  last_price
    )

    packet = header + payload
    assert len(packet) == TOTAL_SIZE, f"Size mismatch: {len(packet)} != {TOTAL_SIZE}"
    return packet


# ── Sender ───────────────────────────────────────────────────────────────────

def send_packet(
    packet: bytes,
    dest_ip: str,
    dest_port: int,
    src_port: int = 0,
) -> None:
    """Send a UDP datagram to dest_ip:dest_port."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(('', src_port))
        sock.sendto(packet, (dest_ip, dest_port))
    print(f"Sent {len(packet)} bytes to {dest_ip}:{dest_port}")


# ── Debug helpers ─────────────────────────────────────────────────────────────

def hexdump(data: bytes, label: str = "") -> None:
    if label:
        print(f"\n{label}")
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        hex_part  = ' '.join(f'{b:02x}' for b in chunk)
        ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        print(f"  {i:04x}  {hex_part:<47}  {ascii_part}")


def print_fields(
    symbol, aggressor_side, total_quantity, last_price, seq_num, timestamp
) -> None:
    side_str = "BUY" if aggressor_side == SIDE_BUY else "SELL"
    print("\nPacket fields:")
    print(f"  magic_number   : GOIRISH! (0x{MAGIC_NUMBER:016x})")
    print(f"  length         : {TOTAL_SIZE}")
    print(f"  seq_num        : {seq_num}")
    print(f"  timestamp      : {timestamp}")
    print(f"  msg_type       : TRADE_SUMMARY ({MSG_TYPE_TRADE_SUMMARY})")
    print(f"  symbol         : {symbol}")
    print(f"  aggressor_side : {side_str} ({aggressor_side})")
    print(f"  total_quantity : {total_quantity}")
    print(f"  last_price     : {last_price}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Send a trade_summary market data packet over UDP"
    )
    parser.add_argument('--dest-ip',    default='192.168.0.1',
                        help='Destination IP (default: 192.168.0.1)')
    parser.add_argument('--dest-port',  type=int, default=12345,
                        help='Destination UDP port (default: 12345)')
    parser.add_argument('--src-port',   type=int, default=0,
                        help='Source UDP port (default: OS assigned)')
    parser.add_argument('--symbol',     type=int, default=1,
                        help='Symbol ID uint32 (default: 1)')
    parser.add_argument('--side',       choices=['buy', 'sell'], default='buy',
                        help='Aggressor side (default: buy)')
    parser.add_argument('--quantity',   type=int, default=100,
                        help='Total quantity uint32 (default: 100)')
    parser.add_argument('--price',      type=int, default=15,
                        help='Last price int32 (default: 15, which is > 10 '
                             'so the FPGA will respond)')
    parser.add_argument('--seq-num',    type=int, default=1,
                        help='Sequence number (default: 1)')
    parser.add_argument('--count',      type=int, default=1,
                        help='Number of packets to send (default: 1)')
    parser.add_argument('--interval',   type=float, default=1.0,
                        help='Interval between packets in seconds (default: 1.0)')
    parser.add_argument('--dry-run',    action='store_true',
                        help='Build and print the packet but do not send')
    return parser.parse_args()


def main():
    args = parse_args()

    side = SIDE_BUY if args.side == 'buy' else SIDE_SELL

    for i in range(args.count):
        seq = args.seq_num + i
        ts  = int(time.time_ns())

        packet = build_trade_summary(
            symbol         = args.symbol,
            aggressor_side = side,
            total_quantity = args.quantity,
            last_price     = args.price,
            seq_num        = seq,
            timestamp      = ts,
        )

        print_fields(args.symbol, side, args.quantity, args.price, seq, ts)
        hexdump(packet, label="Raw bytes:")

        if args.dry_run:
            print("\n[dry-run] Packet NOT sent.")
        else:
            send_packet(packet, args.dest_ip, args.dest_port, args.src_port)

        if i < args.count - 1:
            time.sleep(args.interval)


if __name__ == '__main__':
    main()