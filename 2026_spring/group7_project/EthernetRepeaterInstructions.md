# EthernetRepeater — Overview & Custom Packet Guide

A Quartus (Cyclone IV) project that receives Ethernet packets via RGMII and mirrors them back out on the same port. Supports 10/100/1000 Mbps auto-negotiation through a Marvell 88E1111 PHY.

Located in [EthernetRepeater/EthernetRepeater/](EthernetRepeater/EthernetRepeater/).

---

## How the Mirror Works

```
PHY (ENET1_RX) → DDR Input Buffers → rgmii_rx_impl → RX RAM + RX FIFO
                                                              ↓
                                                   EthernetRepeater.sv
                                                   (main state machine)
                                                              ↓
                                                   memcopy (RX RAM → TX RAM)
                                                              ↓
                                                        TX FIFO
                                                              ↓
                                              rgmii_tx → DDR Output Buffers → PHY (ENET1_TX)
```

1. The PHY presents incoming frames as DDR RGMII nibbles.
2. `rgmii_rx_impl.sv` strips the preamble/SFD, validates the frame, and writes the payload into one of 8 RX RAM buffers (2 KB each), then signals the RX FIFO.
3. The main state machine in `EthernetRepeater.sv` reads the RX FIFO, uses `memcopy.sv` to copy the packet (minus the 4-byte FCS) from RX RAM into a TX RAM buffer, then pushes a `{buffer#, length}` entry to the TX FIFO.
4. `rgmii_tx.sv` reads the TX FIFO, prepends a preamble + SFD, reads the payload from TX RAM, appends a recalculated CRC32, and drives the DDR output buffers.

---

## Key Files

| File | Purpose |
|---|---|
| `EthernetRepeater.sv` | Top-level module; main RX→TX state machine |
| `rgmii_rx_impl.sv` | RGMII receive state machine, RX RAM writer |
| `rgmii_tx.sv` | RGMII transmit state machine, TX RAM reader |
| `memcopy.sv` | Copies a byte range from one RAM address to another |
| `tx_clock_manager.sv` | Glitch-free clock mux for 10/100/1000 TX clock selection |
| `eth_phy_88e1111_controller.sv` | Configures the Marvell PHY over MDIO |
| `EthernetRepeater.sdc` | Timing constraints (DDR input delays, clock domains) |

---

## Sending a Custom Hardcoded Packet

Instead of mirroring a received packet, the FPGA can transmit a fixed hardcoded payload. The approach reserves **TX buffer 7** (never touched by the normal mirror path) and writes the packet into it once at startup.

All changes are confined to `EthernetRepeater.sv`. No other files need modification.

### Step 1 — Mux the TX RAM write interface

`memcopy` drives `tx_ram_wr_ena/addr/data` directly as module output ports. To also write from a custom init state machine, split the signals:

```systemverilog
// New intermediate signals (driven by memcopy)
logic        mc_tx_ram_wr_ena;
logic [13:0] mc_tx_ram_wr_addr;
logic  [7:0] mc_tx_ram_wr_data;

// Custom packet writer signals
logic        cust_wr_ena    = '0;
logic [13:0] cust_wr_addr   = '0;
logic  [7:0] cust_wr_data   = '0;
logic        cust_init_done = '0;

// Mux: custom writer owns the bus until init finishes
assign tx_ram_wr_ena  = cust_init_done ? mc_tx_ram_wr_ena  : cust_wr_ena;
assign tx_ram_wr_addr = cust_init_done ? mc_tx_ram_wr_addr : cust_wr_addr;
assign tx_ram_wr_data = cust_init_done ? mc_tx_ram_wr_data : cust_wr_data;
```

Update the memcopy port connections (around line 1013) to use the new `mc_*` signals:

```systemverilog
.ram_wr_ena(mc_tx_ram_wr_ena),
.ram_wr_addr(mc_tx_ram_wr_addr),
.ram_wr_data(mc_tx_ram_wr_data)
```

### Step 2 — Define the packet and write it at startup

```systemverilog
localparam int CUSTOM_PKT_LEN = 60;
localparam logic [7:0] CUSTOM_PKT [0:CUSTOM_PKT_LEN-1] = '{
  // Destination MAC: FF:FF:FF:FF:FF:FF (broadcast)
  8'hFF, 8'hFF, 8'hFF, 8'hFF, 8'hFF, 8'hFF,
  // Source MAC: DE:AD:BE:EF:CA:FE
  8'hDE, 8'hAD, 8'hBE, 8'hEF, 8'hCA, 8'hFE,
  // EtherType: 0x0800 (IPv4)
  8'h08, 8'h00,
  // Payload: "Hello FPGA!" + zero-pad to 46 bytes
  8'h48, 8'h65, 8'h6C, 8'h6C, 8'h6F, 8'h20,
  8'h46, 8'h50, 8'h47, 8'h41, 8'h21, 8'h00,
  8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,
  8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,
  8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,
  8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,
  8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,
  8'h00, 8'h00, 8'h00, 8'h00
};
localparam logic [2:0] CUSTOM_PKT_BUF = 3'd7;

logic [5:0] cust_init_idx = '0;

always_ff @(posedge CLOCK_50) begin: custom_pkt_init
  if (!cust_init_done) begin
    cust_wr_ena  <= '1;
    cust_wr_addr <= {CUSTOM_PKT_BUF, 5'b0, cust_init_idx}; // 3 + 11 = 14 bits
    cust_wr_data <= CUSTOM_PKT[cust_init_idx];
    if (cust_init_idx == CUSTOM_PKT_LEN - 1) begin
      cust_init_done <= '1;
      cust_wr_ena    <= '0;
    end else begin
      cust_init_idx <= cust_init_idx + 1'd1;
    end
  end else begin
    cust_wr_ena <= '0;
  end
end: custom_pkt_init
```

### Step 3 — Protect buffer 7

Change the `send_buf` increment (around line 1265) to wrap at 6 so `memcopy` never overwrites buffer 7:

```systemverilog
// Before:
send_buf <= send_buf + 1'd1;

// After:
send_buf <= (send_buf == 3'd6) ? 3'd0 : send_buf + 1'd1;
```

### Step 4 — Choose a trigger

**Trigger on every received packet** — in `S_ERX_QUEUE_SEND` (around line 1287), push the custom packet instead of (or after) the mirrored one:

```systemverilog
S_ERX_QUEUE_SEND: begin
  if (!tx_fifo_wr_full) begin
    tx_fifo_wr_req  <= '1;
    tx_fifo_buf_num <= CUSTOM_PKT_BUF;
    tx_fifo_len     <= CUSTOM_PKT_LEN;
    erx_state <= S_ERX_AWAIT_FIFO;
  end
end
```

**Trigger on button press (KEY[2])** — define `TRANSMIT_ON_KEY_2` in Quartus (Assignments → Settings → Verilog HDL Input) and update the existing block (around line 641) to point at the custom buffer:

```systemverilog
tx_fifo_buf_num <= CUSTOM_PKT_BUF;
tx_fifo_len     <= CUSTOM_PKT_LEN;
tx_fifo_wr_req  <= '1;
```

---

## Ethernet Frame Layout Reference

The TX engine adds the preamble (7× `0x55`), SFD (`0xD5`), and CRC32 automatically. Your packet data in TX RAM should be:

```
Bytes 0–5   Destination MAC (6 bytes)
Bytes 6–11  Source MAC (6 bytes)
Bytes 12–13 EtherType (2 bytes, e.g. 0x0800 = IPv4)
Bytes 14–59 Payload (zero-padded to meet 46-byte minimum = 60 bytes total)
```
