module storage_test(
    input        clk,
    input        rst,
    input        new_payload,
    input        wr_en,
    input [63:0] din,
    input        rd_en,
    output reg [511:0] dout
);

    localparam IDLE  = 2'd0;
    localparam STORE = 2'd1;
    localparam SEND  = 2'd2;

    reg [2:0] curr_state;
    reg [2:0] next_state;

    reg [2:0] store_slot;       // 0 to 7 for 8 slots
    reg [63:0] slot [0:7];      // 8 x 64-bit registers

    wire full = (store_slot == 3'd8);

    // State transition
    always @(posedge clk or posedge rst) begin
        if (rst)
            curr_state <= IDLE;
        else
            curr_state <= next_state;
    end

    // Next state logic
    always @(*) begin
        case (curr_state)
            IDLE:  next_state = wr_en ? STORE : (rd_en && full ? SEND : IDLE);
            STORE: next_state = wr_en ? STORE : (rd_en && full ? SEND : IDLE);
            SEND:  next_state = wr_en ? STORE : IDLE;
            default: next_state = IDLE;
        endcase
    end

    // Storage and output logic
    integer i;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            store_slot <= 3'd0;
            dout <= 512'd0;
            for (i = 0; i < 8; i = i + 1)
                slot[i] <= 64'd0;
        end else begin
            case (curr_state)
                STORE: begin
                    if (wr_en && store_slot < 3'd8) begin
                        slot[store_slot] <= din;
                        store_slot <= store_slot + 3'd1;
                    end
                    if (new_payload) begin
                        store_slot <= 3'd0;
                    end
                end
                SEND: begin
                    if (rd_en && full) begin
                        dout <= {slot[0], slot[1], slot[2], slot[3], slot[4], slot[5], slot[6], slot[7]};
                        store_slot <= 3'd0;  // Optionally clear index for next round
                    end
                end
                default: ;
            endcase
        end
    end

endmodule
