module in_out_mod(
  input wire clk,
  input wire reset,
  input wire [7:0] user_module__reading_data,
  input wire user_module__reading_valid,
  input wire user_module__output_ready,
  output wire [7:0] user_module__output_data,
  output wire user_module__output_valid,
  output wire user_module__reading_ready
);
  reg [63:0] ____state_1;
  reg [63:0] ____state_2;
  reg [63:0] ____state_3;
  reg [63:0] ____state_7;
  wire [7:0] output_out;
  wire [63:0] new_first;
  wire p0_stage_done;
  wire [63:0] new_second;
  wire [63:0] new_third;
  assign output_out = ____state_7[63:56];
  assign new_first = {____state_1[55:0], user_module__reading_data};
  assign p0_stage_done = user_module__reading_valid & user_module__output_ready;
  assign new_second = {____state_2[55:0], ____state_1[63:56]};
  assign new_third = {____state_3[55:0], ____state_2[63:56]};
  always_ff @ (posedge clk) begin
    if (reset) begin
      ____state_1 <= 64'h0000_0000_0000_0000;
      ____state_2 <= 64'h0000_0000_0000_0000;
      ____state_3 <= 64'h0000_0000_0000_0000;
      ____state_7 <= 64'h0000_0000_0000_0000;
    end else begin
      ____state_1 <= p0_stage_done ? new_first : ____state_1;
      ____state_2 <= p0_stage_done ? new_second : ____state_2;
      ____state_3 <= p0_stage_done ? new_third : ____state_3;
      ____state_7 <= p0_stage_done ? new_first : ____state_7;
    end
  end
  assign user_module__output_data = output_out;
  assign user_module__output_valid = user_module__reading_valid;
  assign user_module__reading_ready = p0_stage_done;
endmodule
