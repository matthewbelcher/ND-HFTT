module check_counter(
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
  reg [3:0] ____state_8;
  wire eq_300;
  wire p0_stage_done;
  wire [1:0] concat_328;
  wire [63:0] new_output;
  wire [63:0] new_first;
  wire [1:0] concat_334;
  wire [3:0] new_delay;
  wire [7:0] output_out;
  wire [63:0] new_second;
  wire [63:0] new_third;
  wire [63:0] one_hot_sel_329;
  wire [3:0] one_hot_sel_335;
  assign eq_300 = ____state_8 == 4'h7;
  assign p0_stage_done = user_module__reading_valid & user_module__output_ready;
  assign concat_328 = {eq_300 & p0_stage_done, ~eq_300 & p0_stage_done};
  assign new_output = {____state_7[55:0], 8'h00};
  assign new_first = {____state_1[55:0], user_module__reading_data};
  assign concat_334 = {~eq_300 & p0_stage_done, eq_300 & p0_stage_done};
  assign new_delay = ____state_8 + 4'h1;
  assign output_out = ____state_7[63:56];
  assign new_second = {____state_2[55:0], ____state_1[63:56]};
  assign new_third = {____state_3[55:0], ____state_2[63:56]};
  assign one_hot_sel_329 = new_output & {64{concat_328[0]}} | new_first & {64{concat_328[1]}};
  assign one_hot_sel_335 = 4'h0 & {4{concat_334[0]}} | new_delay & {4{concat_334[1]}};
  always_ff @ (posedge clk) begin
    if (reset) begin
      ____state_1 <= 64'h0000_0000_0000_0000;
      ____state_2 <= 64'h0000_0000_0000_0000;
      ____state_3 <= 64'h0000_0000_0000_0000;
      ____state_7 <= 64'h0000_0000_0000_0000;
      ____state_8 <= 4'h0;
    end else begin
      ____state_1 <= p0_stage_done ? new_first : ____state_1;
      ____state_2 <= p0_stage_done ? new_second : ____state_2;
      ____state_3 <= p0_stage_done ? new_third : ____state_3;
      ____state_7 <= p0_stage_done ? one_hot_sel_329 : ____state_7;
      ____state_8 <= p0_stage_done ? one_hot_sel_335 : ____state_8;
    end
  end
  assign user_module__output_data = output_out;
  assign user_module__output_valid = user_module__reading_valid;
  assign user_module__reading_ready = p0_stage_done;
endmodule
