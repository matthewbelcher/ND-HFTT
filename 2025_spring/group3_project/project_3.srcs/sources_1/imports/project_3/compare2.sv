module compare2_module(
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
  reg [3:0] ____state_8;
  reg [59:0] ____state_7;
  wire eq_428;
  wire eq_429;
  wire eq_435;
  wire p0_stage_done;
  wire and_452;
  wire and_454;
  wire [1:0] __state_7_constant_bits_6_width_2;
  wire [1:0] __state_7_variable_bits_4_width_2;
  wire [1:0] __state_7_constant_bits_6_width_2__1;
  wire [1:0] __state_7_variable_bits_0_width_2;
  wire [1:0] concat_474;
  wire [3:0] new_delay;
  wire [2:0] concat_482;
  wire [59:0] new_output_slice_reduced;
  wire [7:0] output_out;
  wire [63:0] new_first;
  wire [63:0] new_second;
  wire [63:0] new_third;
  wire [3:0] one_hot_sel_475;
  wire [59:0] one_hot_sel_483;
  wire and_496;
  assign eq_428 = ____state_1[55:0] == 56'h44_4444_4444_4444;
  assign eq_429 = user_module__reading_data == 8'h44;
  assign eq_435 = ____state_8 == 4'h7;
  assign p0_stage_done = user_module__reading_valid & user_module__output_ready;
  assign and_452 = eq_435 & ~(eq_428 & eq_429);
  assign and_454 = eq_435 & eq_428 & eq_429;
  assign __state_7_constant_bits_6_width_2 = 2'h0;
  assign __state_7_variable_bits_4_width_2 = ____state_7[3:2];
  assign __state_7_constant_bits_6_width_2__1 = 2'h0;
  assign __state_7_variable_bits_0_width_2 = ____state_7[1:0];
  assign concat_474 = {~eq_435 & p0_stage_done, eq_435 & p0_stage_done};
  assign new_delay = ____state_8 + 4'h1;
  assign concat_482 = {~eq_435 & p0_stage_done, and_452 & p0_stage_done, and_454 & p0_stage_done};
  assign new_output_slice_reduced = {____state_7[51:4], __state_7_constant_bits_6_width_2, __state_7_variable_bits_4_width_2, __state_7_constant_bits_6_width_2__1, __state_7_variable_bits_0_width_2, 4'h0};
  assign output_out = ____state_7[59:52];
  assign new_first = {____state_1[55:0], user_module__reading_data};
  assign new_second = {____state_2[55:0], ____state_1[63:56]};
  assign new_third = {____state_3[55:0], ____state_2[63:56]};
  assign one_hot_sel_475 = 4'h0 & {4{concat_474[0]}} | new_delay & {4{concat_474[1]}};
  assign one_hot_sel_483 = 60'h000_0000_0000_0005 & {60{concat_482[0]}} | 60'h000_0000_0000_000a & {60{concat_482[1]}} | new_output_slice_reduced & {60{concat_482[2]}};
  assign and_496 = (~eq_435 | and_452 | and_454) & p0_stage_done;
  always_ff @ (posedge clk) begin
    if (reset) begin
      ____state_1 <= 64'h0000_0000_0000_0000;
      ____state_2 <= 64'h0000_0000_0000_0000;
      ____state_3 <= 64'h0000_0000_0000_0000;
      ____state_8 <= 4'h0;
      ____state_7 <= 60'h000_0000_0000_0000;
    end else begin
      ____state_1 <= p0_stage_done ? new_first : ____state_1;
      ____state_2 <= p0_stage_done ? new_second : ____state_2;
      ____state_3 <= p0_stage_done ? new_third : ____state_3;
      ____state_8 <= p0_stage_done ? one_hot_sel_475 : ____state_8;
      ____state_7 <= and_496 ? one_hot_sel_483 : ____state_7;
    end
  end
  assign user_module__output_data = output_out;
  assign user_module__output_valid = user_module__reading_valid;
  assign user_module__reading_ready = p0_stage_done;
endmodule