module sequence_mod(
  input wire clk,
  input wire reset,
  input wire [7:0] user_module__reading_data,
  input wire user_module__reading_valid,
  input wire user_module__output_ready,
  output wire [7:0] user_module__output_data,
  output wire user_module__output_valid,
  output wire user_module__reading_ready
);
  reg [63:0] ____state_2;
  reg [63:0] ____state_1;
  reg [63:0] ____state_3;
  reg [3:0] ____state_8;
  reg [55:0] ____state_7;
  wire eq_425;
  wire eq_426;
  wire eq_432;
  wire p0_stage_done;
  wire and_447;
  wire and_449;
  wire [7:0] __state_7_constant_bits_0_width_8;
  wire [1:0] concat_469;
  wire [3:0] new_delay;
  wire [2:0] concat_477;
  wire [55:0] sliced_new_out;
  wire [55:0] sliced_new_output;
  wire [7:0] output_out;
  wire [63:0] new_first;
  wire [63:0] new_second;
  wire [63:0] new_third;
  wire [3:0] one_hot_sel_470;
  wire [55:0] one_hot_sel_478;
  wire and_491;
  assign eq_425 = ____state_2[55:0] == 56'h12_1212_1212_1212;
  assign eq_426 = ____state_1[63:56] == 8'h12;
  assign eq_432 = ____state_8 == 4'h7;
  assign p0_stage_done =   1'b1;//user_module__reading_valid; // & user_module__output_ready;
  assign and_447 = eq_432 & ~(eq_425 & eq_426);
  assign and_449 = eq_432 & eq_425 & eq_426;
  assign __state_7_constant_bits_0_width_8 = 8'h00;
  assign concat_469 = {~eq_432 & p0_stage_done, eq_432 & p0_stage_done};
  assign new_delay = ____state_8 + 4'h1;
  assign concat_477 = {~eq_432 & p0_stage_done, and_447 & p0_stage_done, and_449 & p0_stage_done};
  assign sliced_new_out = {____state_1[23:0], user_module__reading_data, 24'h00_0000};
  assign sliced_new_output = {____state_7[47:0], __state_7_constant_bits_0_width_8};
  assign output_out = ____state_7[55:48];
  assign new_first = {____state_1[55:0], user_module__reading_data};
  assign new_second = {____state_2[55:0], ____state_1[63:56]};
  assign new_third = {____state_3[55:0], ____state_2[63:56]};
  assign one_hot_sel_470 = 4'h0 & {4{concat_469[0]}} | new_delay & {4{concat_469[1]}};
  assign one_hot_sel_478 = sliced_new_out & {56{concat_477[0]}} | 56'h00_0000_0000_0000 & {56{concat_477[1]}} | sliced_new_output & {56{concat_477[2]}};
  assign and_491 = (~eq_432 | and_447 | and_449) & p0_stage_done;
  always_ff @ (posedge clk) begin
    if (reset) begin
      ____state_2 <= 64'h0000_0000_0000_0000;
      ____state_1 <= 64'h0000_0000_0000_0000;
      ____state_3 <= 64'h0000_0000_0000_0000;
      ____state_8 <= 4'h0;
      ____state_7 <= 56'h00_0000_0000_0000;
    end else begin
      ____state_2 <= p0_stage_done ? new_second : ____state_2;
      ____state_1 <= p0_stage_done ? new_first : ____state_1;
      ____state_3 <= p0_stage_done ? new_third : ____state_3;
      ____state_8 <= p0_stage_done ? one_hot_sel_470 : ____state_8;
      ____state_7 <= and_491 ? one_hot_sel_478 : ____state_7;
    end
  end
  assign user_module__output_data = output_out;
  assign user_module__output_valid = user_module__reading_valid;
  assign user_module__reading_ready = p0_stage_done;
endmodule