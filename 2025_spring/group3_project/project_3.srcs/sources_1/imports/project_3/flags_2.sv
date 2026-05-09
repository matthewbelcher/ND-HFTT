module flags2_module(
  input wire clk,
  input wire reset,
  input wire [7:0] user_module__reading_data,
  input wire user_module__reading_valid,
  input wire user_module__valid_data,
  input wire user_module__valid_valid,
  input wire user_module__output_ready,
  input wire user_module__out_val_ready,
  output wire [7:0] user_module__output_data,
  output wire user_module__out_val_data,
  output wire user_module__output_valid,
  output wire user_module__out_val_valid,
  output wire user_module__reading_ready,
  output wire user_module__valid_ready
);
  reg [63:0] ____state_1;
  reg [3:0] ____state_8;
  reg [63:0] ____state_2;
  reg [63:0] ____state_3;
  reg [59:0] ____state_7;
  reg ____state_11;
  reg __user_module__output_data_has_been_sent_reg;
  reg __user_module__out_val_data_has_been_sent_reg;
  wire __user_module__output_data_has_sent_or_is_ready;
  wire __user_module__out_val_data_has_sent_or_is_ready;
  wire eq_494;
  wire eq_495;
  wire eq_500;
  wire p0_all_active_inputs_valid;
  wire p0_all_active_outputs_ready;
  wire and_502;
  wire p0_stage_done;
  wire and_514;
  wire and_515;
  wire __user_module__output_data_not_has_been_sent;
  wire __user_module__out_val_data_not_has_been_sent;
  wire [1:0] __state_7_constant_bits_6_width_2;
  wire [1:0] __state_7_variable_bits_4_width_2;
  wire [1:0] __state_7_constant_bits_6_width_2__1;
  wire [1:0] __state_7_variable_bits_0_width_2;
  wire __user_module__output_data_valid_and_not_has_been_sent;
  wire __user_module__out_val_data_valid_and_not_has_been_sent;
  wire [1:0] concat_552;
  wire [3:0] new_delay;
  wire [2:0] concat_563;
  wire [59:0] new_output_slice_reduced;
  wire __user_module__output_data_valid_and_all_active_outputs_ready;
  wire __user_module__output_data_valid_and_ready_txfr;
  wire __user_module__out_val_data_valid_and_all_active_outputs_ready;
  wire __user_module__out_val_data_valid_and_ready_txfr;
  wire [7:0] output_out;
  wire [63:0] new_first;
  wire [63:0] new_second;
  wire [63:0] new_third;
  wire [3:0] one_hot_sel_553;
  wire and_578;
  wire [59:0] one_hot_sel_564;
  wire and_581;
  wire __user_module__output_data_not_stage_load;
  wire __user_module__output_data_has_been_sent_reg_load_en;
  wire __user_module__out_val_data_not_stage_load;
  wire __user_module__out_val_data_has_been_sent_reg_load_en;
  assign __user_module__output_data_has_sent_or_is_ready = user_module__output_ready | __user_module__output_data_has_been_sent_reg;
  assign __user_module__out_val_data_has_sent_or_is_ready = user_module__out_val_ready | __user_module__out_val_data_has_been_sent_reg;
  assign eq_494 = ____state_1[55:0] == 56'h44_4444_4444_4444;
  assign eq_495 = user_module__reading_data == 8'h44;
  assign eq_500 = ____state_8 == 4'h7;
  assign p0_all_active_inputs_valid = user_module__reading_valid & user_module__valid_valid;
  assign p0_all_active_outputs_ready = __user_module__output_data_has_sent_or_is_ready & __user_module__out_val_data_has_sent_or_is_ready;
  assign and_502 = eq_494 & eq_495 & user_module__valid_data;
  assign p0_stage_done = p0_all_active_inputs_valid & p0_all_active_outputs_ready;
  assign and_514 = eq_500 & ~(eq_494 & eq_495 & user_module__valid_data);
  assign and_515 = eq_500 & and_502;
  assign __user_module__output_data_not_has_been_sent = ~__user_module__output_data_has_been_sent_reg;
  assign __user_module__out_val_data_not_has_been_sent = ~__user_module__out_val_data_has_been_sent_reg;
  assign __state_7_constant_bits_6_width_2 = 2'h0;
  assign __state_7_variable_bits_4_width_2 = ____state_7[3:2];
  assign __state_7_constant_bits_6_width_2__1 = 2'h0;
  assign __state_7_variable_bits_0_width_2 = ____state_7[1:0];
  assign __user_module__output_data_valid_and_not_has_been_sent = p0_all_active_inputs_valid & __user_module__output_data_not_has_been_sent;
  assign __user_module__out_val_data_valid_and_not_has_been_sent = p0_all_active_inputs_valid & __user_module__out_val_data_not_has_been_sent;
  assign concat_552 = {~eq_500 & p0_stage_done, eq_500 & p0_stage_done};
  assign new_delay = ____state_8 + 4'h1;
  assign concat_563 = {~eq_500 & p0_stage_done, and_514 & p0_stage_done, and_515 & p0_stage_done};
  assign new_output_slice_reduced = {____state_7[51:4], __state_7_constant_bits_6_width_2, __state_7_variable_bits_4_width_2, __state_7_constant_bits_6_width_2__1, __state_7_variable_bits_0_width_2, 4'h0};
  assign __user_module__output_data_valid_and_all_active_outputs_ready = p0_all_active_inputs_valid & p0_all_active_outputs_ready;
  assign __user_module__output_data_valid_and_ready_txfr = __user_module__output_data_valid_and_not_has_been_sent & user_module__output_ready;
  assign __user_module__out_val_data_valid_and_all_active_outputs_ready = p0_all_active_inputs_valid & p0_all_active_outputs_ready;
  assign __user_module__out_val_data_valid_and_ready_txfr = __user_module__out_val_data_valid_and_not_has_been_sent & user_module__out_val_ready;
  assign output_out = ____state_7[59:52];
  assign new_first = {____state_1[55:0], user_module__reading_data};
  assign new_second = {____state_2[55:0], ____state_1[63:56]};
  assign new_third = {____state_3[55:0], ____state_2[63:56]};
  assign one_hot_sel_553 = 4'h0 & {4{concat_552[0]}} | new_delay & {4{concat_552[1]}};
  assign and_578 = eq_500 & p0_stage_done;
  assign one_hot_sel_564 = 60'h000_0000_0000_0005 & {60{concat_563[0]}} | 60'h000_0000_0000_000a & {60{concat_563[1]}} | new_output_slice_reduced & {60{concat_563[2]}};
  assign and_581 = (and_514 | and_515 | ~eq_500) & p0_stage_done;
  assign __user_module__output_data_not_stage_load = ~__user_module__output_data_valid_and_all_active_outputs_ready;
  assign __user_module__output_data_has_been_sent_reg_load_en = __user_module__output_data_valid_and_ready_txfr | __user_module__output_data_valid_and_all_active_outputs_ready;
  assign __user_module__out_val_data_not_stage_load = ~__user_module__out_val_data_valid_and_all_active_outputs_ready;
  assign __user_module__out_val_data_has_been_sent_reg_load_en = __user_module__out_val_data_valid_and_ready_txfr | __user_module__out_val_data_valid_and_all_active_outputs_ready;
  always_ff @ (posedge clk) begin
    if (reset) begin
      ____state_1 <= 64'h0000_0000_0000_0000;
      ____state_8 <= 4'h0;
      ____state_2 <= 64'h0000_0000_0000_0000;
      ____state_3 <= 64'h0000_0000_0000_0000;
      ____state_7 <= 60'h000_0000_0000_0000;
      ____state_11 <= 1'h0;
      __user_module__output_data_has_been_sent_reg <= 1'h0;
      __user_module__out_val_data_has_been_sent_reg <= 1'h0;
    end else begin
      ____state_1 <= p0_stage_done ? new_first : ____state_1;
      ____state_8 <= p0_stage_done ? one_hot_sel_553 : ____state_8;
      ____state_2 <= p0_stage_done ? new_second : ____state_2;
      ____state_3 <= p0_stage_done ? new_third : ____state_3;
      ____state_7 <= and_581 ? one_hot_sel_564 : ____state_7;
      ____state_11 <= and_578 ? and_502 : ____state_11;
      __user_module__output_data_has_been_sent_reg <= __user_module__output_data_has_been_sent_reg_load_en ? __user_module__output_data_not_stage_load : __user_module__output_data_has_been_sent_reg;
      __user_module__out_val_data_has_been_sent_reg <= __user_module__out_val_data_has_been_sent_reg_load_en ? __user_module__out_val_data_not_stage_load : __user_module__out_val_data_has_been_sent_reg;
    end
  end
  assign user_module__output_data = output_out;
  assign user_module__out_val_data = ____state_11;
  assign user_module__output_valid = __user_module__output_data_valid_and_not_has_been_sent;
  assign user_module__out_val_valid = __user_module__out_val_data_valid_and_not_has_been_sent;
  assign user_module__reading_ready = p0_stage_done;
  assign user_module__valid_ready = p0_stage_done;
endmodule