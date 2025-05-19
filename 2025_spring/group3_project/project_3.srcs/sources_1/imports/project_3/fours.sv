module fours_mod(
  input wire clk,
  input wire reset,
  input wire [7:0] user_module__reading_data,
  input wire user_module__reading_valid,
  input wire user_module__output_ready,
  output wire [7:0] user_module__output_data,
  output wire user_module__output_valid,
  output wire user_module__reading_ready
);
  reg __state_machine_state_machine___state_7;
  wire [7:0] output_out__1;
  wire literal_272;
  wire p0_stage_done;
  assign output_out__1 = {8{__state_machine_state_machine___state_7}} & 8'h44;
  assign literal_272 = 1'h1;
  assign p0_stage_done = user_module__reading_valid & user_module__output_ready;
  always_ff @ (posedge clk) begin
    if (reset) begin
      __state_machine_state_machine___state_7 <= 1'h0;
    end else begin
      __state_machine_state_machine___state_7 <= p0_stage_done ? literal_272 : __state_machine_state_machine___state_7;
    end
  end
  assign user_module__output_data = output_out__1;
  assign user_module__output_valid = user_module__reading_valid;
  assign user_module__reading_ready = p0_stage_done;
endmodule