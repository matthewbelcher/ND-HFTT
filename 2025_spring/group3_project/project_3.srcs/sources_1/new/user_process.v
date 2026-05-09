`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04/30/2025 01:02:26 PM
// Design Name: 
// Module Name: user_process
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module user_process(
    input clk,
    input reset,
    input out_valid,
    input [7:0] out_data,
    output reg readytoread,
    input tx_req,
    output reg [31:0] tx_data
    );
    
    reg [2047:0] buffer;
    reg [6:0] ptr;
    always @(posedge clk or posedge reset) begin
        readytoread <= 1'd1;
        if (reset) begin
            buffer <=  2047'd0;
            ptr <= 8'd0;
            tx_data <= 32'd0;
            
        end
        else begin
            if (out_valid) // if valid data received store in buffer and then shift 8 bits left
               buffer <= (buffer << 8) | out_data;
               readytoread<= 1'd0;
            if (tx_req) begin //// send data out in 32 bit chunks once requested
                //tx_data <= buffer[ptr*32 +:32];
                tx_data <= buffer[2047-ptr*32 -:32];
                ptr <= ptr + 8'd1;
                if (ptr == 8'd64) begin
                    buffer <= 2047'd0;
                    ptr <= 8'd0;
                end
            end
        end
    end
endmodule
