`timescale 1ns / 1ps

//This Code is adapted from Puzhi demo files to serve our trading system development on the kintex board
module udp_top(
    // basic system signals 
    input              sys_clk_p   , //ϵͳʱ��
    input              sys_clk_n   , //ϵͳʱ��    
    input              sys_rst_n , //ϵͳ��λ�źţ��͵�ƽ��Ч 
    //KSZ9031_RGMII�ӿ�   
    // ethernet reception/ack pins
    output             eth1_mdc  ,
    inout              eth1_mdio ,   
    // reception signals  
    input              net_rxc   , //KSZ9031_RGMII��������ʱ��
    input              net_rx_ctl, //KSZ9031RGMII����������Ч�ź�
    // 4bit per clk cylce data read in
    input       [3:0]  net_rxd   , //KSZ9031RGMII��������
    // transmition signals
    output             net_txc   , //KSZ9031RGMII��������ʱ��    
    output             net_tx_ctl, //KSZ9031RGMII���������Ч�ź�
    // 4bit per clk cycle data out
    output      [3:0]  net_txd   , //KSZ9031RGMII�������        
    // system has reset signal  
    output             net_rst_n  //KSZ9031оƬ��λ�źţ��͵�ƽ��Ч 
    );
    
wire    clk_200m; 
wire    clk_50m;    
    //MMCM/PLL
    // Clock scalling module NOT NECESSARY TO TOUCH 
clk_wiz_0 u_clk_wiz
(
    .clk_in1_p(sys_clk_p),    
    .clk_in1_n(sys_clk_n),    
    .clk_out1  (clk_200m  ),   
    .clk_out2  (clk_50m  ),     
    .reset     (~sys_rst_n)
);

// DATA  properly timed  NOT NECESSARY TO TOUC
(* IODELAY_GROUP = "rgmii_delay" *) 
IDELAYCTRL  IDELAYCTRL_inst (
    .RDY(),                      // 1-bit output: Ready output
    .REFCLK(clk_200m),         // 1-bit input: Reference clock input
    .RST(1'b0)                   // 1-bit input: Active high reset input
);


// Main udp packet processing module ALL FOCUS SHOULD BE HERE 
net_udp_loop  net_udp_loop_inst1(
   .clk_200m (clk_200m ) ,   
   .clk_50m  (clk_50m  ) ,  
   .sys_rst_n(sys_rst_n) , //ϵͳ��λ�źţ��͵�ƽ��Ч 
    //KSZ9031_RGMII�ӿ�   
    .eth_mdc   (eth1_mdc),    // output wire eth_mdc
    .eth_mdio  (eth1_mdio), // inout wire eth_mdio    
    .net_rxc   (net_rxc   ), //KSZ9031_RGMII��������ʱ��
    .net_rx_ctl(net_rx_ctl), //KSZ9031RGMII����������Ч�ź�
    .net_rxd   (net_rxd   ), //KSZ9031RGMII��������
    .net_txc   (net_txc   ), //KSZ9031RGMII��������ʱ��    
    .net_tx_ctl(net_tx_ctl), //KSZ9031RGMII���������Ч�ź�
    .net_txd   (net_txd   ), //KSZ9031RGMII�������          
    .net_rst_n (net_rst_n )  //KSZ9031оƬ��λ�źţ��͵�ƽ��Ч   
    );    
endmodule
