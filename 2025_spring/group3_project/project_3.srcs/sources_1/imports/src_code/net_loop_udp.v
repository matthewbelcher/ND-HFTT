// THIS MODULE DOES EVERYTHING RELATING TO DEVLELOPING THE TRADING SYSTEM 
module net_udp_loop(
    input              clk_200m  ,   
    input              clk_50m  ,  
    input              sys_rst_n , //ϵͳ��λ�źţ��͵�ƽ��Ч 
    //KSZ9031_RGMII�ӿ�   
    output             eth_mdc  ,
    inout              eth_mdio ,     
    input              net_rxc   , //KSZ9031_RGMII��������ʱ��
    input              net_rx_ctl, //KSZ9031RGMII����������Ч�ź�
    input       [3:0]  net_rxd   , //KSZ9031RGMII��������
    output             net_txc   , //KSZ9031RGMII��������ʱ��    
    output             net_tx_ctl, //KSZ9031RGMII���������Ч�ź�
    output      [3:0]  net_txd   , //KSZ9031RGMII�������          
    output             net_rst_n   //KSZ9031оƬ��λ�źţ��͵�ƽ��Ч   
    );

//parameter define THESE ARE HARDCODED BOARD AND DESTINATION ADDRESSES
parameter  IDELAY_VALUE = 0;
parameter  BOARD_MAC = 48'h99_00_33_11_00_00;     
parameter  BOARD_IP  = {8'd192,8'd168,8'd1,8'd10};  
parameter  DES_MAC   = 48'hff_ff_ff_ff_ff_ff;       
parameter  DES_IP    = {8'd192,8'd168,8'd1,8'd100};  

//wire define
            
wire          gmii_rx_clk; //GMII����ʱ��   //Scaled down clk to properly time when 8 bit words can be read from the 4 bit inputs 
wire          gmii_rx_en ; //GMII����������Ч�ź� // 8bit packet is ready to read
wire  [7:0]   gmii_rxd   ; //GMII�������� // 8 bit packet in data 
wire          gmii_tx_clk; //GMII����ʱ��  // Scaled down clk to properly time 8 bit transmission out 4 bits at a time 
wire          gmii_tx_en ; //GMII��������ʹ���ź� // 8 bit packet ready to transmit signal 
wire  [7:0]   gmii_txd   ; //GMII�������� // 8 bit packet out data 

wire          gmii_tx_en_dummy ; //GMII��������ʹ���ź�  //Ignore 
wire  [7:0]   gmii_txd_dummy   ; //GMII��������         //Ignore

wire          arp_gmii_tx_en; //ARP GMII���������Ч�ź�  //gmii 8 bit packet ready to transmit to arp module  
wire  [7:0]   arp_gmii_txd  ; //ARP GMII�������         // gmii arp request specific 8 bit data ready to transmit to arp module 
wire          arp_rx_done   ; //ARP��������ź�           // finished reception of arp specific packet 
wire          arp_rx_type   ; //ARP�������� 0:����  1:Ӧ��   
wire  [47:0]  src_mac       ; //���յ�Ŀ��MAC��ַ      // source MAC Address
wire  [31:0]  src_ip        ; //���յ�Ŀ��IP��ַ       // source IP address
wire          arp_tx_en     ; //ARP����ʹ���ź�       // 
wire          arp_tx_type   ; //ARP�������� 0:����  1:Ӧ��
wire  [47:0]  des_mac       ; //���͵�Ŀ��MAC��ַ
wire  [31:0]  des_ip        ; //���͵�Ŀ��IP��ַ   
wire          arp_tx_done   ; //ARP��������ź�

wire          udp_gmii_tx_en; //UDP GMII���������Ч�ź� 
wire  [7:0]   udp_gmii_txd  ; //UDP GMII�������
wire          rec_pkt_done  ; //UDP�������ݽ�������ź�
wire          rec_en        ; //UDP���յ�����ʹ���ź�
wire  [63:0]  rec_data      ; //UDP���յ�����
wire  [15:0]  rec_byte_num  ; //UDP���յ���Ч�ֽ��� ��λ:byte 
wire  [15:0]  tx_byte_num   ; //UDP���͵���Ч�ֽ��� ��λ:byte 
wire          udp_tx_done   ; //UDP��������ź�
wire          tx_req        ; //UDP�����������ź�
wire  [31:0]  tx_data       ; //UDP����������
wire          payload_done;


wire tx_start_en;
wire dummy;
wire dummy_1;
wire dummy_2;
wire [7:0] dummy_3;
wire dummy_4;
//assign dummy_1 = 1'b1;
///////////////////////main code////////////////////////////////

assign tx_start_en = rec_pkt_done;

//assign tx_data = 32'h12345678;
assign tx_byte_num = 16'd256;

assign des_mac = src_mac;
assign des_ip = src_ip;



//KSZ9031_phy��λ
net_rstn u_net_rstn(
    .clk       (clk_50m       ),
    .sysrstn   (sys_rst_n     ),
    .net_rst_n (net_rst_n     )
);

RTL8211_Config_IP_0 inst_RTL8211_Config_IP_0 (
  .sys_clk(clk_200m),    // input wire sys_clk
  .sys_rstn(net_rst_n),  // input wire sys_rstn
  .eth_mdc(eth_mdc),    // output wire eth_mdc
  .eth_mdio(eth_mdio) // inout wire eth_mdio
);
//GMII�ӿ�תRGMII�ӿ�

// Only need to understand on a high level
// This constructs 8 bit readable data packets from the 4 bit fpga input and formats 8bit output words to 4 bits so they can be properly sent by the board 
gmii_to_rgmii 
    #(
     .IDELAY_VALUE (IDELAY_VALUE)
     )
    u_gmii_to_rgmii(
    .idelay_clk    (clk_200m    ),

    .gmii_rx_clk   (gmii_rx_clk ),
    .gmii_rx_en    (gmii_rx_en  ),
    .gmii_rxd      (gmii_rxd    ),
    .gmii_tx_clk   (gmii_tx_clk ),
    .gmii_tx_en    (gmii_tx_en  ),
    .gmii_txd      (gmii_txd    ),
    
    .rgmii_rxc     (net_rxc     ),
    .rgmii_rx_ctl  (net_rx_ctl  ),
    .rgmii_rxd     (net_rxd     ),
    .rgmii_txc     (net_txc     ),
    .rgmii_tx_ctl  (net_tx_ctl  ),
    .rgmii_txd     (net_txd     )
    );

//ARPͨ��
// Only need to understand on a high level
// This reads in the packet data and determines if it is a ARP request
// If so, this module responds to inidcate to sender the correct MAC address to send to so that the FPGA recieves
arp                                             
   #(
    .BOARD_MAC     (BOARD_MAC),      //��������
    .BOARD_IP      (BOARD_IP ),
    .DES_MAC       (DES_MAC  ),
    .DES_IP        (DES_IP   )
    )
   u_arp(
    .rst_n         (sys_rst_n  ),
                    
    .gmii_rx_clk   (gmii_rx_clk),
    .gmii_rx_en    (gmii_rx_en ),
    .gmii_rxd      (gmii_rxd   ),
    .gmii_tx_clk   (gmii_tx_clk),
    .gmii_tx_en    (arp_gmii_tx_en ),
    .gmii_txd      (arp_gmii_txd),
                    
    .arp_rx_done   (arp_rx_done),
    .arp_rx_type   (arp_rx_type),
    .src_mac       (src_mac    ),
    .src_ip        (src_ip     ),
    .arp_tx_en     (arp_tx_en  ),
    .arp_tx_type   (arp_tx_type),
    .des_mac       (des_mac    ),
    .des_ip        (des_ip     ),
    .tx_done       (arp_tx_done)
    );

//UDPͨ��
// This module is where udp packet processing actually takes place
udp                                             
   #(
    .BOARD_MAC     (BOARD_MAC),      //��������
    .BOARD_IP      (BOARD_IP ),
    .DES_MAC       (DES_MAC  ),
    .DES_IP        (DES_IP   )
    )
   u_udp(
    .rst_n         (sys_rst_n   ),  
    
    .gmii_rx_clk   (gmii_rx_clk ),           
    .gmii_rx_en    (gmii_rx_en  ),         
    .gmii_rxd      (gmii_rxd    ),                   
    .gmii_tx_clk   (gmii_tx_clk ), 
    .gmii_tx_en    (udp_gmii_tx_en),         
    .gmii_txd      (udp_gmii_txd),  

    .rec_pkt_done  (rec_pkt_done),    
    .rec_en        (rec_en      ),     
    .rec_data      (rec_data    ),         
    .rec_byte_num  (rec_byte_num),   
    .tx_start_en   (tx_start_en ),        
    .tx_data       (tx_data     ),         
    .tx_byte_num   (tx_byte_num ),   
    .tx_done       (udp_tx_done ),        
    .tx_req        (tx_req      ),
    .payload_done  (payload_done)           
    ); 


//��̫������ģ��
// determines which module to select signals from as both udp and arp recieve signal in parallel
net_ctrl u_net_ctrl(
    .clk            (gmii_rx_clk),
    .rst_n          (sys_rst_n),

    .arp_rx_done    (arp_rx_done   ),
    .arp_rx_type    (arp_rx_type   ),
    .arp_tx_en      (arp_tx_en     ),
    .arp_tx_type    (arp_tx_type   ),
    .arp_tx_done    (arp_tx_done   ),
    .arp_gmii_tx_en (arp_gmii_tx_en),
    .arp_gmii_txd   (arp_gmii_txd  ),
                     
    .udp_gmii_tx_en (udp_gmii_tx_en),
    .udp_gmii_txd   (udp_gmii_txd  ),
                     
    .gmii_tx_en     (gmii_tx_en    ),
    .gmii_txd       (gmii_txd      )
    );
    
    
    wire dummy_a;
    wire dummy_b;
    wire dummy_c;
    wire dummy_d;
    //Current hls module being tested
  e1less_module(
    .clk    (gmii_rx_clk),
    .reset    (~sys_rst_n),
    .user_module__reading_data  (gmii_rxd), //8 bit data in 
    .user_module__reading_valid  (gmii_rx_en), //data in valid and ready to be received
    .user_module__output_ready   (gmii_rx_en),// a similar flag to reading_valid (hls seems to like generaing the valid and ready flag pair, which our testing has found collectively function the same as an en)
    .user_module__output_data    (dummy_3),// 8 but data out 
    .user_module__output_valid   (dummy_4),// out data valid and ready to be sent
    .user_module__reading_ready  (dummy_2) // out data ready to be sent/read (again this valid and ready pair are functionally a tx_en)

  );
  //takes hls outputs and buffers them  to the udp transmission out payload so debugging is possible 
  user_process(
    .clk    (gmii_rx_clk),
    .reset  (~sys_rst_n),
    .out_valid  (dummy_2), // data can be read flag
    .out_data   (dummy_3), // 8 bit data in
    .tx_data    (tx_data), // 8 bit data out 
    .tx_req     (tx_req),  // 8 bit data out requested
    .readytoread   (dummy_1) // ignore was used for testing requests to e1less goes high when ready to read nother byte
    );

endmodule