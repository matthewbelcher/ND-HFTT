
module net_ctrl(
    input              clk       ,    //ϵͳʱ��
    input              rst_n     ,    //ϵͳ��λ�źţ��͵�ƽ��Ч 
    //ARP��ض˿��ź�                                  
    input              arp_rx_done,   //ARP��������ź�
    input              arp_rx_type,   //ARP�������� 0:����  1:Ӧ��
    output             arp_tx_en,     //ARP����ʹ���ź�
    output             arp_tx_type,   //ARP�������� 0:����  1:Ӧ��
    input              arp_tx_done,   //ARP��������ź�
    input              arp_gmii_tx_en,//ARP GMII���������Ч�ź� 
    input     [7:0]    arp_gmii_txd,  //ARP GMII�������
    //UDP��ض˿��ź�
    input              udp_gmii_tx_en,//UDP GMII���������Ч�ź�  
    input     [7:0]    udp_gmii_txd,  //UDP GMII�������   
    //GMII��������
    output             gmii_tx_en,    //GMII���������Ч�ź� 
    output    [7:0]    gmii_txd       //UDP GMII������� 
    );

//reg define
reg        protocol_sw; //Э���л��ź�

//*****************************************************
//**                    main code
//*****************************************************

assign arp_tx_en = arp_rx_done && (arp_rx_type == 1'b0);
assign arp_tx_type = 1'b1;   //ARP�������͹̶�ΪARPӦ��                                   
assign gmii_tx_en = protocol_sw ? udp_gmii_tx_en : arp_gmii_tx_en;
assign gmii_txd = protocol_sw ? udp_gmii_txd : arp_gmii_txd;

//����ARP����ʹ��/����ź�,�л�GMII����
always @(posedge clk or negedge rst_n) begin
    if(!rst_n)
        protocol_sw <= 1'b1;
    else if(arp_tx_en)   
        protocol_sw <= 1'b0;
    else if(arp_tx_done)
        protocol_sw <= 1'b1;
end

endmodule