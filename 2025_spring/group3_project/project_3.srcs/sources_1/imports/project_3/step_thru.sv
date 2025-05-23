module step_thru(
  input wire clk,
  input wire reset,
  input wire [7:0] user_module__reading_data,
  input wire user_module__reading_valid,
  input wire user_module__output_ready,
  output wire [7:0] user_module__output_data,
  output wire user_module__output_valid,
  output wire user_module__reading_ready
);
  wire [63:0] ____state_2_init[7];
  assign ____state_2_init = '{64'h0000_0000_0000_0000, 64'h0000_0000_0000_0000, 64'h0000_0000_0000_0000, 64'h0000_0000_0000_0000, 64'h0000_0000_0000_0000, 64'h0000_0000_0000_0000, 64'h0000_0000_0000_0000};
  wire [2:0] literal_2736[5];
  assign literal_2736 = '{3'h0, 3'h0, 3'h0, 3'h0, 3'h7};
  wire [63:0] literal_2837[7];
  assign literal_2837 = '{64'h0000_0000_0000_0000, 64'h0000_0000_0000_0000, 64'h0000_0000_0000_0000, 64'h0000_0000_0000_0000, 64'h0000_0000_0000_0000, 64'h0000_0000_0000_0000, 64'h0000_0000_0000_0000};
  reg [3:0] ____state_3;
  reg [3:0] ____state_6;
  reg [3:0] ____state_0;
  reg [63:0] ____state_2[7];
  reg ____state_4;
  reg [63:0] ____state_1;
  reg [63:0] ____state_5;
  reg [63:0] ____state_7;
  wire eq_2657;
  wire or_reduce_2658;
  wire nor_2673;
  wire [63:0] array_index_2674;
  wire or_2691;
  wire [3:0] new_count__4;
  wire eq_2713;
  wire eq_2714;
  wire eq_2701;
  wire eq_2700;
  wire eq_2698;
  wire ne_2716;
  wire eq_2696;
  wire eq_2717;
  wire eq_2718;
  wire eq_2699;
  wire eq_2664;
  wire eq_2697;
  wire [3:0] new_count__1;
  wire and_2739;
  wire or_2985;
  wire and_2746;
  wire [3:0] new_count;
  wire nor_2682;
  wire and_2796;
  wire p0_stage_done;
  wire and_2775;
  wire and_2776;
  wire and_2777;
  wire nor_2789;
  wire and_2778;
  wire and_2779;
  wire and_2780;
  wire nor_2790;
  wire and_2781;
  wire and_2782;
  wire and_2791;
  wire and_2783;
  wire and_2784;
  wire and_2785;
  wire nor_2792;
  wire nor_2793;
  wire and_2786;
  wire nor_2794;
  wire and_2795;
  wire and_2797;
  wire [63:0] new_first;
  wire [3:0] new_count__8;
  wire nor_2799;
  wire nor_2803;
  wire and_2800;
  wire nor_2804;
  wire nor_2805;
  wire and_2801;
  wire and_2802;
  wire and_2788;
  wire [3:0] new_count__3;
  wire [3:0] new_count__2;
  wire [20:0] concat_2898;
  wire [3:0] new_count__6;
  wire [3:0] new_count__5;
  wire [1:0] concat_2904;
  wire [6:0] concat_2919;
  wire [63:0] array_update_2834[7];
  wire [63:0] array_update_2830[7];
  wire [63:0] new_regs[7];
  wire [63:0] new_regs__1[7];
  wire [63:0] array_update_2824[7];
  wire [63:0] array_update_2819[7];
  wire [4:0] concat_2750;
  wire [22:0] concat_2949;
  wire [63:0] new_output__1;
  wire [63:0] new_output;
  wire [1:0] concat_2955;
  wire [3:0] new_delay__9;
  wire [7:0] output_out;
  wire [3:0] one_hot_sel_2899;
  wire and_2966;
  wire [63:0] one_hot_sel_2905;
  wire [63:0] one_hot_sel_2920[7];
  wire and_2972;
  wire [3:0] one_hot_sel_2807;
  wire nor_2808;
  wire [63:0] one_hot_sel_2950;
  wire and_2977;
  wire [3:0] one_hot_sel_2956;
  wire [63:0] new_order;
  wire and_2982;
  assign eq_2657 = ____state_6 == 4'h7;
  assign or_reduce_2658 = |____state_3[3:1];
  assign nor_2673 = ~(or_reduce_2658 | ____state_3[0]);
  assign array_index_2674 = ____state_2[3'h5];
  assign or_2691 = ~eq_2657 | nor_2673;
  assign new_count__4 = 4'h1;
  assign eq_2713 = ____state_1[55:0] == 56'h55_5555_5555_5555;
  assign eq_2714 = user_module__reading_data == 8'hd5;
  assign eq_2701 = ____state_0 == 4'h2;
  assign eq_2700 = ____state_0 == 4'h4;
  assign eq_2698 = ____state_0 == 4'h5;
  assign ne_2716 = array_index_2674[15:8] != 8'h04;
  assign eq_2696 = ____state_0 == 4'h7;
  assign eq_2717 = ____state_0 == 4'h0;
  assign eq_2718 = ____state_0 == new_count__4;
  assign eq_2699 = ____state_0 == 4'h3;
  assign eq_2664 = ____state_0 == 4'h6;
  assign eq_2697 = ____state_0 == 4'h8;
  assign new_count__1 = 4'h1;
  assign and_2739 = eq_2698 & ~or_2691;
  assign or_2985 = eq_2717 | eq_2718 | eq_2701 | eq_2699 | eq_2700 | eq_2698 | eq_2664 | eq_2696 | eq_2697 | ~eq_2657;
  assign and_2746 = eq_2718 & eq_2657;
  assign new_count = 4'h1;
  assign nor_2682 = ~(____state_3[2] | ____state_3[3]);
  assign and_2796 = eq_2717 & eq_2657 & eq_2713 & eq_2714;
  assign p0_stage_done = user_module__reading_valid & user_module__output_ready;
  assign and_2775 = eq_2717 & ~(eq_2657 & eq_2713 & eq_2714);
  assign and_2776 = eq_2718 & ~or_2691;
  assign and_2777 = eq_2718 & or_2691;
  assign nor_2789 = ~(~eq_2701 | eq_2657);
  assign and_2778 = eq_2701 & eq_2657;
  assign and_2779 = eq_2699 & ~or_2691;
  assign and_2780 = eq_2699 & or_2691;
  assign nor_2790 = ~(~eq_2700 | eq_2657);
  assign and_2781 = eq_2700 & eq_2657;
  assign and_2782 = eq_2698 & or_2691;
  assign and_2791 = eq_2664 & eq_2657 & or_reduce_2658;
  assign and_2783 = eq_2664 & ~(eq_2657 & or_reduce_2658);
  assign and_2784 = and_2739 & ~ne_2716;
  assign and_2785 = and_2739 & ne_2716;
  assign nor_2792 = ~(~eq_2696 | eq_2657);
  assign nor_2793 = ~(~eq_2697 | eq_2657);
  assign and_2786 = eq_2697 & eq_2657;
  assign nor_2794 = ~(~eq_2696 | ~eq_2657 | ____state_4);
  assign and_2795 = eq_2696 & eq_2657 & ____state_4;
  assign and_2797 = and_2746 & nor_2673;
  assign new_first = {____state_1[55:0], user_module__reading_data};
  assign new_count__8 = ____state_3 + new_count;
  assign nor_2799 = ~(~eq_2718 | eq_2657);
  assign nor_2803 = ~(~eq_2699 | eq_2657);
  assign and_2800 = eq_2699 & eq_2657;
  assign nor_2804 = ~(~eq_2698 | eq_2657);
  assign nor_2805 = ~(~eq_2664 | eq_2657);
  assign and_2801 = eq_2664 & eq_2657;
  assign and_2802 = eq_2698 & eq_2657 & nor_2673;
  assign and_2788 = ____state_0 > 4'h5 & ~eq_2664 & ~eq_2696 & ~eq_2697 & ~eq_2657;
  assign new_count__3 = 4'h1;
  assign new_count__2 = 4'h1;
  assign concat_2898 = {and_2796 & p0_stage_done, and_2775 & p0_stage_done, and_2776 & p0_stage_done, and_2777 & p0_stage_done, nor_2789 & p0_stage_done, and_2778 & p0_stage_done, and_2779 & p0_stage_done, and_2780 & p0_stage_done, nor_2790 & p0_stage_done, and_2781 & p0_stage_done, and_2782 & p0_stage_done, and_2791 & p0_stage_done, and_2783 & p0_stage_done, and_2784 & p0_stage_done, and_2785 & p0_stage_done, nor_2792 & p0_stage_done, ~or_2985 & p0_stage_done, nor_2793 & p0_stage_done, and_2786 & p0_stage_done, nor_2794 & p0_stage_done, and_2795 & p0_stage_done};
  assign new_count__6 = 4'h1;
  assign new_count__5 = 4'h1;
  assign concat_2904 = {~or_2985 & p0_stage_done, or_2985 & p0_stage_done};
  assign concat_2919 = {eq_2701 & p0_stage_done, eq_2699 & p0_stage_done, and_2776 & p0_stage_done, and_2797 & p0_stage_done, eq_2700 & p0_stage_done, eq_2698 & p0_stage_done, ~or_2985 & p0_stage_done};
  assign array_update_2834[0] = ____state_2[0];
  assign array_update_2834[1] = ____state_2[1];
  assign array_update_2834[2] = ____state_2[2];
  assign array_update_2834[3] = ____state_2[3];
  assign array_update_2834[4] = ____state_2[4];
  assign array_update_2834[5] = ____state_2[5];
  assign array_update_2834[6] = ~(eq_2657 & ~or_reduce_2658 & ~____state_3[0]) ? ____state_2[3'h6] : new_first;
  assign array_update_2830[0] = ____state_2[0];
  assign array_update_2830[1] = ____state_2[1];
  assign array_update_2830[2] = ____state_2[2];
  assign array_update_2830[3] = ____state_2[3];
  assign array_update_2830[4] = ____state_2[4];
  assign array_update_2830[5] = eq_2657 ? new_first : array_index_2674;
  assign array_update_2830[6] = ____state_2[6];
  assign new_regs[0] = new_first;
  assign new_regs[1] = ____state_2[1];
  assign new_regs[2] = ____state_2[2];
  assign new_regs[3] = ____state_2[3];
  assign new_regs[4] = ____state_2[4];
  assign new_regs[5] = ____state_2[5];
  assign new_regs[6] = ____state_2[6];
  assign new_regs__1[0] = new_count__3 == 4'h0 ? new_first : ____state_2[0];
  assign new_regs__1[1] = new_count__3 == 4'h1 ? new_first : ____state_2[1];
  assign new_regs__1[2] = new_count__3 == 4'h2 ? new_first : ____state_2[2];
  assign new_regs__1[3] = new_count__3 == 4'h3 ? new_first : ____state_2[3];
  assign new_regs__1[4] = new_count__3 == 4'h4 ? new_first : ____state_2[4];
  assign new_regs__1[5] = new_count__3 == 4'h5 ? new_first : ____state_2[5];
  assign new_regs__1[6] = new_count__3 == 4'h6 ? new_first : ____state_2[6];
  assign array_update_2824[0] = ____state_2[0];
  assign array_update_2824[1] = ____state_2[1];
  assign array_update_2824[2] = ____state_2[2];
  assign array_update_2824[3] = eq_2657 ? new_first : ____state_2[3'h3];
  assign array_update_2824[4] = ____state_2[4];
  assign array_update_2824[5] = ____state_2[5];
  assign array_update_2824[6] = ____state_2[6];
  assign array_update_2819[0] = ____state_2[0];
  assign array_update_2819[1] = ____state_2[1];
  assign array_update_2819[2] = eq_2657 ? new_first : ____state_2[3'h2];
  assign array_update_2819[3] = ____state_2[3];
  assign array_update_2819[4] = ____state_2[4];
  assign array_update_2819[5] = ____state_2[5];
  assign array_update_2819[6] = ____state_2[6];
  assign concat_2750 = {eq_2718, eq_2698 | eq_2699, eq_2664, eq_2696 | eq_2700 | eq_2701, eq_2697};
  assign concat_2949 = {and_2796 & p0_stage_done, and_2775 & p0_stage_done, nor_2799 & p0_stage_done, and_2746 & p0_stage_done, nor_2789 & p0_stage_done, and_2778 & p0_stage_done, nor_2803 & p0_stage_done, and_2800 & p0_stage_done, nor_2790 & p0_stage_done, and_2781 & p0_stage_done, nor_2804 & p0_stage_done, nor_2805 & p0_stage_done, and_2801 & p0_stage_done, and_2802 & p0_stage_done, nor_2792 & p0_stage_done, and_2784 & p0_stage_done, and_2785 & p0_stage_done, and_2788 & p0_stage_done, ~or_2985 & p0_stage_done, nor_2793 & p0_stage_done, and_2786 & p0_stage_done, nor_2794 & p0_stage_done, and_2795 & p0_stage_done};
  assign new_output__1 = 64'h5555_5555_5555_555d;
  assign new_output = {____state_5[63:8], 8'h00};
  assign concat_2955 = {~eq_2657 & p0_stage_done, eq_2657 & p0_stage_done};
  assign new_delay__9 = ____state_6 + new_count__2;
  assign output_out = ____state_5[7:0];
  assign one_hot_sel_2899 = 4'h8 & {4{concat_2898[0]}} | 4'hf & {4{concat_2898[1]}} | {1'h1, literal_2736[____state_3 > 4'h4 ? 3'h4 : ____state_3[2:0]]} & {4{concat_2898[2]}} | 4'h8 & {4{concat_2898[3]}} | 4'h0 & {4{concat_2898[4]}} | 4'h7 & {4{concat_2898[5]}} | 4'hf & {4{concat_2898[6]}} | 4'h6 & {4{concat_2898[7]}} | 4'h6 & {4{concat_2898[8]}} | 4'h7 & {4{concat_2898[9]}} | 4'h5 & {4{concat_2898[10]}} | 4'h5 & {4{concat_2898[11]}} | 4'h4 & {4{concat_2898[12]}} | 4'h3 & {4{concat_2898[13]}} | 4'h4 & {4{concat_2898[14]}} | 4'h3 & {4{concat_2898[15]}} | 4'h2 & {4{concat_2898[16]}} | new_count__6 & {4{concat_2898[17]}} | 4'h2 & {4{concat_2898[18]}} | 4'h0 & {4{concat_2898[19]}} | new_count__5 & {4{concat_2898[20]}};
  assign and_2966 = (and_2775 | and_2776 | and_2777 | and_2778 | and_2779 | and_2780 | and_2781 | and_2782 | and_2783 | and_2784 | and_2785 | and_2786 | ~or_2985 | nor_2789 | nor_2790 | and_2791 | nor_2792 | nor_2793 | nor_2794 | and_2795 | and_2796) & p0_stage_done;
  assign one_hot_sel_2905 = new_first & {64{concat_2904[0]}} | 64'h0000_0000_0000_0000 & {64{concat_2904[1]}};
  assign one_hot_sel_2920[0] = literal_2837[0] & {64{concat_2919[0]}} | array_update_2834[0] & {64{concat_2919[1]}} | array_update_2830[0] & {64{concat_2919[2]}} | new_regs[0] & {64{concat_2919[3]}} | new_regs__1[0] & {64{concat_2919[4]}} | array_update_2824[0] & {64{concat_2919[5]}} | array_update_2819[0] & {64{concat_2919[6]}};
  assign one_hot_sel_2920[1] = literal_2837[1] & {64{concat_2919[0]}} | array_update_2834[1] & {64{concat_2919[1]}} | array_update_2830[1] & {64{concat_2919[2]}} | new_regs[1] & {64{concat_2919[3]}} | new_regs__1[1] & {64{concat_2919[4]}} | array_update_2824[1] & {64{concat_2919[5]}} | array_update_2819[1] & {64{concat_2919[6]}};
  assign one_hot_sel_2920[2] = literal_2837[2] & {64{concat_2919[0]}} | array_update_2834[2] & {64{concat_2919[1]}} | array_update_2830[2] & {64{concat_2919[2]}} | new_regs[2] & {64{concat_2919[3]}} | new_regs__1[2] & {64{concat_2919[4]}} | array_update_2824[2] & {64{concat_2919[5]}} | array_update_2819[2] & {64{concat_2919[6]}};
  assign one_hot_sel_2920[3] = literal_2837[3] & {64{concat_2919[0]}} | array_update_2834[3] & {64{concat_2919[1]}} | array_update_2830[3] & {64{concat_2919[2]}} | new_regs[3] & {64{concat_2919[3]}} | new_regs__1[3] & {64{concat_2919[4]}} | array_update_2824[3] & {64{concat_2919[5]}} | array_update_2819[3] & {64{concat_2919[6]}};
  assign one_hot_sel_2920[4] = literal_2837[4] & {64{concat_2919[0]}} | array_update_2834[4] & {64{concat_2919[1]}} | array_update_2830[4] & {64{concat_2919[2]}} | new_regs[4] & {64{concat_2919[3]}} | new_regs__1[4] & {64{concat_2919[4]}} | array_update_2824[4] & {64{concat_2919[5]}} | array_update_2819[4] & {64{concat_2919[6]}};
  assign one_hot_sel_2920[5] = literal_2837[5] & {64{concat_2919[0]}} | array_update_2834[5] & {64{concat_2919[1]}} | array_update_2830[5] & {64{concat_2919[2]}} | new_regs[5] & {64{concat_2919[3]}} | new_regs__1[5] & {64{concat_2919[4]}} | array_update_2824[5] & {64{concat_2919[5]}} | array_update_2819[5] & {64{concat_2919[6]}};
  assign one_hot_sel_2920[6] = literal_2837[6] & {64{concat_2919[0]}} | array_update_2834[6] & {64{concat_2919[1]}} | array_update_2830[6] & {64{concat_2919[2]}} | new_regs[6] & {64{concat_2919[3]}} | new_regs__1[6] & {64{concat_2919[4]}} | array_update_2824[6] & {64{concat_2919[5]}} | array_update_2819[6] & {64{concat_2919[6]}};
  assign and_2972 = (eq_2698 | eq_2699 | eq_2700 | eq_2701 | and_2776 | ~or_2985 | and_2797) & p0_stage_done;
  assign one_hot_sel_2807 = (eq_2657 ? new_count__8 & {4{nor_2682}} : ____state_3) & {4{concat_2750[0]}} | ____state_3 & {4{~eq_2657}} & {4{concat_2750[1]}} | (eq_2657 ? new_count__8 & {4{~or_reduce_2658}} : ____state_3) & {4{concat_2750[2]}} | (eq_2657 ? new_count__8 & {4{nor_2673}} : ____state_3) & {4{concat_2750[3]}} | (eq_2657 ? {3'h0, nor_2673} : ____state_3) & {4{concat_2750[4]}};
  assign nor_2808 = ~(eq_2717 | ~(~eq_2664 | ~eq_2657 | nor_2673 ? ____state_4 : ~(____state_3 == new_count__1 & ~____state_4)));
  assign one_hot_sel_2950 = 64'h0000_0000_0000_0007 & {64{concat_2949[0]}} | 64'h0000_0000_0000_0017 & {64{concat_2949[1]}} | 64'h0000_0000_0000_0008 & {64{concat_2949[2]}} | new_output & {64{concat_2949[3]}} | 64'h0000_0000_0000_0009 & {64{concat_2949[4]}} | new_output & {64{concat_2949[5]}} | 64'h0888_8888_8888_8888 & {64{concat_2949[6]}} | new_output__1 & {64{concat_2949[7]}} | new_output & {64{concat_2949[8]}} | 64'h0000_0000_0000_0006 & {64{concat_2949[9]}} | 64'h0000_0000_0000_0006 & {64{concat_2949[10]}} | new_output & {64{concat_2949[11]}} | new_output & {64{concat_2949[12]}} | 64'h0000_0000_0000_0005 & {64{concat_2949[13]}} | new_output & {64{concat_2949[14]}} | 64'h0000_0000_0000_0004 & {64{concat_2949[15]}} | new_output & {64{concat_2949[16]}} | 64'h0000_0000_0000_0003 & {64{concat_2949[17]}} | new_output & {64{concat_2949[18]}} | 64'h0000_0000_0000_0002 & {64{concat_2949[19]}} | new_output & {64{concat_2949[20]}} | new_first & {64{concat_2949[21]}} | 64'h0000_0000_0000_0001 & {64{concat_2949[22]}};
  assign and_2977 = (and_2746 | and_2775 | and_2778 | and_2781 | and_2784 | and_2785 | and_2786 | ~or_2985 | and_2788 | nor_2789 | nor_2790 | nor_2792 | nor_2793 | nor_2794 | and_2795 | and_2796 | nor_2799 | and_2800 | and_2801 | and_2802 | nor_2803 | nor_2804 | nor_2805) & p0_stage_done;
  assign one_hot_sel_2956 = 4'h0 & {4{concat_2955[0]}} | new_delay__9 & {4{concat_2955[1]}};
  assign new_order = ____state_7 + 64'h0000_0000_0000_0001;
  assign and_2982 = ~(eq_2717 | eq_2718 | eq_2701 | eq_2699 | eq_2700 | eq_2698 | eq_2664 | eq_2696 | ~eq_2697 | ~eq_2657 | nor_2682) & p0_stage_done;
  always_ff @ (posedge clk) begin
    if (reset) begin
      ____state_3 <= 4'h0;
      ____state_6 <= 4'h0;
      ____state_0 <= 4'h0;
      ____state_2 <= ____state_2_init;
      ____state_4 <= 1'h0;
      ____state_1 <= 64'h0000_0000_0000_0000;
      ____state_5 <= 64'h0000_0000_0000_0000;
      ____state_7 <= 64'h0000_0000_0000_0000;
    end else begin
      ____state_3 <= p0_stage_done ? one_hot_sel_2807 : ____state_3;
      ____state_6 <= p0_stage_done ? one_hot_sel_2956 : ____state_6;
      ____state_0 <= and_2966 ? one_hot_sel_2899 : ____state_0;
      ____state_2 <= and_2972 ? one_hot_sel_2920 : ____state_2;
      ____state_4 <= p0_stage_done ? nor_2808 : ____state_4;
      ____state_1 <= p0_stage_done ? one_hot_sel_2905 : ____state_1;
      ____state_5 <= and_2977 ? one_hot_sel_2950 : ____state_5;
      ____state_7 <= and_2982 ? new_order : ____state_7;
    end
  end
  assign user_module__output_data = output_out;
  assign user_module__output_valid = user_module__reading_valid;
  assign user_module__reading_ready = p0_stage_done;
endmodule
