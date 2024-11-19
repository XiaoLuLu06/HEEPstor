// This file was generated from a SpinalHDL module available at https://github.com/tomverbeure/math

// CAUTION: Be careful, the generated modules may have sub-modules with the same name as other Spinal HDL files.
// Please, if you regenerate this file, rename the submodule names to something unique to avoid naming problems. 

// Generator : SpinalHDL v1.10.1    git head : 2527c7c6b0fb0f95e5e1a5722a0be732b364ce43
// Component : Fp32Add
// Git hash  : 628be231b362b875b9d15540aca1df8532a9b73d

module Spinal_HDL_FP32_Add (
  input  wire          io_op_valid,
  input  wire [22:0]   io_op_payload_a_mant,
  input  wire [7:0]    io_op_payload_a_exp,
  input  wire          io_op_payload_a_sign,
  input  wire [22:0]   io_op_payload_b_mant,
  input  wire [7:0]    io_op_payload_b_exp,
  input  wire          io_op_payload_b_sign,
  output wire          io_result_valid,
  output wire [22:0]   io_result_payload_mant,
  output wire [7:0]    io_result_payload_exp,
  output wire          io_result_payload_sign
);

  wire       [24:0]   fixTo_1_dout;
  wire       [23:0]   _zz_n0_mant_a;
  wire       [23:0]   _zz_n0_mant_b;
  wire       [8:0]    _zz_n0_exp_diff_a_b;
  wire       [8:0]    _zz_n0_exp_diff_a_b_1;
  wire       [8:0]    _zz_n0_exp_diff_a_b_2;
  wire       [8:0]    _zz_n0_exp_diff_a_b_3;
  wire       [8:0]    _zz_n0_exp_diff;
  wire       [8:0]    _zz_n0_exp_diff_1;
  wire       [26:0]   _zz_n1_mant_a_adj;
  wire       [26:0]   _zz_n1__mant_b_shift_3;
  wire       [26:0]   _zz_n1__mant_b_shift_4;
  wire       [5:0]    _zz__zz_n1__mant_b_shift;
  wire       [5:0]    _zz__zz_n1__mant_b_shift_1;
  wire       [23:0]   _zz_n1__mant_b_shift_5;
  wire       [31:0]   _zz_n1__mant_b_shift_6;
  wire       [31:0]   _zz_n1__mant_b_shift_7;
  wire       [4:0]    _zz_n1__mant_b_shift_8;
  wire       [5:0]    _zz_n1__mant_b_shift_9;
  wire       [5:0]    _zz_n1__mant_b_shift_10;
  wire       [5:0]    _zz_n1__mant_b_shift_11;
  wire       [5:0]    _zz_n1__mant_b_shift_12;
  wire       [0:0]    _zz_n1__mant_b_shift_13;
  wire       [28:0]   _zz_n3_mant_add;
  wire       [26:0]   _zz__zz_switch_Misc_l241;
  wire       [0:0]    _zz_switch_Misc_l241_64;
  wire       [0:0]    _zz_switch_Misc_l241_65;
  wire       [0:0]    _zz_switch_Misc_l241_66;
  wire       [0:0]    _zz_switch_Misc_l241_1_1;
  wire       [0:0]    _zz_switch_Misc_l241_1_2;
  wire       [0:0]    _zz_switch_Misc_l241_1_3;
  wire       [0:0]    _zz__zz_switch_Misc_l241_9;
  wire       [0:0]    _zz__zz_switch_Misc_l241_9_1;
  wire       [0:0]    _zz__zz_switch_Misc_l241_9_2;
  wire       [0:0]    _zz_switch_Misc_l241_3_1;
  wire       [0:0]    _zz_switch_Misc_l241_3_2;
  wire       [0:0]    _zz_switch_Misc_l241_3_3;
  wire       [0:0]    _zz_switch_Misc_l241_4_1;
  wire       [0:0]    _zz_switch_Misc_l241_4_2;
  wire       [0:0]    _zz_switch_Misc_l241_4_3;
  wire       [0:0]    _zz__zz_switch_Misc_l241_16;
  wire       [0:0]    _zz__zz_switch_Misc_l241_16_1;
  wire       [0:0]    _zz__zz_switch_Misc_l241_16_2;
  wire       [1:0]    _zz__zz_switch_Misc_l241_18;
  wire       [1:0]    _zz__zz_switch_Misc_l241_18_1;
  wire       [1:0]    _zz__zz_switch_Misc_l241_18_2;
  wire       [0:0]    _zz_switch_Misc_l241_7_1;
  wire       [0:0]    _zz_switch_Misc_l241_7_2;
  wire       [0:0]    _zz_switch_Misc_l241_7_3;
  wire       [0:0]    _zz_switch_Misc_l241_8_1;
  wire       [0:0]    _zz_switch_Misc_l241_8_2;
  wire       [0:0]    _zz_switch_Misc_l241_8_3;
  wire       [0:0]    _zz__zz_switch_Misc_l241_26;
  wire       [0:0]    _zz__zz_switch_Misc_l241_26_1;
  wire       [0:0]    _zz__zz_switch_Misc_l241_26_2;
  wire       [0:0]    _zz_switch_Misc_l241_10_1;
  wire       [0:0]    _zz_switch_Misc_l241_10_2;
  wire       [0:0]    _zz_switch_Misc_l241_10_3;
  wire       [0:0]    _zz_switch_Misc_l241_11_1;
  wire       [0:0]    _zz_switch_Misc_l241_11_2;
  wire       [0:0]    _zz_switch_Misc_l241_11_3;
  wire       [0:0]    _zz__zz_switch_Misc_l241_33;
  wire       [0:0]    _zz__zz_switch_Misc_l241_33_1;
  wire       [0:0]    _zz__zz_switch_Misc_l241_33_2;
  wire       [1:0]    _zz__zz_switch_Misc_l241_35;
  wire       [1:0]    _zz__zz_switch_Misc_l241_35_1;
  wire       [1:0]    _zz__zz_switch_Misc_l241_35_2;
  wire       [2:0]    _zz__zz_switch_Misc_l241_37;
  wire       [2:0]    _zz__zz_switch_Misc_l241_37_1;
  wire       [2:0]    _zz__zz_switch_Misc_l241_37_2;
  wire       [0:0]    _zz_switch_Misc_l241_15_1;
  wire       [0:0]    _zz_switch_Misc_l241_15_2;
  wire       [0:0]    _zz_switch_Misc_l241_15_3;
  wire       [0:0]    _zz_switch_Misc_l241_16_1;
  wire       [0:0]    _zz_switch_Misc_l241_16_2;
  wire       [0:0]    _zz_switch_Misc_l241_16_3;
  wire       [0:0]    _zz__zz_switch_Misc_l241_46;
  wire       [0:0]    _zz__zz_switch_Misc_l241_46_1;
  wire       [0:0]    _zz__zz_switch_Misc_l241_46_2;
  wire       [0:0]    _zz_switch_Misc_l241_18_1;
  wire       [0:0]    _zz_switch_Misc_l241_18_2;
  wire       [0:0]    _zz_switch_Misc_l241_18_3;
  wire       [0:0]    _zz_switch_Misc_l241_19_1;
  wire       [0:0]    _zz_switch_Misc_l241_19_2;
  wire       [0:0]    _zz_switch_Misc_l241_19_3;
  wire       [0:0]    _zz__zz_switch_Misc_l241_53;
  wire       [0:0]    _zz__zz_switch_Misc_l241_53_1;
  wire       [0:0]    _zz__zz_switch_Misc_l241_53_2;
  wire       [1:0]    _zz__zz_switch_Misc_l241_55;
  wire       [1:0]    _zz__zz_switch_Misc_l241_55_1;
  wire       [1:0]    _zz__zz_switch_Misc_l241_55_2;
  wire       [0:0]    _zz_switch_Misc_l241_22_1;
  wire       [0:0]    _zz_switch_Misc_l241_22_2;
  wire       [0:0]    _zz_switch_Misc_l241_22_3;
  wire       [0:0]    _zz__zz_switch_Misc_l241_59;
  wire       [0:0]    _zz__zz_switch_Misc_l241_59_1;
  wire       [0:0]    _zz__zz_switch_Misc_l241_60;
  wire       [0:0]    _zz__zz_switch_Misc_l241_60_1;
  wire       [0:0]    _zz__zz_switch_Misc_l241_60_2;
  wire       [2:0]    _zz__zz_switch_Misc_l241_62;
  wire       [2:0]    _zz__zz_switch_Misc_l241_62_1;
  wire       [2:0]    _zz__zz_switch_Misc_l241_62_2;
  wire       [3:0]    _zz__zz_n4__lz;
  wire       [3:0]    _zz__zz_n4__lz_1;
  wire       [3:0]    _zz__zz_n4__lz_2;
  wire       [4:0]    _zz_n4__lz_1;
  wire       [8:0]    _zz_n5_exp_add_m_lz;
  wire       [8:0]    _zz_n5_exp_add_m_lz_1;
  wire       [8:0]    _zz_n5_exp_add_m_lz_2;
  wire       [8:0]    _zz_n5_exp_add_m_lz_3;
  wire       [8:0]    _zz_n5_exp_add_m_lz_4;
  wire       [8:0]    _zz_n5_exp_add_m_lz_5;
  wire       [1:0]    _zz_n5_exp_add_m_lz_6;
  wire       [7:0]    _zz_n5_exp_eq_lz;
  wire       [7:0]    _zz_n5_exp_final;
  wire       [8:0]    _zz_n5_exp_final_1;
  wire       [22:0]   _zz_n5_mant_final;
  wire                n4_isValid;
  wire                n4_n2_sign_add;
  wire                n4_n0_is_inf;
  wire                n4_n0_is_nan;
  wire                n3_isValid;
  wire                n3_n2_sign_add;
  wire       [7:0]    n3_n0_exp_add;
  wire                n3_n0_is_inf;
  wire                n3_n0_is_nan;
  wire                n3_n0_is_zero;
  wire                n2_isValid;
  wire       [7:0]    n2_n0_exp_add;
  wire                n2_n0_is_inf;
  wire                n2_n0_is_nan;
  wire                n2_n0_is_zero;
  wire                n1_isValid;
  wire       [7:0]    n1_n0_exp_add;
  wire                n1_n0_sign_b_swap;
  wire                n1_n0_sign_a_swap;
  wire                n1_n0_is_inf;
  wire                n1_n0_is_nan;
  wire                n1_n0_is_zero;
  wire                n0_isValid;
  wire                n5_valid;
  wire                n4_valid;
  wire                n3_valid;
  wire                n2_valid;
  wire                n1_valid;
  wire                n5_isValid;
  wire                n5_n2_sign_add;
  wire                n5_n0_is_inf;
  wire                n5_n0_is_nan;
  wire       [7:0]    n5_n4_exp_add_adj;
  wire       [4:0]    n5_n4_lz;
  wire       [26:0]   n5_n4_mant_add_adj;
  wire       [26:0]   n4_mant_add_adj;
  wire       [7:0]    n4_exp_add_adj;
  wire       [4:0]    n4_lz;
  wire       [7:0]    n4_n0_exp_add;
  wire       [27:0]   n4_n3_mant_add;
  wire                n4_n0_is_zero;
  wire       [27:0]   n3_mant_add;
  wire       [28:0]   n3_n2_mant_b_opt_inv;
  wire       [28:0]   n3_n2_mant_a_opt_inv;
  wire       [28:0]   n2_mant_b_opt_inv;
  wire       [28:0]   n2_mant_a_opt_inv;
  wire                n2_sign_add;
  wire       [27:0]   n2_n1_mant_b_adj;
  wire       [27:0]   n2_n1_mant_a_adj;
  wire                n2_n0_sign_b_swap;
  wire                n2_n0_sign_a_swap;
  wire       [27:0]   n1_mant_b_adj;
  wire                n1_n0_exp_diff_ovfl;
  wire       [4:0]    n1_n0_exp_diff;
  wire       [23:0]   n1_n0_mant_b_swap;
  wire       [27:0]   n1_mant_a_adj;
  wire       [23:0]   n1_n0_mant_a_swap;
  wire       [23:0]   n0_mant_b_swap;
  wire       [23:0]   n0_mant_a_swap;
  wire       [4:0]    n0_exp_diff;
  wire                n0_exp_diff_ovfl;
  wire       [7:0]    n0_exp_add;
  wire                n0_sign_b_swap;
  wire                n0_sign_a_swap;
  wire                n0_is_inf;
  wire                n0_is_nan;
  wire                n0_is_zero;
  wire                n0_b_is_inf;
  wire                n0_a_is_inf;
  wire                n0_b_is_zero;
  wire                n0_a_is_zero;
  wire       [22:0]   n0_b_mant;
  wire       [7:0]    n0_b_exp;
  wire                n0_b_sign;
  wire       [22:0]   n0_a_mant;
  wire       [7:0]    n0_a_exp;
  wire                n0_a_sign;
  wire                n0_valid;
  wire       [23:0]   n0_mant_a;
  wire       [23:0]   n0_mant_b;
  wire       [8:0]    n0_exp_diff_a_b;
  wire       [7:0]    n0_exp_diff_b_a;
  wire                n0_a_geq_b;
  reg        [27:0]   n1__mant_b_shift;
  wire       [5:0]    _zz_n1__mant_b_shift;
  wire       [5:0]    _zz_n1__mant_b_shift_1;
  wire       [5:0]    _zz_n1__mant_b_shift_2;
  reg                 n2__sign_add;
  reg        [28:0]   n2__mant_a_opt_inv;
  reg        [28:0]   n2__mant_b_opt_inv;
  wire                when_FpxxAdd_l81;
  wire                when_FpxxAdd_l86;
  wire       [26:0]   _zz_switch_Misc_l241;
  wire       [15:0]   _zz_switch_Misc_l241_1;
  wire       [7:0]    _zz_switch_Misc_l241_2;
  wire       [3:0]    _zz_switch_Misc_l241_3;
  wire       [1:0]    _zz_switch_Misc_l241_4;
  wire       [1:0]    switch_Misc_l241;
  reg        [1:0]    _zz_switch_Misc_l241_5;
  wire       [1:0]    _zz_switch_Misc_l241_6;
  wire       [1:0]    switch_Misc_l241_1;
  reg        [1:0]    _zz_switch_Misc_l241_7;
  wire       [1:0]    _zz_switch_Misc_l241_8;
  wire       [1:0]    switch_Misc_l241_2;
  reg        [2:0]    _zz_switch_Misc_l241_9;
  wire       [3:0]    _zz_switch_Misc_l241_10;
  wire       [1:0]    _zz_switch_Misc_l241_11;
  wire       [1:0]    switch_Misc_l241_3;
  reg        [1:0]    _zz_switch_Misc_l241_12;
  wire       [1:0]    _zz_switch_Misc_l241_13;
  wire       [1:0]    switch_Misc_l241_4;
  reg        [1:0]    _zz_switch_Misc_l241_14;
  wire       [1:0]    _zz_switch_Misc_l241_15;
  wire       [1:0]    switch_Misc_l241_5;
  reg        [2:0]    _zz_switch_Misc_l241_16;
  wire       [2:0]    _zz_switch_Misc_l241_17;
  wire       [1:0]    switch_Misc_l241_6;
  reg        [3:0]    _zz_switch_Misc_l241_18;
  wire       [7:0]    _zz_switch_Misc_l241_19;
  wire       [3:0]    _zz_switch_Misc_l241_20;
  wire       [1:0]    _zz_switch_Misc_l241_21;
  wire       [1:0]    switch_Misc_l241_7;
  reg        [1:0]    _zz_switch_Misc_l241_22;
  wire       [1:0]    _zz_switch_Misc_l241_23;
  wire       [1:0]    switch_Misc_l241_8;
  reg        [1:0]    _zz_switch_Misc_l241_24;
  wire       [1:0]    _zz_switch_Misc_l241_25;
  wire       [1:0]    switch_Misc_l241_9;
  reg        [2:0]    _zz_switch_Misc_l241_26;
  wire       [3:0]    _zz_switch_Misc_l241_27;
  wire       [1:0]    _zz_switch_Misc_l241_28;
  wire       [1:0]    switch_Misc_l241_10;
  reg        [1:0]    _zz_switch_Misc_l241_29;
  wire       [1:0]    _zz_switch_Misc_l241_30;
  wire       [1:0]    switch_Misc_l241_11;
  reg        [1:0]    _zz_switch_Misc_l241_31;
  wire       [1:0]    _zz_switch_Misc_l241_32;
  wire       [1:0]    switch_Misc_l241_12;
  reg        [2:0]    _zz_switch_Misc_l241_33;
  wire       [2:0]    _zz_switch_Misc_l241_34;
  wire       [1:0]    switch_Misc_l241_13;
  reg        [3:0]    _zz_switch_Misc_l241_35;
  wire       [3:0]    _zz_switch_Misc_l241_36;
  wire       [1:0]    switch_Misc_l241_14;
  reg        [4:0]    _zz_switch_Misc_l241_37;
  wire       [10:0]   _zz_switch_Misc_l241_38;
  wire       [7:0]    _zz_switch_Misc_l241_39;
  wire       [3:0]    _zz_switch_Misc_l241_40;
  wire       [1:0]    _zz_switch_Misc_l241_41;
  wire       [1:0]    switch_Misc_l241_15;
  reg        [1:0]    _zz_switch_Misc_l241_42;
  wire       [1:0]    _zz_switch_Misc_l241_43;
  wire       [1:0]    switch_Misc_l241_16;
  reg        [1:0]    _zz_switch_Misc_l241_44;
  wire       [1:0]    _zz_switch_Misc_l241_45;
  wire       [1:0]    switch_Misc_l241_17;
  reg        [2:0]    _zz_switch_Misc_l241_46;
  wire       [3:0]    _zz_switch_Misc_l241_47;
  wire       [1:0]    _zz_switch_Misc_l241_48;
  wire       [1:0]    switch_Misc_l241_18;
  reg        [1:0]    _zz_switch_Misc_l241_49;
  wire       [1:0]    _zz_switch_Misc_l241_50;
  wire       [1:0]    switch_Misc_l241_19;
  reg        [1:0]    _zz_switch_Misc_l241_51;
  wire       [1:0]    _zz_switch_Misc_l241_52;
  wire       [1:0]    switch_Misc_l241_20;
  reg        [2:0]    _zz_switch_Misc_l241_53;
  wire       [2:0]    _zz_switch_Misc_l241_54;
  wire       [1:0]    switch_Misc_l241_21;
  reg        [3:0]    _zz_switch_Misc_l241_55;
  wire       [2:0]    _zz_switch_Misc_l241_56;
  wire       [1:0]    _zz_switch_Misc_l241_57;
  wire       [1:0]    switch_Misc_l241_22;
  reg        [1:0]    _zz_switch_Misc_l241_58;
  wire       [1:0]    _zz_switch_Misc_l241_59;
  wire       [1:0]    switch_Misc_l241_23;
  reg        [2:0]    _zz_switch_Misc_l241_60;
  wire       [3:0]    _zz_switch_Misc_l241_61;
  wire       [1:0]    switch_Misc_l241_24;
  reg        [4:0]    _zz_switch_Misc_l241_62;
  wire       [4:0]    _zz_switch_Misc_l241_63;
  wire       [1:0]    switch_Misc_l241_25;
  reg        [5:0]    _zz_n4__lz;
  reg        [4:0]    n4__lz;
  reg        [7:0]    n4__exp_add_adj;
  reg        [26:0]   n4__mant_add_adj;
  wire                when_FpxxAdd_l115;
  reg                 n5_sign_final;
  reg        [7:0]    n5_exp_final;
  wire       [26:0]   n5_mant_renormed;
  reg        [22:0]   n5_mant_final;
  wire       [8:0]    n5_exp_add_m_lz;
  wire                n5_exp_eq_lz;
  wire                when_FpxxAdd_l152;

  assign _zz_n0_mant_a = {1'd0, n0_a_mant};
  assign _zz_n0_mant_b = {1'd0, n0_b_mant};
  assign _zz_n0_exp_diff_a_b = _zz_n0_exp_diff_a_b_1;
  assign _zz_n0_exp_diff_a_b_1 = {1'd0, n0_a_exp};
  assign _zz_n0_exp_diff_a_b_2 = _zz_n0_exp_diff_a_b_3;
  assign _zz_n0_exp_diff_a_b_3 = {1'd0, n0_b_exp};
  assign _zz_n0_exp_diff = (n0_a_geq_b ? n0_exp_diff_a_b : _zz_n0_exp_diff_1);
  assign _zz_n0_exp_diff_1 = {1'd0, n0_exp_diff_b_a};
  assign _zz_n1_mant_a_adj = ({3'd0,n1_n0_mant_a_swap} <<< 2'd3);
  assign _zz_n1__mant_b_shift_3 = (_zz_n1__mant_b_shift_4 >>> n1_n0_exp_diff);
  assign _zz_n1__mant_b_shift_4 = ({3'd0,n1_n0_mant_b_swap} <<< 2'd3);
  assign _zz__zz_n1__mant_b_shift = ($signed(_zz__zz_n1__mant_b_shift_1) - $signed(6'h03));
  assign _zz__zz_n1__mant_b_shift_1 = {1'b0,n1_n0_exp_diff};
  assign _zz_n1__mant_b_shift_6 = (_zz_n1__mant_b_shift_7 - 32'h00000001);
  assign _zz_n1__mant_b_shift_5 = _zz_n1__mant_b_shift_6[23:0];
  assign _zz_n1__mant_b_shift_7 = ({31'd0,1'b1} <<< _zz_n1__mant_b_shift_8);
  assign _zz_n1__mant_b_shift_9 = (_zz_n1__mant_b_shift_10 + _zz_n1__mant_b_shift_12);
  assign _zz_n1__mant_b_shift_8 = _zz_n1__mant_b_shift_9[4:0];
  assign _zz_n1__mant_b_shift_10 = (_zz_n1__mant_b_shift_2[5] ? _zz_n1__mant_b_shift_11 : _zz_n1__mant_b_shift_2);
  assign _zz_n1__mant_b_shift_11 = (~ _zz_n1__mant_b_shift_2);
  assign _zz_n1__mant_b_shift_13 = _zz_n1__mant_b_shift_2[5];
  assign _zz_n1__mant_b_shift_12 = {5'd0, _zz_n1__mant_b_shift_13};
  assign _zz_n3_mant_add = (n3_n2_mant_a_opt_inv + n3_n2_mant_b_opt_inv);
  assign _zz__zz_switch_Misc_l241 = n4_n3_mant_add[26:0];
  assign _zz_switch_Misc_l241_64 = _zz_switch_Misc_l241_4[1 : 1];
  assign _zz_switch_Misc_l241_65 = _zz_switch_Misc_l241_66;
  assign _zz_switch_Misc_l241_66 = _zz_switch_Misc_l241_4[0:0];
  assign _zz_switch_Misc_l241_1_1 = _zz_switch_Misc_l241_6[1 : 1];
  assign _zz_switch_Misc_l241_1_2 = _zz_switch_Misc_l241_1_3;
  assign _zz_switch_Misc_l241_1_3 = _zz_switch_Misc_l241_6[0:0];
  assign _zz__zz_switch_Misc_l241_9 = _zz_switch_Misc_l241_5[0:0];
  assign _zz__zz_switch_Misc_l241_9_1 = _zz_switch_Misc_l241_8[0:0];
  assign _zz__zz_switch_Misc_l241_9_2 = _zz_switch_Misc_l241_5[0:0];
  assign _zz_switch_Misc_l241_3_1 = _zz_switch_Misc_l241_11[1 : 1];
  assign _zz_switch_Misc_l241_3_2 = _zz_switch_Misc_l241_3_3;
  assign _zz_switch_Misc_l241_3_3 = _zz_switch_Misc_l241_11[0:0];
  assign _zz_switch_Misc_l241_4_1 = _zz_switch_Misc_l241_13[1 : 1];
  assign _zz_switch_Misc_l241_4_2 = _zz_switch_Misc_l241_4_3;
  assign _zz_switch_Misc_l241_4_3 = _zz_switch_Misc_l241_13[0:0];
  assign _zz__zz_switch_Misc_l241_16 = _zz_switch_Misc_l241_12[0:0];
  assign _zz__zz_switch_Misc_l241_16_1 = _zz_switch_Misc_l241_15[0:0];
  assign _zz__zz_switch_Misc_l241_16_2 = _zz_switch_Misc_l241_12[0:0];
  assign _zz__zz_switch_Misc_l241_18 = _zz_switch_Misc_l241_9[1:0];
  assign _zz__zz_switch_Misc_l241_18_1 = _zz_switch_Misc_l241_17[1:0];
  assign _zz__zz_switch_Misc_l241_18_2 = _zz_switch_Misc_l241_9[1:0];
  assign _zz_switch_Misc_l241_7_1 = _zz_switch_Misc_l241_21[1 : 1];
  assign _zz_switch_Misc_l241_7_2 = _zz_switch_Misc_l241_7_3;
  assign _zz_switch_Misc_l241_7_3 = _zz_switch_Misc_l241_21[0:0];
  assign _zz_switch_Misc_l241_8_1 = _zz_switch_Misc_l241_23[1 : 1];
  assign _zz_switch_Misc_l241_8_2 = _zz_switch_Misc_l241_8_3;
  assign _zz_switch_Misc_l241_8_3 = _zz_switch_Misc_l241_23[0:0];
  assign _zz__zz_switch_Misc_l241_26 = _zz_switch_Misc_l241_22[0:0];
  assign _zz__zz_switch_Misc_l241_26_1 = _zz_switch_Misc_l241_25[0:0];
  assign _zz__zz_switch_Misc_l241_26_2 = _zz_switch_Misc_l241_22[0:0];
  assign _zz_switch_Misc_l241_10_1 = _zz_switch_Misc_l241_28[1 : 1];
  assign _zz_switch_Misc_l241_10_2 = _zz_switch_Misc_l241_10_3;
  assign _zz_switch_Misc_l241_10_3 = _zz_switch_Misc_l241_28[0:0];
  assign _zz_switch_Misc_l241_11_1 = _zz_switch_Misc_l241_30[1 : 1];
  assign _zz_switch_Misc_l241_11_2 = _zz_switch_Misc_l241_11_3;
  assign _zz_switch_Misc_l241_11_3 = _zz_switch_Misc_l241_30[0:0];
  assign _zz__zz_switch_Misc_l241_33 = _zz_switch_Misc_l241_29[0:0];
  assign _zz__zz_switch_Misc_l241_33_1 = _zz_switch_Misc_l241_32[0:0];
  assign _zz__zz_switch_Misc_l241_33_2 = _zz_switch_Misc_l241_29[0:0];
  assign _zz__zz_switch_Misc_l241_35 = _zz_switch_Misc_l241_26[1:0];
  assign _zz__zz_switch_Misc_l241_35_1 = _zz_switch_Misc_l241_34[1:0];
  assign _zz__zz_switch_Misc_l241_35_2 = _zz_switch_Misc_l241_26[1:0];
  assign _zz__zz_switch_Misc_l241_37 = _zz_switch_Misc_l241_18[2:0];
  assign _zz__zz_switch_Misc_l241_37_1 = _zz_switch_Misc_l241_36[2:0];
  assign _zz__zz_switch_Misc_l241_37_2 = _zz_switch_Misc_l241_18[2:0];
  assign _zz_switch_Misc_l241_15_1 = _zz_switch_Misc_l241_41[1 : 1];
  assign _zz_switch_Misc_l241_15_2 = _zz_switch_Misc_l241_15_3;
  assign _zz_switch_Misc_l241_15_3 = _zz_switch_Misc_l241_41[0:0];
  assign _zz_switch_Misc_l241_16_1 = _zz_switch_Misc_l241_43[1 : 1];
  assign _zz_switch_Misc_l241_16_2 = _zz_switch_Misc_l241_16_3;
  assign _zz_switch_Misc_l241_16_3 = _zz_switch_Misc_l241_43[0:0];
  assign _zz__zz_switch_Misc_l241_46 = _zz_switch_Misc_l241_42[0:0];
  assign _zz__zz_switch_Misc_l241_46_1 = _zz_switch_Misc_l241_45[0:0];
  assign _zz__zz_switch_Misc_l241_46_2 = _zz_switch_Misc_l241_42[0:0];
  assign _zz_switch_Misc_l241_18_1 = _zz_switch_Misc_l241_48[1 : 1];
  assign _zz_switch_Misc_l241_18_2 = _zz_switch_Misc_l241_18_3;
  assign _zz_switch_Misc_l241_18_3 = _zz_switch_Misc_l241_48[0:0];
  assign _zz_switch_Misc_l241_19_1 = _zz_switch_Misc_l241_50[1 : 1];
  assign _zz_switch_Misc_l241_19_2 = _zz_switch_Misc_l241_19_3;
  assign _zz_switch_Misc_l241_19_3 = _zz_switch_Misc_l241_50[0:0];
  assign _zz__zz_switch_Misc_l241_53 = _zz_switch_Misc_l241_49[0:0];
  assign _zz__zz_switch_Misc_l241_53_1 = _zz_switch_Misc_l241_52[0:0];
  assign _zz__zz_switch_Misc_l241_53_2 = _zz_switch_Misc_l241_49[0:0];
  assign _zz__zz_switch_Misc_l241_55 = _zz_switch_Misc_l241_46[1:0];
  assign _zz__zz_switch_Misc_l241_55_1 = _zz_switch_Misc_l241_54[1:0];
  assign _zz__zz_switch_Misc_l241_55_2 = _zz_switch_Misc_l241_46[1:0];
  assign _zz_switch_Misc_l241_22_1 = _zz_switch_Misc_l241_57[1 : 1];
  assign _zz_switch_Misc_l241_22_2 = _zz_switch_Misc_l241_22_3;
  assign _zz_switch_Misc_l241_22_3 = _zz_switch_Misc_l241_57[0:0];
  assign _zz__zz_switch_Misc_l241_59 = _zz__zz_switch_Misc_l241_59_1;
  assign _zz__zz_switch_Misc_l241_59_1 = _zz_switch_Misc_l241_56[0:0];
  assign _zz__zz_switch_Misc_l241_60 = _zz_switch_Misc_l241_58[0:0];
  assign _zz__zz_switch_Misc_l241_60_1 = _zz_switch_Misc_l241_59[0:0];
  assign _zz__zz_switch_Misc_l241_60_2 = _zz_switch_Misc_l241_58[0:0];
  assign _zz__zz_switch_Misc_l241_62 = _zz_switch_Misc_l241_55[2:0];
  assign _zz__zz_switch_Misc_l241_62_1 = _zz_switch_Misc_l241_61[2:0];
  assign _zz__zz_switch_Misc_l241_62_2 = _zz_switch_Misc_l241_55[2:0];
  assign _zz__zz_n4__lz = _zz_switch_Misc_l241_37[3:0];
  assign _zz__zz_n4__lz_1 = _zz_switch_Misc_l241_63[3:0];
  assign _zz__zz_n4__lz_2 = _zz_switch_Misc_l241_37[3:0];
  assign _zz_n4__lz_1 = _zz_n4__lz[4:0];
  assign _zz_n5_exp_add_m_lz = ($signed(_zz_n5_exp_add_m_lz_1) - $signed(_zz_n5_exp_add_m_lz_3));
  assign _zz_n5_exp_add_m_lz_1 = _zz_n5_exp_add_m_lz_2;
  assign _zz_n5_exp_add_m_lz_2 = {1'd0, n5_n4_exp_add_adj};
  assign _zz_n5_exp_add_m_lz_3 = _zz_n5_exp_add_m_lz_4;
  assign _zz_n5_exp_add_m_lz_4 = {4'd0, n5_n4_lz};
  assign _zz_n5_exp_add_m_lz_6 = {1'b0,fixTo_1_dout[24]};
  assign _zz_n5_exp_add_m_lz_5 = {{7{_zz_n5_exp_add_m_lz_6[1]}}, _zz_n5_exp_add_m_lz_6};
  assign _zz_n5_exp_eq_lz = {3'd0, n5_n4_lz};
  assign _zz_n5_exp_final_1 = n5_exp_add_m_lz;
  assign _zz_n5_exp_final = _zz_n5_exp_final_1[7:0];
  assign _zz_n5_mant_final = fixTo_1_dout[22:0];
  AddFixTo fixTo_1 (
    .din  (n5_mant_renormed[26:0]), //i
    .dout (fixTo_1_dout[24:0]    )  //o
  );
  assign n0_valid = io_op_valid;
  assign n0_a_mant = io_op_payload_a_mant;
  assign n0_a_exp = io_op_payload_a_exp;
  assign n0_a_sign = io_op_payload_a_sign;
  assign n0_b_mant = io_op_payload_b_mant;
  assign n0_b_exp = io_op_payload_b_exp;
  assign n0_b_sign = io_op_payload_b_sign;
  assign n0_a_is_zero = (((n0_a_exp == 8'h00) && (n0_a_mant == 23'h000000)) || (n0_a_exp == 8'h00));
  assign n0_b_is_zero = (((n0_b_exp == 8'h00) && (n0_b_mant == 23'h000000)) || (n0_b_exp == 8'h00));
  assign n0_a_is_inf = ((&n0_a_exp) && (! (|n0_a_mant)));
  assign n0_b_is_inf = ((&n0_b_exp) && (! (|n0_b_mant)));
  assign n0_is_zero = (n0_a_is_zero || n0_b_is_zero);
  assign n0_is_nan = ((((&n0_a_exp) && (|n0_a_mant)) || ((&n0_b_exp) && (|n0_b_mant))) || ((n0_a_is_inf && n0_b_is_inf) && (n0_a_sign != n0_b_sign)));
  assign n0_is_inf = (n0_a_is_inf || n0_b_is_inf);
  assign n0_mant_a = (n0_a_is_zero ? 24'h000000 : (_zz_n0_mant_a | 24'h800000));
  assign n0_mant_b = (n0_b_is_zero ? 24'h000000 : (_zz_n0_mant_b | 24'h800000));
  assign n0_exp_diff_a_b = ($signed(_zz_n0_exp_diff_a_b) - $signed(_zz_n0_exp_diff_a_b_2));
  assign n0_exp_diff_b_a = (n0_b_exp - n0_a_exp);
  assign n0_a_geq_b = ($signed(9'h000) <= $signed(n0_exp_diff_a_b));
  assign n0_sign_a_swap = (n0_a_geq_b ? n0_a_sign : n0_b_sign);
  assign n0_sign_b_swap = (n0_a_geq_b ? n0_b_sign : n0_a_sign);
  assign n0_exp_add = (n0_a_geq_b ? n0_a_exp : n0_b_exp);
  assign n0_exp_diff_ovfl = (n0_a_geq_b ? ($signed(9'h01a) < $signed(n0_exp_diff_a_b)) : (8'h1a < n0_exp_diff_b_a));
  assign n0_exp_diff = _zz_n0_exp_diff[4:0];
  assign n0_mant_a_swap = (n0_a_geq_b ? n0_mant_a : n0_mant_b);
  assign n0_mant_b_swap = (n0_a_geq_b ? n0_mant_b : n0_mant_a);
  assign n1_mant_a_adj = {1'd0, _zz_n1_mant_a_adj};
  always @(*) begin
    n1__mant_b_shift = {1'd0, _zz_n1__mant_b_shift_3};
    n1__mant_b_shift[0] = (|(_zz_n1__mant_b_shift_5 & n1_n0_mant_b_swap));
  end

  assign _zz_n1__mant_b_shift = ($signed(_zz__zz_n1__mant_b_shift) + $signed(6'h01));
  assign _zz_n1__mant_b_shift_1 = (($signed(_zz_n1__mant_b_shift) < $signed(6'h00)) ? 6'h00 : _zz_n1__mant_b_shift);
  assign _zz_n1__mant_b_shift_2 = (($signed(_zz_n1__mant_b_shift_1) == $signed(6'h20)) ? 6'h21 : _zz_n1__mant_b_shift_1);
  assign n1_mant_b_adj = (n1_n0_exp_diff_ovfl ? 28'h0000000 : n1__mant_b_shift);
  assign when_FpxxAdd_l81 = (n2_n0_sign_a_swap == n2_n0_sign_b_swap);
  always @(*) begin
    if(when_FpxxAdd_l81) begin
      n2__sign_add = n2_n0_sign_a_swap;
    end else begin
      if(when_FpxxAdd_l86) begin
        n2__sign_add = n2_n0_sign_a_swap;
      end else begin
        n2__sign_add = n2_n0_sign_b_swap;
      end
    end
  end

  always @(*) begin
    if(when_FpxxAdd_l81) begin
      n2__mant_a_opt_inv = {n2_n1_mant_a_adj,1'b0};
    end else begin
      if(when_FpxxAdd_l86) begin
        n2__mant_a_opt_inv = {n2_n1_mant_a_adj,1'b1};
      end else begin
        n2__mant_a_opt_inv = {(~ n2_n1_mant_a_adj),1'b1};
      end
    end
  end

  always @(*) begin
    if(when_FpxxAdd_l81) begin
      n2__mant_b_opt_inv = {n2_n1_mant_b_adj,1'b0};
    end else begin
      if(when_FpxxAdd_l86) begin
        n2__mant_b_opt_inv = {(~ n2_n1_mant_b_adj),1'b1};
      end else begin
        n2__mant_b_opt_inv = {n2_n1_mant_b_adj,1'b1};
      end
    end
  end

  assign when_FpxxAdd_l86 = (n2_n1_mant_b_adj <= n2_n1_mant_a_adj);
  assign n2_sign_add = n2__sign_add;
  assign n2_mant_a_opt_inv = n2__mant_a_opt_inv;
  assign n2_mant_b_opt_inv = n2__mant_b_opt_inv;
  assign n3_mant_add = _zz_n3_mant_add[28 : 1];
  assign _zz_switch_Misc_l241 = (~ _zz__zz_switch_Misc_l241);
  assign _zz_switch_Misc_l241_1 = _zz_switch_Misc_l241[26 : 11];
  assign _zz_switch_Misc_l241_2 = _zz_switch_Misc_l241_1[15 : 8];
  assign _zz_switch_Misc_l241_3 = _zz_switch_Misc_l241_2[7 : 4];
  assign _zz_switch_Misc_l241_4 = _zz_switch_Misc_l241_3[3 : 2];
  assign switch_Misc_l241 = {_zz_switch_Misc_l241_64[0],_zz_switch_Misc_l241_65[0]};
  always @(*) begin
    case(switch_Misc_l241)
      2'b11 : begin
        _zz_switch_Misc_l241_5 = 2'b10;
      end
      2'b10 : begin
        _zz_switch_Misc_l241_5 = 2'b01;
      end
      default : begin
        _zz_switch_Misc_l241_5 = 2'b00;
      end
    endcase
  end

  assign _zz_switch_Misc_l241_6 = _zz_switch_Misc_l241_3[1:0];
  assign switch_Misc_l241_1 = {_zz_switch_Misc_l241_1_1[0],_zz_switch_Misc_l241_1_2[0]};
  always @(*) begin
    case(switch_Misc_l241_1)
      2'b11 : begin
        _zz_switch_Misc_l241_7 = 2'b10;
      end
      2'b10 : begin
        _zz_switch_Misc_l241_7 = 2'b01;
      end
      default : begin
        _zz_switch_Misc_l241_7 = 2'b00;
      end
    endcase
  end

  assign _zz_switch_Misc_l241_8 = _zz_switch_Misc_l241_7;
  assign switch_Misc_l241_2 = {_zz_switch_Misc_l241_5[1],_zz_switch_Misc_l241_8[1]};
  always @(*) begin
    case(switch_Misc_l241_2)
      2'b11 : begin
        _zz_switch_Misc_l241_9 = {2'b10,_zz__zz_switch_Misc_l241_9};
      end
      2'b10 : begin
        _zz_switch_Misc_l241_9 = {2'b01,_zz__zz_switch_Misc_l241_9_1};
      end
      default : begin
        _zz_switch_Misc_l241_9 = {2'b00,_zz__zz_switch_Misc_l241_9_2};
      end
    endcase
  end

  assign _zz_switch_Misc_l241_10 = _zz_switch_Misc_l241_2[3:0];
  assign _zz_switch_Misc_l241_11 = _zz_switch_Misc_l241_10[3 : 2];
  assign switch_Misc_l241_3 = {_zz_switch_Misc_l241_3_1[0],_zz_switch_Misc_l241_3_2[0]};
  always @(*) begin
    case(switch_Misc_l241_3)
      2'b11 : begin
        _zz_switch_Misc_l241_12 = 2'b10;
      end
      2'b10 : begin
        _zz_switch_Misc_l241_12 = 2'b01;
      end
      default : begin
        _zz_switch_Misc_l241_12 = 2'b00;
      end
    endcase
  end

  assign _zz_switch_Misc_l241_13 = _zz_switch_Misc_l241_10[1:0];
  assign switch_Misc_l241_4 = {_zz_switch_Misc_l241_4_1[0],_zz_switch_Misc_l241_4_2[0]};
  always @(*) begin
    case(switch_Misc_l241_4)
      2'b11 : begin
        _zz_switch_Misc_l241_14 = 2'b10;
      end
      2'b10 : begin
        _zz_switch_Misc_l241_14 = 2'b01;
      end
      default : begin
        _zz_switch_Misc_l241_14 = 2'b00;
      end
    endcase
  end

  assign _zz_switch_Misc_l241_15 = _zz_switch_Misc_l241_14;
  assign switch_Misc_l241_5 = {_zz_switch_Misc_l241_12[1],_zz_switch_Misc_l241_15[1]};
  always @(*) begin
    case(switch_Misc_l241_5)
      2'b11 : begin
        _zz_switch_Misc_l241_16 = {2'b10,_zz__zz_switch_Misc_l241_16};
      end
      2'b10 : begin
        _zz_switch_Misc_l241_16 = {2'b01,_zz__zz_switch_Misc_l241_16_1};
      end
      default : begin
        _zz_switch_Misc_l241_16 = {2'b00,_zz__zz_switch_Misc_l241_16_2};
      end
    endcase
  end

  assign _zz_switch_Misc_l241_17 = _zz_switch_Misc_l241_16;
  assign switch_Misc_l241_6 = {_zz_switch_Misc_l241_9[2],_zz_switch_Misc_l241_17[2]};
  always @(*) begin
    case(switch_Misc_l241_6)
      2'b11 : begin
        _zz_switch_Misc_l241_18 = {2'b10,_zz__zz_switch_Misc_l241_18};
      end
      2'b10 : begin
        _zz_switch_Misc_l241_18 = {2'b01,_zz__zz_switch_Misc_l241_18_1};
      end
      default : begin
        _zz_switch_Misc_l241_18 = {2'b00,_zz__zz_switch_Misc_l241_18_2};
      end
    endcase
  end

  assign _zz_switch_Misc_l241_19 = _zz_switch_Misc_l241_1[7:0];
  assign _zz_switch_Misc_l241_20 = _zz_switch_Misc_l241_19[7 : 4];
  assign _zz_switch_Misc_l241_21 = _zz_switch_Misc_l241_20[3 : 2];
  assign switch_Misc_l241_7 = {_zz_switch_Misc_l241_7_1[0],_zz_switch_Misc_l241_7_2[0]};
  always @(*) begin
    case(switch_Misc_l241_7)
      2'b11 : begin
        _zz_switch_Misc_l241_22 = 2'b10;
      end
      2'b10 : begin
        _zz_switch_Misc_l241_22 = 2'b01;
      end
      default : begin
        _zz_switch_Misc_l241_22 = 2'b00;
      end
    endcase
  end

  assign _zz_switch_Misc_l241_23 = _zz_switch_Misc_l241_20[1:0];
  assign switch_Misc_l241_8 = {_zz_switch_Misc_l241_8_1[0],_zz_switch_Misc_l241_8_2[0]};
  always @(*) begin
    case(switch_Misc_l241_8)
      2'b11 : begin
        _zz_switch_Misc_l241_24 = 2'b10;
      end
      2'b10 : begin
        _zz_switch_Misc_l241_24 = 2'b01;
      end
      default : begin
        _zz_switch_Misc_l241_24 = 2'b00;
      end
    endcase
  end

  assign _zz_switch_Misc_l241_25 = _zz_switch_Misc_l241_24;
  assign switch_Misc_l241_9 = {_zz_switch_Misc_l241_22[1],_zz_switch_Misc_l241_25[1]};
  always @(*) begin
    case(switch_Misc_l241_9)
      2'b11 : begin
        _zz_switch_Misc_l241_26 = {2'b10,_zz__zz_switch_Misc_l241_26};
      end
      2'b10 : begin
        _zz_switch_Misc_l241_26 = {2'b01,_zz__zz_switch_Misc_l241_26_1};
      end
      default : begin
        _zz_switch_Misc_l241_26 = {2'b00,_zz__zz_switch_Misc_l241_26_2};
      end
    endcase
  end

  assign _zz_switch_Misc_l241_27 = _zz_switch_Misc_l241_19[3:0];
  assign _zz_switch_Misc_l241_28 = _zz_switch_Misc_l241_27[3 : 2];
  assign switch_Misc_l241_10 = {_zz_switch_Misc_l241_10_1[0],_zz_switch_Misc_l241_10_2[0]};
  always @(*) begin
    case(switch_Misc_l241_10)
      2'b11 : begin
        _zz_switch_Misc_l241_29 = 2'b10;
      end
      2'b10 : begin
        _zz_switch_Misc_l241_29 = 2'b01;
      end
      default : begin
        _zz_switch_Misc_l241_29 = 2'b00;
      end
    endcase
  end

  assign _zz_switch_Misc_l241_30 = _zz_switch_Misc_l241_27[1:0];
  assign switch_Misc_l241_11 = {_zz_switch_Misc_l241_11_1[0],_zz_switch_Misc_l241_11_2[0]};
  always @(*) begin
    case(switch_Misc_l241_11)
      2'b11 : begin
        _zz_switch_Misc_l241_31 = 2'b10;
      end
      2'b10 : begin
        _zz_switch_Misc_l241_31 = 2'b01;
      end
      default : begin
        _zz_switch_Misc_l241_31 = 2'b00;
      end
    endcase
  end

  assign _zz_switch_Misc_l241_32 = _zz_switch_Misc_l241_31;
  assign switch_Misc_l241_12 = {_zz_switch_Misc_l241_29[1],_zz_switch_Misc_l241_32[1]};
  always @(*) begin
    case(switch_Misc_l241_12)
      2'b11 : begin
        _zz_switch_Misc_l241_33 = {2'b10,_zz__zz_switch_Misc_l241_33};
      end
      2'b10 : begin
        _zz_switch_Misc_l241_33 = {2'b01,_zz__zz_switch_Misc_l241_33_1};
      end
      default : begin
        _zz_switch_Misc_l241_33 = {2'b00,_zz__zz_switch_Misc_l241_33_2};
      end
    endcase
  end

  assign _zz_switch_Misc_l241_34 = _zz_switch_Misc_l241_33;
  assign switch_Misc_l241_13 = {_zz_switch_Misc_l241_26[2],_zz_switch_Misc_l241_34[2]};
  always @(*) begin
    case(switch_Misc_l241_13)
      2'b11 : begin
        _zz_switch_Misc_l241_35 = {2'b10,_zz__zz_switch_Misc_l241_35};
      end
      2'b10 : begin
        _zz_switch_Misc_l241_35 = {2'b01,_zz__zz_switch_Misc_l241_35_1};
      end
      default : begin
        _zz_switch_Misc_l241_35 = {2'b00,_zz__zz_switch_Misc_l241_35_2};
      end
    endcase
  end

  assign _zz_switch_Misc_l241_36 = _zz_switch_Misc_l241_35;
  assign switch_Misc_l241_14 = {_zz_switch_Misc_l241_18[3],_zz_switch_Misc_l241_36[3]};
  always @(*) begin
    case(switch_Misc_l241_14)
      2'b11 : begin
        _zz_switch_Misc_l241_37 = {2'b10,_zz__zz_switch_Misc_l241_37};
      end
      2'b10 : begin
        _zz_switch_Misc_l241_37 = {2'b01,_zz__zz_switch_Misc_l241_37_1};
      end
      default : begin
        _zz_switch_Misc_l241_37 = {2'b00,_zz__zz_switch_Misc_l241_37_2};
      end
    endcase
  end

  assign _zz_switch_Misc_l241_38 = _zz_switch_Misc_l241[10:0];
  assign _zz_switch_Misc_l241_39 = _zz_switch_Misc_l241_38[10 : 3];
  assign _zz_switch_Misc_l241_40 = _zz_switch_Misc_l241_39[7 : 4];
  assign _zz_switch_Misc_l241_41 = _zz_switch_Misc_l241_40[3 : 2];
  assign switch_Misc_l241_15 = {_zz_switch_Misc_l241_15_1[0],_zz_switch_Misc_l241_15_2[0]};
  always @(*) begin
    case(switch_Misc_l241_15)
      2'b11 : begin
        _zz_switch_Misc_l241_42 = 2'b10;
      end
      2'b10 : begin
        _zz_switch_Misc_l241_42 = 2'b01;
      end
      default : begin
        _zz_switch_Misc_l241_42 = 2'b00;
      end
    endcase
  end

  assign _zz_switch_Misc_l241_43 = _zz_switch_Misc_l241_40[1:0];
  assign switch_Misc_l241_16 = {_zz_switch_Misc_l241_16_1[0],_zz_switch_Misc_l241_16_2[0]};
  always @(*) begin
    case(switch_Misc_l241_16)
      2'b11 : begin
        _zz_switch_Misc_l241_44 = 2'b10;
      end
      2'b10 : begin
        _zz_switch_Misc_l241_44 = 2'b01;
      end
      default : begin
        _zz_switch_Misc_l241_44 = 2'b00;
      end
    endcase
  end

  assign _zz_switch_Misc_l241_45 = _zz_switch_Misc_l241_44;
  assign switch_Misc_l241_17 = {_zz_switch_Misc_l241_42[1],_zz_switch_Misc_l241_45[1]};
  always @(*) begin
    case(switch_Misc_l241_17)
      2'b11 : begin
        _zz_switch_Misc_l241_46 = {2'b10,_zz__zz_switch_Misc_l241_46};
      end
      2'b10 : begin
        _zz_switch_Misc_l241_46 = {2'b01,_zz__zz_switch_Misc_l241_46_1};
      end
      default : begin
        _zz_switch_Misc_l241_46 = {2'b00,_zz__zz_switch_Misc_l241_46_2};
      end
    endcase
  end

  assign _zz_switch_Misc_l241_47 = _zz_switch_Misc_l241_39[3:0];
  assign _zz_switch_Misc_l241_48 = _zz_switch_Misc_l241_47[3 : 2];
  assign switch_Misc_l241_18 = {_zz_switch_Misc_l241_18_1[0],_zz_switch_Misc_l241_18_2[0]};
  always @(*) begin
    case(switch_Misc_l241_18)
      2'b11 : begin
        _zz_switch_Misc_l241_49 = 2'b10;
      end
      2'b10 : begin
        _zz_switch_Misc_l241_49 = 2'b01;
      end
      default : begin
        _zz_switch_Misc_l241_49 = 2'b00;
      end
    endcase
  end

  assign _zz_switch_Misc_l241_50 = _zz_switch_Misc_l241_47[1:0];
  assign switch_Misc_l241_19 = {_zz_switch_Misc_l241_19_1[0],_zz_switch_Misc_l241_19_2[0]};
  always @(*) begin
    case(switch_Misc_l241_19)
      2'b11 : begin
        _zz_switch_Misc_l241_51 = 2'b10;
      end
      2'b10 : begin
        _zz_switch_Misc_l241_51 = 2'b01;
      end
      default : begin
        _zz_switch_Misc_l241_51 = 2'b00;
      end
    endcase
  end

  assign _zz_switch_Misc_l241_52 = _zz_switch_Misc_l241_51;
  assign switch_Misc_l241_20 = {_zz_switch_Misc_l241_49[1],_zz_switch_Misc_l241_52[1]};
  always @(*) begin
    case(switch_Misc_l241_20)
      2'b11 : begin
        _zz_switch_Misc_l241_53 = {2'b10,_zz__zz_switch_Misc_l241_53};
      end
      2'b10 : begin
        _zz_switch_Misc_l241_53 = {2'b01,_zz__zz_switch_Misc_l241_53_1};
      end
      default : begin
        _zz_switch_Misc_l241_53 = {2'b00,_zz__zz_switch_Misc_l241_53_2};
      end
    endcase
  end

  assign _zz_switch_Misc_l241_54 = _zz_switch_Misc_l241_53;
  assign switch_Misc_l241_21 = {_zz_switch_Misc_l241_46[2],_zz_switch_Misc_l241_54[2]};
  always @(*) begin
    case(switch_Misc_l241_21)
      2'b11 : begin
        _zz_switch_Misc_l241_55 = {2'b10,_zz__zz_switch_Misc_l241_55};
      end
      2'b10 : begin
        _zz_switch_Misc_l241_55 = {2'b01,_zz__zz_switch_Misc_l241_55_1};
      end
      default : begin
        _zz_switch_Misc_l241_55 = {2'b00,_zz__zz_switch_Misc_l241_55_2};
      end
    endcase
  end

  assign _zz_switch_Misc_l241_56 = _zz_switch_Misc_l241_38[2:0];
  assign _zz_switch_Misc_l241_57 = _zz_switch_Misc_l241_56[2 : 1];
  assign switch_Misc_l241_22 = {_zz_switch_Misc_l241_22_1[0],_zz_switch_Misc_l241_22_2[0]};
  always @(*) begin
    case(switch_Misc_l241_22)
      2'b11 : begin
        _zz_switch_Misc_l241_58 = 2'b10;
      end
      2'b10 : begin
        _zz_switch_Misc_l241_58 = 2'b01;
      end
      default : begin
        _zz_switch_Misc_l241_58 = 2'b00;
      end
    endcase
  end

  assign _zz_switch_Misc_l241_59 = {1'd0, _zz__zz_switch_Misc_l241_59};
  assign switch_Misc_l241_23 = {_zz_switch_Misc_l241_58[1],_zz_switch_Misc_l241_59[1]};
  always @(*) begin
    case(switch_Misc_l241_23)
      2'b11 : begin
        _zz_switch_Misc_l241_60 = {2'b10,_zz__zz_switch_Misc_l241_60};
      end
      2'b10 : begin
        _zz_switch_Misc_l241_60 = {2'b01,_zz__zz_switch_Misc_l241_60_1};
      end
      default : begin
        _zz_switch_Misc_l241_60 = {2'b00,_zz__zz_switch_Misc_l241_60_2};
      end
    endcase
  end

  assign _zz_switch_Misc_l241_61 = {1'd0, _zz_switch_Misc_l241_60};
  assign switch_Misc_l241_24 = {_zz_switch_Misc_l241_55[3],_zz_switch_Misc_l241_61[3]};
  always @(*) begin
    case(switch_Misc_l241_24)
      2'b11 : begin
        _zz_switch_Misc_l241_62 = {2'b10,_zz__zz_switch_Misc_l241_62};
      end
      2'b10 : begin
        _zz_switch_Misc_l241_62 = {2'b01,_zz__zz_switch_Misc_l241_62_1};
      end
      default : begin
        _zz_switch_Misc_l241_62 = {2'b00,_zz__zz_switch_Misc_l241_62_2};
      end
    endcase
  end

  assign _zz_switch_Misc_l241_63 = _zz_switch_Misc_l241_62;
  assign switch_Misc_l241_25 = {_zz_switch_Misc_l241_37[4],_zz_switch_Misc_l241_63[4]};
  always @(*) begin
    case(switch_Misc_l241_25)
      2'b11 : begin
        _zz_n4__lz = {2'b10,_zz__zz_n4__lz};
      end
      2'b10 : begin
        _zz_n4__lz = {2'b01,_zz__zz_n4__lz_1};
      end
      default : begin
        _zz_n4__lz = {2'b00,_zz__zz_n4__lz_2};
      end
    endcase
  end

  always @(*) begin
    n4__lz = (n4_n0_is_zero ? 5'h00 : _zz_n4__lz_1);
    if(when_FpxxAdd_l115) begin
      n4__lz = 5'h00;
    end
  end

  assign when_FpxxAdd_l115 = n4_n3_mant_add[27];
  always @(*) begin
    if(when_FpxxAdd_l115) begin
      n4__mant_add_adj = (n4_n3_mant_add >>> 1'd1);
      n4__mant_add_adj[0] = (n4_n3_mant_add[0] || n4_n3_mant_add[1]);
    end else begin
      n4__mant_add_adj = n4_n3_mant_add[26:0];
    end
  end

  always @(*) begin
    if(when_FpxxAdd_l115) begin
      n4__exp_add_adj = (n4_n0_exp_add + 8'h01);
    end else begin
      n4__exp_add_adj = n4_n0_exp_add;
    end
  end

  assign n4_lz = n4__lz;
  assign n4_exp_add_adj = n4__exp_add_adj;
  assign n4_mant_add_adj = n4__mant_add_adj;
  assign n5_mant_renormed = (n5_n4_mant_add_adj <<< n5_n4_lz);
  assign n5_exp_add_m_lz = ($signed(_zz_n5_exp_add_m_lz) + $signed(_zz_n5_exp_add_m_lz_5));
  assign n5_exp_eq_lz = (n5_n4_exp_add_adj == _zz_n5_exp_eq_lz);
  always @(*) begin
    if(n5_n0_is_nan) begin
      n5_sign_final = 1'b0;
    end else begin
      if(when_FpxxAdd_l152) begin
        n5_sign_final = n5_n2_sign_add;
      end else begin
        n5_sign_final = n5_n2_sign_add;
      end
    end
  end

  always @(*) begin
    if(n5_n0_is_nan) begin
      n5_exp_final = 8'hff;
    end else begin
      if(when_FpxxAdd_l152) begin
        n5_exp_final = 8'hff;
      end else begin
        n5_exp_final = (((n5_n4_lz < 5'h1b) && (! n5_exp_add_m_lz[8])) ? _zz_n5_exp_final : 8'h00);
      end
    end
  end

  always @(*) begin
    if(n5_n0_is_nan) begin
      n5_mant_final = 23'h000000;
      n5_mant_final[22] = 1'b1;
    end else begin
      if(when_FpxxAdd_l152) begin
        n5_mant_final = 23'h000000;
      end else begin
        n5_mant_final = (((! n5_exp_add_m_lz[8]) && (! n5_exp_eq_lz)) ? _zz_n5_mant_final : 23'h000000);
      end
    end
  end

  assign when_FpxxAdd_l152 = (n5_n0_is_inf || (&n5_n4_exp_add_adj));
  assign io_result_payload_sign = n5_sign_final;
  assign io_result_payload_exp = n5_exp_final;
  assign io_result_payload_mant = n5_mant_final;
  assign io_result_valid = n5_isValid;
  assign n1_valid = n0_isValid;
  assign n1_n0_is_zero = n0_is_zero;
  assign n1_n0_is_nan = n0_is_nan;
  assign n1_n0_is_inf = n0_is_inf;
  assign n1_n0_sign_a_swap = n0_sign_a_swap;
  assign n1_n0_sign_b_swap = n0_sign_b_swap;
  assign n1_n0_exp_add = n0_exp_add;
  assign n1_n0_exp_diff_ovfl = n0_exp_diff_ovfl;
  assign n1_n0_exp_diff = n0_exp_diff;
  assign n1_n0_mant_a_swap = n0_mant_a_swap;
  assign n1_n0_mant_b_swap = n0_mant_b_swap;
  assign n2_valid = n1_isValid;
  assign n2_n0_is_zero = n1_n0_is_zero;
  assign n2_n0_is_nan = n1_n0_is_nan;
  assign n2_n0_is_inf = n1_n0_is_inf;
  assign n2_n0_sign_a_swap = n1_n0_sign_a_swap;
  assign n2_n0_sign_b_swap = n1_n0_sign_b_swap;
  assign n2_n0_exp_add = n1_n0_exp_add;
  assign n2_n1_mant_a_adj = n1_mant_a_adj;
  assign n2_n1_mant_b_adj = n1_mant_b_adj;
  assign n3_valid = n2_isValid;
  assign n3_n0_is_zero = n2_n0_is_zero;
  assign n3_n0_is_nan = n2_n0_is_nan;
  assign n3_n0_is_inf = n2_n0_is_inf;
  assign n3_n0_exp_add = n2_n0_exp_add;
  assign n3_n2_sign_add = n2_sign_add;
  assign n3_n2_mant_a_opt_inv = n2_mant_a_opt_inv;
  assign n3_n2_mant_b_opt_inv = n2_mant_b_opt_inv;
  assign n4_valid = n3_isValid;
  assign n4_n0_is_zero = n3_n0_is_zero;
  assign n4_n0_is_nan = n3_n0_is_nan;
  assign n4_n0_is_inf = n3_n0_is_inf;
  assign n4_n0_exp_add = n3_n0_exp_add;
  assign n4_n2_sign_add = n3_n2_sign_add;
  assign n4_n3_mant_add = n3_mant_add;
  assign n5_valid = n4_isValid;
  assign n5_n0_is_nan = n4_n0_is_nan;
  assign n5_n0_is_inf = n4_n0_is_inf;
  assign n5_n2_sign_add = n4_n2_sign_add;
  assign n5_n4_lz = n4_lz;
  assign n5_n4_exp_add_adj = n4_exp_add_adj;
  assign n5_n4_mant_add_adj = n4_mant_add_adj;
  assign n0_isValid = n0_valid;
  assign n1_isValid = n1_valid;
  assign n2_isValid = n2_valid;
  assign n3_isValid = n3_valid;
  assign n4_isValid = n4_valid;
  assign n5_isValid = n5_valid;

endmodule

module AddFixTo (
  input  wire [26:0]   din,
  output wire [24:0]   dout
);

  wire       [24:0]   _zz__zz_dout_1;
  wire       [1:0]    _zz__zz_dout_1_1;
  wire       [25:0]   _zz__zz_dout;
  reg        [24:0]   _zz_dout;
  wire                when_UInt_l238;
  reg        [24:0]   _zz_dout_1;
  wire                when_UInt_l219;

  assign _zz__zz_dout_1_1 = {1'b0,1'b1};
  assign _zz__zz_dout_1 = {23'd0, _zz__zz_dout_1_1};
  assign _zz__zz_dout = ({1'b0,din[26 : 2]} + {1'b0,{24'h000000,1'b1}});
  assign when_UInt_l238 = (! din[3]);
  assign when_UInt_l219 = (din[2] && (|din[1 : 0]));
  always @(*) begin
    if(when_UInt_l219) begin
      _zz_dout_1 = ({1'b0,din[26 : 3]} + _zz__zz_dout_1);
    end else begin
      _zz_dout_1 = {1'b0,din[26 : 3]};
    end
  end

  always @(*) begin
    if(when_UInt_l238) begin
      _zz_dout = _zz_dout_1;
    end else begin
      _zz_dout = (_zz__zz_dout >>> 1'd1);
    end
  end

  assign dout = _zz_dout;

endmodule
