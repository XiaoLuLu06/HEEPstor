// This file was generated from a SpinalHDL module available at https://github.com/tomverbeure/math

// CAUTION: Be careful, the generated modules may have sub-modules with the same name as other Spinal HDL files.
// Please, if you regenerate this file, rename the submodule names to something unique to avoid naming problems. 

// Generator : SpinalHDL v1.10.1    git head : 2527c7c6b0fb0f95e5e1a5722a0be732b364ce43
// Component : Fp32Mul
// Git hash  : 628be231b362b875b9d15540aca1df8532a9b73d


module Spinal_HDL_FP32_Mul (
  input  wire          io_input_valid,
  input  wire [22:0]   io_input_payload_a_mant,
  input  wire [7:0]    io_input_payload_a_exp,
  input  wire          io_input_payload_a_sign,
  input  wire [22:0]   io_input_payload_b_mant,
  input  wire [7:0]    io_input_payload_b_exp,
  input  wire          io_input_payload_b_sign,
  output wire          io_result_valid,
  output reg  [22:0]   io_result_payload_mant,
  output reg  [7:0]    io_result_payload_exp,
  output wire          io_result_payload_sign
);

  wire       [23:0]   fixTo_1_dout;
  wire       [7:0]    _zz_n0_is_nan;
  wire       [22:0]   _zz_n0_is_nan_1;
  wire                _zz_n0_is_nan_2;
  wire       [9:0]    _zz_n1_exp_mul;
  wire       [8:0]    _zz_n1_exp_mul_1;
  wire       [48:0]   _zz_n2_mant_mul_adj;
  wire       [9:0]    _zz_n2_exp_mul_adj;
  wire       [9:0]    _zz_n2_exp_mul_adj_1;
  wire       [1:0]    _zz_n2_exp_mul_adj_2;
  wire       [9:0]    _zz_n2_exp_mul_adj_3;
  wire       [1:0]    _zz_n2_exp_mul_adj_4;
  wire       [9:0]    _zz_io_result_payload_exp;
  wire                n1_isValid;
  wire                n1_n0_sign_mul;
  wire                n1_n0_is_zero;
  wire                n1_n0_is_inf;
  wire                n1_n0_is_nan;
  wire                n0_isValid;
  wire                n2_valid;
  wire                n1_valid;
  wire                n2_n0_is_zero;
  wire                n2_n0_is_inf;
  wire                n2_n0_is_nan;
  wire                n2_n0_sign_mul;
  wire       [9:0]    n2_n1_exp_mul;
  wire       [47:0]   n2_n1_mant_mul;
  wire                n2_isValid;
  wire       [47:0]   n1_mant_mul;
  wire       [23:0]   n1_n0_mant_b;
  wire       [23:0]   n1_n0_mant_a;
  wire       [9:0]    n1_exp_mul;
  wire       [22:0]   n1_n0_b_mant;
  wire       [7:0]    n1_n0_b_exp;
  wire                n1_n0_b_sign;
  wire       [22:0]   n1_n0_a_mant;
  wire       [7:0]    n1_n0_a_exp;
  wire                n1_n0_a_sign;
  wire                n0_sign_mul;
  wire       [23:0]   n0_mant_b;
  wire       [23:0]   n0_mant_a;
  wire                n0_is_zero;
  wire                n0_b_is_zero;
  wire                n0_a_is_zero;
  wire                n0_is_inf;
  wire                n0_is_nan;
  wire       [22:0]   n0_b_mant;
  wire       [7:0]    n0_b_exp;
  wire                n0_b_sign;
  wire       [22:0]   n0_a_mant;
  wire       [7:0]    n0_a_exp;
  wire                n0_a_sign;
  wire                n0_valid;
  wire       [46:0]   n2_mant_mul_adj;
  wire       [9:0]    n2_exp_mul_adj;
  wire                when_FpxxMul_l73;
  wire                when_FpxxMul_l75;

  assign _zz_n1_exp_mul = {1'b0,_zz_n1_exp_mul_1};
  assign _zz_n1_exp_mul_1 = ({1'b0,n1_n0_a_exp} + {1'b0,n1_n0_b_exp});
  assign _zz_n2_mant_mul_adj = ({n2_n1_mant_mul,1'b0} >>> n2_n1_mant_mul[47]);
  assign _zz_n2_exp_mul_adj = ($signed(n2_n1_exp_mul) + $signed(_zz_n2_exp_mul_adj_1));
  assign _zz_n2_exp_mul_adj_2 = {1'b0,n2_n1_mant_mul[47]};
  assign _zz_n2_exp_mul_adj_1 = {{8{_zz_n2_exp_mul_adj_2[1]}}, _zz_n2_exp_mul_adj_2};
  assign _zz_n2_exp_mul_adj_4 = {1'b0,fixTo_1_dout[23]};
  assign _zz_n2_exp_mul_adj_3 = {{8{_zz_n2_exp_mul_adj_4[1]}}, _zz_n2_exp_mul_adj_4};
  assign _zz_io_result_payload_exp = n2_exp_mul_adj;
  assign _zz_n0_is_nan = 8'h00;
  assign _zz_n0_is_nan_1 = 23'h000000;
  assign _zz_n0_is_nan_2 = (|n0_b_mant);
  MulFixTo fixTo_1 (
    .din  (n2_mant_mul_adj[46:0]), //i
    .dout (fixTo_1_dout[23:0]   )  //o
  );
  assign n0_valid = io_input_valid;
  assign n0_a_mant = io_input_payload_a_mant;
  assign n0_a_exp = io_input_payload_a_exp;
  assign n0_a_sign = io_input_payload_a_sign;
  assign n0_b_mant = io_input_payload_b_mant;
  assign n0_b_exp = io_input_payload_b_exp;
  assign n0_b_sign = io_input_payload_b_sign;
  assign n0_is_nan = (((((&n0_a_exp) && (|n0_a_mant)) || ((&n0_b_exp) && (|n0_b_mant))) || (((n0_a_exp == _zz_n0_is_nan) && (n0_a_mant == _zz_n0_is_nan_1)) && ((&n0_b_exp) && (! _zz_n0_is_nan_2)))) || (((n0_b_exp == 8'h00) && (n0_b_mant == 23'h000000)) && ((&n0_a_exp) && (! (|n0_a_mant)))));
  assign n0_is_inf = (((&n0_a_exp) && (! (|n0_a_mant))) || ((&n0_b_exp) && (! (|n0_b_mant))));
  assign n0_a_is_zero = (((n0_a_exp == 8'h00) && (n0_a_mant == 23'h000000)) || (n0_a_exp == 8'h00));
  assign n0_b_is_zero = (((n0_b_exp == 8'h00) && (n0_b_mant == 23'h000000)) || (n0_b_exp == 8'h00));
  assign n0_is_zero = (n0_a_is_zero || n0_b_is_zero);
  assign n0_mant_a = {1'b1,n0_a_mant};
  assign n0_mant_b = {1'b1,n0_b_mant};
  assign n0_sign_mul = (n0_a_sign ^ n0_b_sign);
  assign n1_exp_mul = ($signed(_zz_n1_exp_mul) - $signed(10'h07f));
  assign n1_mant_mul = (n1_n0_mant_a * n1_n0_mant_b);
  assign io_result_valid = n2_isValid;
  assign n2_mant_mul_adj = _zz_n2_mant_mul_adj[46 : 0];
  assign n2_exp_mul_adj = ($signed(_zz_n2_exp_mul_adj) + $signed(_zz_n2_exp_mul_adj_3));
  assign io_result_payload_sign = n2_n0_sign_mul;
  always @(*) begin
    if(n2_n0_is_nan) begin
      io_result_payload_exp = 8'hff;
    end else begin
      if(n2_n0_is_inf) begin
        io_result_payload_exp = 8'hff;
      end else begin
        if(when_FpxxMul_l73) begin
          io_result_payload_exp = 8'h00;
        end else begin
          if(when_FpxxMul_l75) begin
            io_result_payload_exp = 8'hff;
          end else begin
            io_result_payload_exp = _zz_io_result_payload_exp[7:0];
          end
        end
      end
    end
  end

  always @(*) begin
    if(n2_n0_is_nan) begin
      io_result_payload_mant = 23'h7fffff;
    end else begin
      if(n2_n0_is_inf) begin
        io_result_payload_mant = 23'h000000;
      end else begin
        if(when_FpxxMul_l73) begin
          io_result_payload_mant = 23'h000000;
        end else begin
          if(when_FpxxMul_l75) begin
            io_result_payload_mant = 23'h000000;
          end else begin
            io_result_payload_mant = fixTo_1_dout[22:0];
          end
        end
      end
    end
  end

  assign when_FpxxMul_l73 = (n2_n0_is_zero || ($signed(n2_exp_mul_adj) <= $signed(10'h000)));
  assign when_FpxxMul_l75 = ($signed(10'h0ff) <= $signed(n2_exp_mul_adj));
  assign n1_valid = n0_isValid;
  assign n1_n0_a_mant = n0_a_mant;
  assign n1_n0_a_exp = n0_a_exp;
  assign n1_n0_a_sign = n0_a_sign;
  assign n1_n0_b_mant = n0_b_mant;
  assign n1_n0_b_exp = n0_b_exp;
  assign n1_n0_b_sign = n0_b_sign;
  assign n1_n0_is_nan = n0_is_nan;
  assign n1_n0_is_inf = n0_is_inf;
  assign n1_n0_is_zero = n0_is_zero;
  assign n1_n0_mant_a = n0_mant_a;
  assign n1_n0_mant_b = n0_mant_b;
  assign n1_n0_sign_mul = n0_sign_mul;
  assign n2_valid = n1_isValid;
  assign n2_n0_is_nan = n1_n0_is_nan;
  assign n2_n0_is_inf = n1_n0_is_inf;
  assign n2_n0_is_zero = n1_n0_is_zero;
  assign n2_n0_sign_mul = n1_n0_sign_mul;
  assign n2_n1_exp_mul = n1_exp_mul;
  assign n2_n1_mant_mul = n1_mant_mul;
  assign n0_isValid = n0_valid;
  assign n1_isValid = n1_valid;
  assign n2_isValid = n2_valid;

endmodule

module MulFixTo (
  input  wire [46:0]   din,
  output wire [23:0]   dout
);

  wire       [23:0]   _zz__zz_dout_1;
  wire       [1:0]    _zz__zz_dout_1_1;
  wire       [24:0]   _zz__zz_dout;
  reg        [23:0]   _zz_dout;
  wire                when_UInt_l238;
  reg        [23:0]   _zz_dout_1;
  wire                when_UInt_l219;

  assign _zz__zz_dout_1_1 = {1'b0,1'b1};
  assign _zz__zz_dout_1 = {22'd0, _zz__zz_dout_1_1};
  assign _zz__zz_dout = ({1'b0,din[46 : 23]} + {1'b0,{23'h000000,1'b1}});
  assign when_UInt_l238 = (! din[24]);
  assign when_UInt_l219 = (din[23] && (|din[22 : 0]));
  always @(*) begin
    if(when_UInt_l219) begin
      _zz_dout_1 = ({1'b0,din[46 : 24]} + _zz__zz_dout_1);
    end else begin
      _zz_dout_1 = {1'b0,din[46 : 24]};
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
