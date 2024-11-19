// This module multiplies 32-bit floating-point number (A) and a *SIGN-MAGNITUDE* 8-bit integer (B).
// Note: The module expects B in *SIGN-MAGNITUDE*, instead of the more common two's complement for
//      efficiency and simplicity reasons.
// Note: This module doesn't handle NaNs, subnormals, wrapping or rounding properly. 
module FP32_int8_mult(
    // A is an IEEE-754 32-bit floating point number with the following format:
    //
    // A[31]:       Sign (1 bit). 0=Positive, 1=Negative
    // A[30:23]:    Exponent (8 bit). It is biased by 127: A value of 127 represents exponent 0.
    // A[22:0]:     Mantissa (23 bits). Unsigned number.
    //
    // a = (-1)**sign * 2**(exp - 127) * 1.mantissa
    //
    // 31 30       22                       0
    // +-+--------+-------------------------+
    // |S|  Exp   |        Mantissa         |
    // +-+--------+-------------------------+
    input logic[31:0] A,
    
    // B is a sign-magnitude 8-bit integer with the following format:
    // 
    // B[7]:      Sign (1 bit). 0=Positive, 1=Negative.
    // B[6:0]:    Magnitude (7 bit). Unsigned integer.
    //
    // b = (-1)**sign * magnitude 
    //
    // 7 6           0
    // +-+-----------+
    // |S| Magnitude |
    // +-+-----------+
    input logic[7:0] B,
    
    // A is an IEEE-754 32-bit floating point number with the same format as input A.
    output logic[31:0] res
);
    // --------------------------------
    // Decomposition of FP32-A
    // --------------------------------

    logic A_sign;
    logic[7:0] A_exp;
    logic[22:0] A_mantissa;

    assign A_sign = A[31];
    assign A_exp = A[30:23];
    assign A_mantissa = A[22:0];

    // ------------------------------------
    // Decomposition of sign-magnitude B
    // ------------------------------------

    logic B_sign;
    logic[6:0] B_mag;

    assign B_sign = B[7];
    assign B_mag = B[6:0];

    // ------------------------------------
    // Decomposition of sign-magnitude B
    // ------------------------------------

    logic res_sign;
    logic[7:0] res_exp;
    logic[22:0] res_mantissa;

    assign res[31] = res_sign;
    assign res[30:23] = res_exp;
    assign res[22:0] = res_mantissa;

    // ------------------------------------
    //  Zero-input detection
    // ------------------------------------

    logic is_A_zero;
    assign is_A_zero = (A_exp == 0) && (A_mantissa == 0);

    logic is_B_zero;
    assign is_B_zero = B_mag == 0;

    logic is_any_input_zero;
    assign is_any_input_zero = is_A_zero || is_B_zero;

    // ------------------------------------
    //  Non-zero res decomposition
    // ------------------------------------

    logic nz_res_sign;
    logic[7:0] nz_res_exp;
    logic[22:0] nz_res_mantissa;

    // ------------------------------------
    //  Non-zero res computation
    // ------------------------------------

    // Sign: A_sign XOR B_sign
    assign nz_res_sign = A_sign ^ B_sign;

    // Multiplicationn of A's mantissa (with the implicit one concatenated) and B magnitude
    logic[31:0] mult;
    assign mult = {1'b1, A_mantissa} * B_mag;

    logic[3:0] fsb_mult_idx;

    Find_First_Set_Bit #(
        .WIDTH    (9)
    ) u_fsb (
        .in       (mult[31 -: 9]),
        .out      (fsb_mult_idx),
        .valid () // Floating output
    );

    assign nz_res_exp = A_exp + fsb_mult_idx;
    assign nz_res_mantissa = mult >> fsb_mult_idx;

    // --------------------------------------------------------
    //  Actual res selection based on zero or non-zero inputs
    // --------------------------------------------------------

    always_comb begin
        if (is_any_input_zero) begin
            res_sign = 0;
            res_exp = '0;
            res_mantissa = '0;
        end else begin
            res_sign = nz_res_sign;
            res_exp = nz_res_exp;
            res_mantissa = nz_res_mantissa;
        end
    end

endmodule