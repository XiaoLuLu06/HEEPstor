// From: https://github.com/tomverbeure/math
// Doesn't handle denormals, proper rounding, etc.

module Fp32_Multiplier_Latency2(
    input logic clk,
    input logic resetn,

    // Inputs and outputs are expected to be IEEE-754 32-bit floating-point numbers
    input logic[31:0] A,
    input logic[31:0] B,

    output logic[31:0] res,

    input logic input_valid,
    output logic output_valid
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
logic[7:0] B_exp;
logic[22:0] B_mantissa;

assign B_sign = B[31];
assign B_exp = B[30:23];
assign B_mantissa = B[22:0];

// ------------------------------------
// Decomposition of sign-magnitude B
// ------------------------------------

logic res_sign;
logic[7:0] res_exp;
logic[22:0] res_mantissa;

assign res[31] = res_sign;
assign res[30:23] = res_exp;
assign res[22:0] = res_mantissa;


Spinal_HDL_FP32_Mul_Latency2 u_Spinal_HDL_FP32_Mul_Latency2 (
    .clk(clk),
    .reset(~resetn),

    .io_input_valid               (input_valid),

    .io_input_payload_a_mant      (A_mantissa),
    .io_input_payload_a_exp       (A_exp),
    .io_input_payload_a_sign      (A_sign),

    .io_input_payload_b_mant      (B_mantissa),
    .io_input_payload_b_exp       (B_exp),
    .io_input_payload_b_sign      (B_sign),

    .io_result_valid           (output_valid),
    .io_result_payload_mant    (res_mantissa),
    .io_result_payload_exp     (res_exp),
    .io_result_payload_sign    (res_sign)
);

endmodule