module PE_FP32_Int8(
    input logic resetn,
    input logic clk,

    input logic signed[31:0] in,
    input logic signed[31:0] acc, 
    input logic signed[7:0] w, 
    
    output logic signed[31:0] out
);

logic[31:0] mult;

FP32_int8_mult u_FP32_int8_mult (
    .A      (in),
    .B      (w),
    .res    (mult)
);

Fp32_Adder u_Fp32_Adder (
    .A      (mult),
    .B      (acc),
    .res    (out)
);

endmodule 
