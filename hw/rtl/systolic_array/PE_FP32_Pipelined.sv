module PE_FP32_Pipelined(
    input logic resetn,
    input logic clk,

    input logic signed[31:0] in,
    input logic signed[31:0] acc, 
    input logic signed[31:0] w, 
    
    output logic signed[31:0] out,

    input logic input_valid,
    output logic output_valid
);

logic[31:0] mult;
logic mult_valid;

Fp32_Multiplier_Latency2 u_FP32_mult (
    .clk(clk),
    .resetn(resetn),

    .A      (in),
    .B      (w),
    .res    (mult),

    .input_valid(input_valid),
    .output_valid(mult_valid)
);

logic[31:0] mult_reg;
logic mult_valid_reg;

always_ff @(posedge clk) begin
    if (~resetn) begin
        mult_reg <= '0;
        mult_valid_reg <= 1'b0;
    end else begin
        mult_reg <= mult;
        mult_valid_reg <= mult_valid;
    end
end

Fp32_Adder_Latency3 u_Fp32_Adder (
    .clk(clk),
    .resetn(resetn),

    .A      (mult_reg),
    .B      (acc),
    .res    (out),

    .input_valid(mult_valid_reg),
    .output_valid(output_valid)
);

endmodule 
