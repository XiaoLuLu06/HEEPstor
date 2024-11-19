module PE_FP32_Int8_Pipelined(
    input logic resetn,
    input logic clk,

    input logic signed[31:0] in,
    input logic signed[31:0] acc, 
    input logic signed[7:0] w, 
    
    output logic signed[31:0] out,

    input logic input_valid,
    output logic output_valid
);

// Latency of inputs to FP32_int8_mult, that will then be synthesized using 
//  retiming (to avoid manually pipelining FP32_int8_mult). 
localparam DELAY_LINE_LATENCY = 2;

// Create a delay line of DELAY_LINE_LATENCY stages for A, B and input_valid
logic[31:0] in_delay_line[DELAY_LINE_LATENCY-1:0];
logic[7:0] w_delay_line[DELAY_LINE_LATENCY-1:0];
logic input_valid_delay_line[DELAY_LINE_LATENCY-1:0];

// Connect the first stage of the delay line to the inputs
assign in_delay_line[0] = in;
assign w_delay_line[0] = w;
assign input_valid_delay_line[0] = input_valid;

// Connect the rest of the stages of the delay line
genvar i;
generate
    for (i = 1; i < DELAY_LINE_LATENCY; i = i + 1) begin: DELAY_LINE_GEN
        always_ff @(posedge clk) begin
            if (~resetn) begin
                in_delay_line[i] <= '0;
                w_delay_line[i] <= '0;
                input_valid_delay_line[i] <= 1'b0;
            end else begin
                in_delay_line[i] <= in_delay_line[i-1];
                w_delay_line[i] <= w_delay_line[i-1];
                input_valid_delay_line[i] <= input_valid_delay_line[i-1];
            end
        end
    end
endgenerate;

logic[31:0] mult;

FP32_int8_mult u_FP32_int8_mult (
    .A      (in_delay_line[DELAY_LINE_LATENCY-1]),
    .B      (w_delay_line[DELAY_LINE_LATENCY-1]),
    .res    (mult)
);

logic[31:0] mult_reg;
logic mult_valid_reg;

always_ff @(posedge clk) begin
    if (~resetn) begin
        mult_reg <= '0;
        mult_valid_reg <= 1'b0;
    end else begin
        mult_reg <= mult;
        mult_valid_reg <= input_valid_delay_line[DELAY_LINE_LATENCY-1];
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
