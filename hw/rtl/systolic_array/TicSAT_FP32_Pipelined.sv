import TicSAT_pkg::*;

module TicSAT_FP32_Pipelined #(
    parameter SA_SIZE = -1,

    // This module uses FP32 weights and activations 
    // This parameters cannot be changed without recoding!!! They should be localparams,
    //  but DesignCompiler will complain. 
    parameter WEIGHT_SIZE = 32,
    parameter ACTIVATION_SIZE = 32
) (
    input logic resetn,
    input logic clk,

    // Used to load both weights and activations, depending on cmd
    input logic[ACTIVATION_SIZE-1:0] in_val,

    // Used to select in which FIFO position to write the input and read the output
    input logic[ $clog2(SA_SIZE)-1:0] in_idx,

    output logic[ACTIVATION_SIZE-1:0] out,

    input command_t cmd
);

logic[ACTIVATION_SIZE-1:0] systolic_array_inputs[SA_SIZE-1:0];
logic[ACTIVATION_SIZE-1:0] systolic_array_outputs[SA_SIZE-1:0];

logic outputs_valid;

SA_FP32_Pipelined #(
    .SA_SIZE            (SA_SIZE)
) u_SA_FP32 (
    .resetn             (resetn),
    .clk                (clk),
    .weight_input       (in_val),
    .inputs             (systolic_array_inputs),
    .outputs            (systolic_array_outputs),
    .cmd                (cmd),
    .outputs_valid      (outputs_valid)
);

FIFO_in #(
    .SA_SIZE            (SA_SIZE)
) u_FIFO_in (
    .resetn             (resetn),
    .clk                (clk),
    .in                 (in_val),
    .in_row_idx         (in_idx),
    .outputs            (systolic_array_inputs),
    .cmd                (cmd)
);

FIFO_out_Pipelined #(
    .SA_SIZE            (SA_SIZE)
) u_FIFO_out (
    .resetn             (resetn),
    .clk                (clk),
    .inputs             (systolic_array_outputs),
    .in_row_idx         (in_idx),
    .out                (out),
    .outputs_ready      (outputs_valid),
    .cmd                (cmd)
);

endmodule