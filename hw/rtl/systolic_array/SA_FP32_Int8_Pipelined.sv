import TicSAT_pkg::*;

module SA_FP32_Int8_Pipelined #(
    parameter int SA_SIZE = 8,

    // This module uses int8_t (sign + magnitude) weights and FP32 activations.
    // This parameters cannot be changed without recoding!!! They should be localparams,
    //  but DesignCompiler will complain. 
    parameter WEIGHT_SIZE = 8,
    parameter ACTIVATION_SIZE = 32
) (
    input logic resetn,
    input logic clk,

    // Weight loading is done in a delay-line fashion.
    // Weights are input in the systolic array in reverse order. 
    // On cmd==CMD_WRITE_WEIGHTS, the weights are shifted to the right (and to the-bottom if it's the last row), and input_weight is loaded into the first weight. 
    // In order to load all the weights, load the last weight, then the second-to-last, and so on.
    // As weights are 8-bit, but weight_input is 32-bit, we load 4 weights at a time.
    // The assignment is the following:
    //      weights_reg[0][0] <= weight_input[31:24];
    //      weights_reg[0][1] <= weight_input[23:16];
    //      weights_reg[0][2] <= weight_input[15:8];
    //      weights_reg[0][3] <= weight_input[7:0];
    input logic[ACTIVATION_SIZE-1:0] weight_input,
    
    input logic[ACTIVATION_SIZE-1:0] inputs[SA_SIZE-1:0],
    output logic[ACTIVATION_SIZE-1:0] outputs[SA_SIZE-1:0],

    output logic outputs_valid,

    input command_t cmd
);
    if (SA_SIZE < 4) begin
//        $error("SA_SIZE must be at least 4 due to how weights are loaded");
    end
    
    if (SA_SIZE % 4 != 0) begin
//        $error("SA_SIZE must be a multiple of 4 due to how weights are loaded");
    end
    
    // Weights for each processing element. 
    //    Access as weights_reg[r][c]
    logic[WEIGHT_SIZE-1:0] weights_reg[SA_SIZE-1:0][SA_SIZE-1:0];

    // Accumulator registers storing the output of each PE.
    // Note that there are only SA_SIZE-1 rows of accumulators, as the first accumulator (input to first PE) is always 0.
    //    Access as accs_reg[r][c]
    logic[ACTIVATION_SIZE-1:0] accs_reg[SA_SIZE-2:0][SA_SIZE-1:0];

    // Registers that hold the input of each PE (which is passed to the right).
    // Note that there are only SA_SIZE-1 rows of input registers, as the input to the first PE comes from the SA module input inputs.
    //    Access as pe_inputs_reg[r][c]
    logic[ACTIVATION_SIZE-1:0] pe_inputs_reg[SA_SIZE-1:0][SA_SIZE-2:0];

    // The computation should only be advanced if the command is CMD_STREAM
    logic should_advance_computation;
    assign should_advance_computation = (cmd == CMD_STREAM);

    //////////////////////////////////////////////////////////////////////////
    //                       PROCESSING ELEMENTS
    //////////////////////////////////////////////////////////////////////////

    // Instantiate each processing element
    genvar r, c;
    generate
        for (r = 0; r < SA_SIZE; r = r + 1) begin: R_GEN
            for (c = 0; c < SA_SIZE; c = c + 1) begin: C_GEN                
                
                // PE INPUT
                logic[ACTIVATION_SIZE-1:0] pe_in;

                // If this is the first column, the input to the PE is the input to the SA module
                // Otherwise, the input to the PE is the output of the PE to the left
                assign pe_in = (c == 0) ? inputs[r] : pe_inputs_reg[r][c-1];

                // If should_advance_computation, advance the input to the right (if there's a next PE to the right)
                if (c < SA_SIZE-1) begin
                    always_ff @(posedge clk) begin
                        if (~resetn) begin
                            // TODO: Do we really need to reset the input registers?
                            pe_inputs_reg[r][c] <= '0;
                        end else begin
                            if (should_advance_computation) begin
                                pe_inputs_reg[r][c] <= pe_in;
                            end else begin
                                pe_inputs_reg[r][c] <= pe_inputs_reg[r][c];
                            end
                        end
                    end
                end

                // PE ACCUMULATOR
                logic[ACTIVATION_SIZE-1:0] pe_acc;
                
                // If this is the first row, the accumulator is 0
                // Otherwise, the accumulator is the output of the PE above
                assign pe_acc = (r == 0) ? '0 : accs_reg[r-1][c];
                
                // PE OUTPUT
                logic[ACTIVATION_SIZE-1:0] pe_out;
                
                logic PE_output_valid;

                // If this is the last row, then the output of the accumulator is the output of the SA module
                if (r == SA_SIZE-1) begin
                    assign outputs[c] = pe_out;
                
                    // Choose the first PE of the last row as an indicator of whether the outputs are valid.
                    // Note that all PEs will finish at the same time.
                    if (c == 0) begin
                        assign outputs_valid = PE_output_valid;
                    end
                
                end else begin
                    // Otherwise, the output of the PE is stored in accs_reg for the next row to use
                    always_ff @(posedge clk) begin
                        if (~resetn) begin
                            // TODO: Do we really need to reset the accumulator registers?
                            accs_reg[r][c] <= '0;
                        end else begin
                            if (PE_output_valid) begin
                                accs_reg[r][c] <= pe_out;
                            end else begin
                                accs_reg[r][c] <= accs_reg[r][c];
                            end
                        end
                    end
                end

                PE_FP32_Int8_Pipelined u_PE_FP32 (
                    .resetn(resetn),
                    .clk(clk),
                    .in(pe_in),

                    .acc(pe_acc),
                    .w(weights_reg[r][c]),
                    .out(pe_out),

                    .input_valid(should_advance_computation),
                    .output_valid(PE_output_valid)
                );

                //////////////////////////////////////////////////////////////////////////
                //                     WEIGHTS REGISTERS
                //////////////////////////////////////////////////////////////////////////

                always_ff @(posedge clk) begin
                    if (~resetn) begin
                        // TODO: Do we really need to reset the accumulator registers?
                        weights_reg[r][c] <= '0;
                    end else begin
                        weights_reg[r][c] <= weights_reg[r][c];

                        // If cmd is CMD_WRITE_WEIGHTS, then shift the weights to the right (and possible to the bottom),
                        //  and load weight_input into the first weight register. Note that weights are 8-bit and we load
                        //  32-bits at a time, so we have to load and shift 32/8 = 4 weights each time.
                        if (cmd == CMD_WRITE_WEIGHTS) begin
                            if (r == 0 && c == 0) begin
                                weights_reg[r][c] <= weight_input[31:24];
                            end else if (r == 0 && c == 1) begin
                                weights_reg[r][c] <= weight_input[23:16];
                            end else if (r == 0 && c == 2) begin 
                                weights_reg[r][c] <= weight_input[15:8];
                            end else if (r == 0 && c == 3) begin
                                weights_reg[r][c] <= weight_input[7:0];
                            end else begin
                                if (c == 0) begin
                                    weights_reg[r][c] <= weights_reg[r-1][SA_SIZE - 4];
                                end else if (c == 1) begin
                                    weights_reg[r][c] <= weights_reg[r-1][SA_SIZE - 3];
                                end else if (c == 2) begin
                                    weights_reg[r][c] <= weights_reg[r-1][SA_SIZE - 2];
                                end else if (c == 3) begin
                                    weights_reg[r][c] <= weights_reg[r-1][SA_SIZE - 1];
                                end else begin
                                    weights_reg[r][c] <= weights_reg[r][c-4];
                                end
                            end
                        end
                    end
                end
            end
        end
    endgenerate

endmodule