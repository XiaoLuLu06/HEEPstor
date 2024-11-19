import TicSAT_pkg::*;

module FIFO_out_Pipelined #(
    parameter SA_SIZE = 8,

    parameter ACTIVATION_SIZE = 32
) (
    input logic resetn,
    input logic clk,

    input logic[ACTIVATION_SIZE-1:0] inputs[SA_SIZE-1:0],

    input logic[$clog2(SA_SIZE)-1:0] in_row_idx,
    output logic[ACTIVATION_SIZE-1:0] out,
    
    // True when CMD_STREAM has been issued and the last-row pipelined PE has finished computing: 
    //   the SA output is ready to be read.
    input logic outputs_ready,

    input command_t cmd
);

// Need to cache the inputs to avoid the pipelined PEs from modifying the inputs while we are reading them.
// The cached_inputs will be set whenever outputs_ready is high, but it will only be copied into col_shift_reg
//  when CMD_STREAM is issued.
logic[ACTIVATION_SIZE-1:0] cached_inputs[SA_SIZE-1:0];

genvar i;
generate
    for (i = 0; i < SA_SIZE; i = i + 1) begin: C_GEN
        always_ff @(posedge clk) begin
            if (~resetn) begin
                // TODO: Do we really need to reset the cached inputs?
                cached_inputs[i] <= '0;
            end else begin
                if (outputs_ready) begin
                    cached_inputs[i] <= inputs[i];
                end else begin
                    cached_inputs[i] <= cached_inputs[i];
                end
            end
        end
    end
endgenerate;


// For each column, last_shift_reg contains the last value of the shift register
logic[ACTIVATION_SIZE-1:0] last_shift_reg[SA_SIZE-1:0];

assign out = last_shift_reg[in_row_idx];

// For each column, we generate a shift register of size SA_SIZE-c, that is only shifted on CMD_STREAM,
//   with the first register input taken from inputs. 
genvar r, c;
generate
    for (c = 0; c < SA_SIZE; c = c + 1) begin: R_GEN
        logic[ACTIVATION_SIZE-1:0] col_shift_reg[SA_SIZE-c-1:0];

        always_ff @(posedge clk) begin
            if (~resetn) begin
                col_shift_reg[0] <= '0;
            end else begin
                if (cmd == CMD_STREAM) begin
                    col_shift_reg[0] <= cached_inputs[c];
                end else begin
                    col_shift_reg[0] <= col_shift_reg[0];
                end
            end
        end

        // Shift-registers
        for (r = 1; r < SA_SIZE - c; r = r + 1) begin: C_GEN
            // Shift on CMD_STREAM to the right
            always_ff @(posedge clk) begin
                if (~resetn) begin
                    col_shift_reg[r] <= '0;
                end else begin
                    if (cmd == CMD_STREAM) begin
                        col_shift_reg[r] <= col_shift_reg[r-1];
                    end else begin
                        col_shift_reg[r] <= col_shift_reg[r];
                    end
                end
            end
        end

        assign last_shift_reg[c] = col_shift_reg[SA_SIZE-c-1];
    end
endgenerate;

endmodule