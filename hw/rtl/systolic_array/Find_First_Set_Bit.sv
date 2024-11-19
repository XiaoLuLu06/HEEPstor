// Find_First_Set takes as an input an array of WIDTH bits.
// If none of them is set, then valid=false.
// If any of them is set, then valid=true and out=index of the most-significant bit set to 1.
module Find_First_Set_Bit #(
    parameter WIDTH = 8
) (
    input logic[WIDTH-1:0] in,
    output logic[$clog2(WIDTH)-1:0] out,
    output logic valid
);
    integer i;

    always_comb begin
        valid = 0;
        out = '0;
        for (i = WIDTH-1; i >= 0; i = i - 1) begin
            if (in[i] && !valid) begin
                valid = 1;
                out = i;
                // Could use break instead of '&& !valid', but Icarus Verilog 11.0 doesn't support breaks
                // break;
            end
        end
    end
    
endmodule