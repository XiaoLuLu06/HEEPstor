module twos_complement_to_sign_mag (
    // Input: 4 packed int8_t in two's complement
    input  logic [31:0] in,

    // Output: 4 packed int8_t in sign-magnitude.
    //  -128 in two's complement saturates to -127 in sign-magnitude.
    output logic [31:0] out
);

    // Helper function to convert 8-bit two's complement to sign-magnitude
    // with saturation for -128 case.
    function automatic logic [7:0] convert_byte(logic [7:0] twos_comp);
        logic [7:0] magnitude;
        logic sign;

        sign = twos_comp[7];  // Extract sign bit

        // Check for -128 case (10000000)
        if (twos_comp == 8'h80) begin
            // Saturate to largest negative value in sign-magnitude (-127)
            return 8'b11111111;
        end else begin
            // If negative, take absolute value by inverting and adding 1
            magnitude = sign ? (~twos_comp + 8'd1) : twos_comp;

            // Combine sign and magnitude
            // For sign-magnitude: MSB is sign, rest is absolute value
            return {sign, magnitude[6:0]};
        end
    endfunction

    // Convert each byte independently
    always_comb begin
        for (int i = 0; i < 4; i++) begin
            out[i*8 +: 8] = convert_byte(in[i*8 +: 8]);
        end
    end

endmodule