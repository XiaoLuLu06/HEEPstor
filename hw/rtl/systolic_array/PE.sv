module PE(
    input logic resetn,
    input logic clk,

    input logic signed[7:0] in,
    input logic signed[7:0] acc, 
    input logic signed[7:0] w, 
    
    output logic signed[7:0] out
);

logic signed[15:0] mult;
logic signed[16:0] mult_acc;

assign mult = in * w;
assign mult_acc = mult + acc;
assign out = mult_acc[7:0];

endmodule 
