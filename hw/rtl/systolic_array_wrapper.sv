module systolic_array_wrapper
  import obi_pkg::*;
  import reg_pkg::*;
  import heepstor_pkg::*;
  import TicSAT_pkg::*;
(
    inout logic clk_i,
    inout logic rst_n,

    input obi_req_t obi_bus_req_i,
    output obi_resp_t obi_bus_resp_o
);

// ===========================================================================
//    Bus signal basic handling
// ===========================================================================

  // Register targeted by the request address. Only the 20 least significant
  //  bits are taken into account, because the systolic array memory mapped
  //  area is 32'h100000
  logic[19:0] req_addr;
  assign req_addr = obi_bus_req_i.addr[19:0];

  // For now, assume that we can always respond to any transaction in one cycle
  assign obi_bus_resp_o.gnt = obi_bus_req_i.req;

  // Delay the response by one cycle due to restrictions of the OBI protocol.
  logic response_valid_reg;
  logic [31:0] response_data_reg;

  assign obi_bus_resp_o.rvalid = response_valid_reg;
  assign obi_bus_resp_o.rdata = response_data_reg;

// ===========================================================================
//    Systolic array instantiation
// ===========================================================================

  // Inputs to the systolic array
  logic[31:0] input_value;
  logic[LOG_SYSTOLIC_ARRAY_SIZE-1:0] input_idx;
  TicSAT_pkg::command_t cmd;

  // Systolic array output
  logic[31:0] output_value;

  TicSAT_FP32_Int8_Pipelined #(
    .SA_SIZE(heepstor_pkg::SYSTOLIC_ARRAY_SIZE)
  ) ticsat_fp32_int8_pipelined_i (
    .clk(clk_i),
    .resetn(rst_n),

    .in_val(input_value),
    .in_idx(input_idx),
    .out(output_value),

    .cmd(cmd)
  );

// ###################################################
//    Storage of output value
// ###################################################

  logic[31:0] output_value_reg;

  // When performing a CMD_QUEUE or CMD_STREAM, store the output in output_value_reg, to be able to read it later.
  always_ff @(posedge clk_i) begin
      if (rst_n == 1'b0) begin
          output_value_reg <= 32'h0;
      end else if (cmd == CMD_QUEUE || cmd == CMD_STREAM) begin
          output_value_reg <= output_value;
      end
  end

// ###################################################
//    Command handling
// ###################################################

  // Bits 19 and 18 of the address control the TicSat command.
  // The index is set by the address from bits LOG_SYSTOLIC_ARRAY_SIZE+1 to bit 2.
  // The lowest 2 bits of the address will always be 0 for alignment purposes.
  // Therefore, the write offset of a systolic array command will be ((CMD << 18) | (IDX << 2))

  logic[31:0] weights_twos_complement;
  logic[31:0] weights_sign_magnitude;

  twos_complement_to_sign_mag twos_complement_to_sign_magnitude_i (
   .in(weights_twos_complement),
   .out(weights_sign_magnitude)
  );

  // The weights will be loaded in the bus
  assign weights_twos_complement = obi_bus_req_i.wdata;

  // If the instruction is weight-load, first convert the weights from two's complement to sign+magnitude.
  // The systolic array expects the weights to be in sign+magnitude format, but we'll expose to the software
  //  a two's complement interface, to avoid software overhead.
  assign input_value = (cmd == CMD_WRITE_WEIGHTS) ? weights_sign_magnitude : obi_bus_req_i.wdata;

  assign input_idx = req_addr[LOG_SYSTOLIC_ARRAY_SIZE+1:2];
  assign cmd = (obi_bus_req_i.req && obi_bus_req_i.we) ?
                      TicSAT_pkg::command_t'(req_addr[19:18]) : CMD_NONE;

// ###################################################
//    Response logic
// ###################################################


  // Includes correct handling of the valid signal (which is req with one cycle of delay)
  //  and correct handling of read return values.
  always_ff @(posedge clk_i) begin
    if (rst_n == 1'b0) begin
      response_valid_reg <= 1'b0;
      response_data_reg <= 32'h0;
    end else begin
      response_valid_reg <= obi_bus_req_i.req;

      // Note that we don't always have a read command, but the read command will always
      //  read the previous output_value_reg
      response_data_reg <= output_value_reg;
    end
  end

endmodule  // systolic_array_wrapper
