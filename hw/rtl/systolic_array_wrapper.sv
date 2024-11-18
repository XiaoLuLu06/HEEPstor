module systolic_array_wrapper 
  import obi_pkg::*;
  import reg_pkg::*;
  import heepstor_pkg::*;
(
    inout logic clk_i,
    inout logic rst_n,

    input obi_req_t obi_bus_req_i,
    output obi_resp_t obi_bus_resp_o
);
  // Register targeted by the request address. Only the 20 least significant
  //  bits are taken into account, because the systolic array memory mapped
  //  area is 32'h100000
  logic[19:0] req_addr_reg;
  assign req_addr_reg = obi_bus_req_i.addr[19:0];

  // For now, assume that we can always respond to any transaction in one cycle
  // TODO: Maybe instead of 1'b1 we should connect this straight to req? That was Simone's suggestion.
  assign obi_bus_resp_o.gnt = obi_bus_req_i.req;

  // Delay the response by one cycle due to restrictions of the OBI protocol.
  logic response_valid_reg;
  logic [31:0] response_data_reg;

  assign obi_bus_resp_o.rvalid = response_valid_reg;
  assign obi_bus_resp_o.rdata = response_data_reg;

  // ===========================================================================
  // TODO: Remove and change with the actual systolic array registers.

  //  For now, it's a single example register.
  logic[31:0] test_reg;

  // When we write, set the test_reg to (req_addr_reg << 12) | (wdata + 3).
  // On read, return the test_reg.

  // Response logic. Includes correct handling of the valid signal (which is req with one cycle of delay)
  //  and correct handling of read return values.
  always_ff @(posedge clk_i) begin
    if (rst_n == 1'b0) begin
      response_valid_reg <= 1'b0;
      response_data_reg <= 32'h0;
    end else begin
      response_valid_reg <= obi_bus_req_i.req;

      // TODO: We don't always have to return a valid read. In the future we can
      //  separate this and only read when necessary and whatever is necessary.
      response_data_reg <= test_reg;
    end
  end

  // Write logic
  always_ff @(posedge clk_i) begin
      if (rst_n == 1'b0) begin
          test_reg <= 32'h0;
      end else if (obi_bus_req_i.req && obi_bus_req_i.we) begin
          test_reg <= (req_addr_reg << 12) | (obi_bus_req_i.wdata + 32'd3);
      end
  end

// ===========================================================================


endmodule  // systolic_array_wrapper
