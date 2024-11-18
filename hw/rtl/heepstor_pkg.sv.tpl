// Copyright 2022 EPFL
// Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

package heepstor_pkg;

  import addr_map_rule_pkg::*;
  import core_v_mini_mcu_pkg::*;

  localparam SYSTOLIC_ARRAY_SIZE = ${systolic_array_size};

  // TODO: Simplify this. We probably only need one EXT_XBAR_NSLAVE, no EXT_XBAR_NMASTER and no EXT_PERIPHERALS. 

  // One slave port to the systolic array
  localparam EXT_XBAR_NSLAVE = 1;

  //slave mmap and idx
  localparam logic [31:0] SYSTOLIC_ARRAY_START_ADDRESS = core_v_mini_mcu_pkg::EXT_SLAVE_START_ADDRESS + 32'h000000;
  localparam logic [31:0] SYSTOLIC_ARRAY_MEM_SIZE = 32'h100000;
  localparam logic [31:0] SYSTOLIC_ARRAY_END_ADDRESS = SYSTOLIC_ARRAY_START_ADDRESS + SYSTOLIC_ARRAY_MEM_SIZE;
  localparam logic [31:0] SYSTOLIC_ARRAY_IDX = 32'd0;

  localparam addr_map_rule_t [EXT_XBAR_NSLAVE-1:0] EXT_XBAR_ADDR_RULES = '{
	'{idx: SYSTOLIC_ARRAY_IDX, start_addr: SYSTOLIC_ARRAY_START_ADDRESS, end_addr: SYSTOLIC_ARRAY_END_ADDRESS}
  };

endpackage  // heepstor_pkg
