// Copyright 2022 EPFL
// Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

package heepstor_pkg;

  import addr_map_rule_pkg::*;
  import core_v_mini_mcu_pkg::*;

  localparam SYSTOLIC_ARRAY_SIZE = ${heepstor_pkg.sv.tpl};

  // TODO: Simplify this. We probably only need one EXT_XBAR_NSLAVE, no EXT_XBAR_NMASTER and no EXT_PERIPHERALS. 

  // For now, the systolic array has no master ports, it's fed the data by the CPU / DMA
  localparam EXT_XBAR_NMASTER = 0;

  // One slave port to the systolic array
  localparam EXT_XBAR_NSLAVE = 1;

  localparam int unsigned LOG_EXT_XBAR_NMASTER = EXT_XBAR_NMASTER > 1 ? $clog2(
      CGRA_XBAR_NMASTER
  ) : 32'd1;
  localparam int unsigned LOG_EXT_XBAR_NSLAVE = EXT_XBAR_NSLAVE > 1 ? $clog2(
      EXT_XBAR_NSLAVE
  ) : 32'd1;

  //slave mmap and idx
  localparam logic [31:0] SYSTOLIC_ARRAY_START_ADDRESS = core_v_mini_mcu_pkg::EXT_SLAVE_START_ADDRESS + 32'h000000;
  localparam logic [31:0] SYSTOLIC_ARRAY_MEM_SIZE = 32'h100000;
  localparam logic [31:0] SYSTOLIC_ARRAY_END_ADDRESS = SYSTOLIC_ARRAY_START_ADDRESS + SYSTOLIC_ARRAY_MEM_SIZE;
  localparam logic [31:0] SYSTOLIC_ARRAY_IDX = 32'd0;

  localparam addr_map_rule_t [EXT_XBAR_NSLAVE-1:0] EXT_XBAR_ADDR_RULES = '{
	'{idx: SYSTOLIC_ARRAY_IDX, start_addr: SYSTOLIC_ARRAY_START_ADDRESS, end_addr: SYSTOLIC_ARRAY_END_ADDRESS}
  };

  //slave encoder
  localparam EXT_SYSTEM_NPERIPHERALS = 1;

  localparam logic [31:0] SYSTOLIC_ARRAY_PERIPH_START_ADDRESS = core_v_mini_mcu_pkg::EXT_PERIPHERAL_START_ADDRESS + 32'h0000000;
  localparam logic [31:0] SYSTOLIC_ARRAY_PERIPH_MEM_SIZE = 32'h0001000;
  localparam logic [31:0] SYSTOLIC_ARRAY_PERIPH_END_ADDRESS = SYSTOLIC_ARRAY_PERIPH_START_ADDRESS + SYSTOLIC_ARRAY_PERIPH_MEM_SIZE;
  localparam logic [31:0] SYSTOLIC_ARRAY_IDX = 32'd0;

  localparam addr_map_rule_t [EXT_SYSTEM_NPERIPHERALS-1:0] EXT_PERIPHERALS_ADDR_RULES = '{
      '{
          idx: SYSTOLIC_ARRAY_IDX,
          start_addr: SYSTOLIC_ARRAY_PERIPH_START_ADDRESS,
          end_addr: SYSTOLIC_ARRAY_PERIPH_END_ADDRESS
      }
  };

  localparam int unsigned EXT_PERIPHERALS_PORT_SEL_WIDTH = EXT_SYSTEM_NPERIPHERALS > 1 ? $clog2(
      EXT_SYSTEM_NPERIPHERALS
  ) : 32'd1;

endpackage  // heepstor_pkg
