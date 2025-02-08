# Copyright EPFL contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0


# Makefile to generates heepsilon files and build the design with fusesoc

.PHONY: clean help

# Heepstor software parameters. Feel free to change them.
ENABLE_DEBUG_HEEPSTOR_ASSERTIONS ?= 1
USE_SOFTWARE_DNN_LAYER_OPERATORS ?= 0

# TARGET 		?= sim
FPGA_BOARD 	?= zcu104
PORT		?= /dev/ttyUSB2

CPU ?= cv32e40p

EXTERNAL_DOMAINS = 1
PROJECT ?= test_playground

MEMORY_BANKS ?= 16 # Multiple of 2
#MEMORY_BANKS_IL ?= 4 # Power of 2
  
export HEEP_DIR = hw/vendor/esl_epfl_x_heep/

# TODO: See if we can use .venv or have to use conda
# include $(HEEP_DIR)Makefile.venv

HEEPSTOR_CFG ?= heepstor_cfg.hjson

# TODO: Make a guide on how to get started in README.md

# TODO: Check automatically generated 
heepstor-gen:
	$(PYTHON) util/heepstor_gen.py --cfg $(HEEPSTOR_CFG) --outdir hw/rtl --pkg-sv hw/rtl/heepstor_pkg.sv.tpl
	$(PYTHON) util/heepstor_gen.py --cfg $(HEEPSTOR_CFG) --outdir sw/external/drivers/systolic_array --header-c sw/external/drivers/systolic_array/systolic_array_def.h.tpl

# TODO: See what EXTERNAL_DOMAINS is, and how to adapt it to our usecase.

# Generates mcu files. First the mcu-gen from X-HEEP is called.
# This is needed to be done after the X-HEEP mcu-gen because the test-bench to be used is the one from heepsilon, not the one from X-HEEP.
mcu-gen: heepstor-gen
	$(MAKE) -f $(XHEEP_MAKE) EXTERNAL_DOMAINS=${EXTERNAL_DOMAINS} CPU=${CPU} MEMORY_BANKS=${MEMORY_BANKS} $(MAKECMDGOALS)

## Builds (synthesis and implementation) the bitstream for the FPGA version using Vivado
## @param FPGA_BOARD=zcu104
## @param FUSESOC_FLAGS=--flag=<flagname>
# TODO: See if we can use .venv or have to use conda
# vivado-fpga: |venv
vivado-fpga:
	fusesoc --cores-root . run --no-export --target=$(FPGA_BOARD) $(FUSESOC_FLAGS) --setup --build eslepfl:systems:heepstor 2>&1 | tee buildvivado.log


# Runs verible formating
verible:
	util/format-verible;

# TODO: Enable simulation. Maybe copy set-up from HEEPsilon?
# Simulation
# verilator-sim:
# 	fusesoc --cores-root . run --no-export --target=sim --tool=verilator $(FUSESOC_FLAGS) --setup --build eslepfl:systems:heepsilon 2>&1 | tee buildsim.log

# questasim-sim:
# 	fusesoc --cores-root . run --no-export --target=sim --tool=modelsim $(FUSESOC_FLAGS) --setup --build eslepfl:systems:heepsilon 2>&1 | tee buildsim.log

# questasim-sim-opt: questasim-sim
# 	$(MAKE) -C build/eslepfl_systems_heepsilon_0/sim-modelsim opt

# vcs-sim:
# 	fusesoc --cores-root . run --no-export --target=sim --tool=vcs $(FUSESOC_FLAGS) --setup --build eslepfl:systems:heepsilon 2>&1 | tee buildsim.log


# TODO: Enable simulation. Maybe copy set-up from HEEPsilon?
## Generates the build output for a given application
## Uses verilator to simulate the HW model and run the FW
## UART Dumping in uart0.log to show recollected results
# run-verilator:
# 	$(MAKE) app PROJECT=$(PROJECT)
# 	cd ./build/eslepfl_systems_heepsilon_0/sim-verilator; \
# 	./Vtestharness +firmware=../../../sw/build/main.hex; \
# 	cat uart0.log; \
# 	cd ../../..;

# TODO: Enable simulation. Maybe copy set-up from HEEPsilon?
## Generates the build output for a given application
## Uses questasim to simulate the HW model and run the FW
## UART Dumping in uart0.log to show recollected results
# run-questasim:
# 	$(MAKE) app PROJECT=$(PROJECT)
# 	cd ./build/eslepfl_systems_heepsilon_0/sim-modelsim; \
# 	make run PLUSARGS="c firmware=../../../sw/build/main.hex"; \
# 	cat uart0.log; \
# 	cd ../../..;


# Builds the program and uses flash-load to run on the FPGA
run-fpga:
	$(MAKE) app PROJECT=$(PROJECT) LINKER=flash_load TARGET=$(FPGA_BOARD) ARCH=rv32imfc
	( cd hw/vendor/esl_epfl_x_heep/sw/vendor/yosyshq_icestorm/iceprog && make clean && make all ) ;\
	$(MAKE) flash-prog ;\

# Builds the program and uses flash-load to run on the FPGA.
# Additionally opens picocom (if available) to see the output.
run-fpga-com:
	$(MAKE) app PROJECT=$(PROJECT) LINKER=flash_load TARGET=$(FPGA_BOARD) ARCH=rv32imfc
	( cd hw/vendor/esl_epfl_x_heep/sw/vendor/yosyshq_icestorm/iceprog && make clean && make all ) ;\
	$(MAKE) flash-prog ;\
	picocom -b 9600 -r -l --imap lfcrlf /dev/serial/by-id/usb-FTDI_Quad_RS232-HS-if02-port0

.PHONY: app
# Add a dependency on the existing app target of XHEEP to create a link to the build folder
app: link_build
	$(PYTHON) util/heepstor_defs.py --enable-debug $(ENABLE_DEBUG_HEEPSTOR_ASSERTIONS) --use-software-dnn $(USE_SOFTWARE_DNN_LAYER_OPERATORS) sw/external/heepstor_defs.h.tpl
	$(MAKE) xheep_app PROJECT=$(PROJECT) LINKER=flash_load TARGET=$(FPGA_BOARD) ARCH=rv32imfc
picocom:
	picocom -b 9600 -r -l --imap lfcrlf /dev/serial/by-id/usb-FTDI_Quad_RS232-HS-if02-port0

openocd:
	openocd -f ./hw/vendor/esl_epfl_x_heep/tb/core-v-mini-mcu-pynq-z2-esl-programmer.cfg

# When running GDB, remember to first build the desired app
gdb: link_build
	$(PYTHON) util/heepstor_defs.py --enable-debug $(ENABLE_DEBUG_HEEPSTOR_ASSERTIONS) --use-software-dnn $(USE_SOFTWARE_DNN_LAYER_OPERATORS) sw/external/heepstor_defs.h.tpl
	$(MAKE) xheep_app PROJECT=$(PROJECT) LINKER=on_chip TARGET=$(FPGA_BOARD) ARCH=rv32imfc
	$(RISCV)/bin/riscv32-unknown-elf-gdb sw/build/main.elf -x util/gdb_openocd_commands

XHEEP_MAKE = $(HEEP_DIR)/external.mk

.PHONY: xheep_app
xheep_app:
	$(MAKE) -f $(XHEEP_MAKE) app

include $(XHEEP_MAKE)

app: link_build

clean-app: link_rm

link_build:
	ln -sf ../hw/vendor/esl_epfl_x_heep/sw/build sw/build

link_rm:
	rm sw/build

clean:
	rm -rf build buildsim.log
