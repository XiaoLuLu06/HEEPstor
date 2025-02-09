<p align="left"><img src="docs/HEEPstor_logo.png" width="500"></p>

HEEPstor is an open-hardware co-design framework for Post-Training Quantized Machine Learning at the edge, built on top of the X-HEEP platform.

HEEPstor enables seamless deployment of unmodified PyTorch models on X-HEEP-based RISC-V heterogeneous SoCs with custom ML accelerators, allowing for rapid design space exploration, optimization and evaluation of novel ML hardware accelerators with real machine learning workloads defined in PyTorch.

This repository also contains a hybridly-quantized systolic array accelerator that serves as a hardware accelerator back-end for HEEPstor, providing end-to-end deployment from PyTorch models to a fully functional X-HEEP RISC-V SoC targeting FPGAs.

# Key features


- **Automated Model Deployment:** Convert PyTorch models directly into hardware-accelerated C++ X-HEEP applications.
- **Hybrid Quantization:** Quantize the weights of any PyTorch model to INT8, and re-scale the results seamlessly to achieve both the accuracy of FP32 activations and part of the area, energy and memory savings of INT8 weights.
- **Quantized Accuracy Evaluation**: Generate a fakely-quantized PyTorch model to efficiently evaluate post-quantization accuracy on the whole test dataset using GPU acceleration.
- **Hardware Flexibility**: Clean hardware abstraction layers and modular design that enable optimization and exploration of different GEMM accelerator architectures while maintaining the same software interface.
- **Performance Analysis:** Create memory usage and detailed per-layer inference performance reports.

## Supported PyTorch layer modules

The framework requires the main PyTorch module to be a `nn.Sequential` containing any of these layers:
- `nn.Linear`
- `nn.ReLU`
- `nn.Conv2d`
- `nn.Flatten`
- `nn.BatchNorm2d`
- `nn.MaxPool2d`
- `nn.Dropout`

Additionally, an optional `Softmax` is supported at the end of the model to generate a probability distribution. This is achieved by passing an additional `append_final_softmax=True` argument to `heepstorch.code_generator.CodeGenerator.generate_code`. This way, the original model is not modified and criterions such as `torch.nn.CrossEntropyLoss` can still be used, without having to add a useless softmax layer to model training.

# Getting started

Due to its modular design, HEEPstor respects the X-HEEP workflow. As such, you can follow [X-HEEP's getting started](https://x-heep.readthedocs.io/en/latest/GettingStarted/index.html) to set up the environment. 

In the rest of this section, we will go over the basic set-up and how to build and run HEEPstor applications, assuming you have at least set up X-HEEP's `apt` packages, Conda environment, RISC-V compiler, Verilator and Verible. 

> [!WARNING]  
> All the `make` commands must be run inside X-HEEP's Conda environment. After installing the environment, you can activate it using `conda activate core-v-mini-mcu`.

> [!NOTE]
> Right now, the only supported FPGA is the Zynq UltraScale+ MPSoC ZCU104 Evaluation Kit (`zcu104`).

There are two steps needed to deploy a PyTorch model to X-HEEP:

1. Generate a C++ X-HEEP inference application from a PyTorch model  
2. Synthesize X-HEEP hardware, build and run the generated C++ inference application

We will cover first the second step, and then the first step (which can be skipped if you have already generated your target C++ inference code, or if you wish to use one of the example pre-generated ones).

### Synthesizing X-HEEP hardware, building and running C++ applications

> [!IMPORTANT]  
> Ensure that the systolic array size defined in `heepstor_cfg.hjson` matches between the loaded FPGA bitstream and the built software. Mismatches may cause unexpected behavior

In order to build the HW and SW C++ applications, you must:

1. Set the desired systolic array size in the config file `heepstor_cfg.hjson`
2. Run `make heepstor-gen` to regenerate the files which depend on `heepstor_cfg.hjson`
3. Run `make mcu-gen` to generate the MCU files, including the vendorized X-HEEP. 
4. Run `make vivado-fpga` to perform synthesis and implementation to generate the bitstream for the FPGA.
5. Load the bitstream into the FPGA using Vivado Hardware Manager.
6. Run `make app PROJECT=your_project_name` to build the C++ application stored in the folder `sw/applications/your_project_name`. You can build a pre-existing C++ application or follow the instructions in the next paragraph to automatically generate C++ inference code from a PyTorch model.
7. Run `make run-fpga-com PROJECT=your_project_name` to load the application into the Flash if you have an ESL-EPFL programmer for X-Heep attached. Alternatively, if you want to load using OpenOCD, see the corresponding section below.

### Automatically generating C++ inference code from a PyTorch model

In order to automatically generate a C++ application from a PyTorch model, you must:

1. Write your Python application in `python-dnn/apps`. Take a look at some examples such as `mnist-single_layer`, `mnist-multi_layer`, `fmnist-conv2d` or `cifar10-conv2d`. The HEEPstor integration with PyTorch is available in the `heepstorch` package, stored in `python-dnn/heepstorch`.
2. Install the prerequisites into your Python installation. The prerequisites can be found in `python-dnn/requirements.txt`.
3. Run the Python application by adding `PYTHONPATH=/your/absolute/path/to/python-dnn/`: `PYTHONPATH=/your/absolute/path/to/python-dnn/ python3 python-dnn/apps/your-app/main.py`. Alternatively, you can open the folder `python-dnn` in an IDE such as PyCharm, which will then handle PYTHONPATH. Run each Python app inside their respective folders, as most of them will download some datasets (such as MNIST). We recommend using PyCharm, which automatically takes care of this.

## Running and debugging using OpenOCD

You will need to open 3 terminal windows, one with an UART screen, another for OpenOCD and a third one for GDB. Follow the official X-Heep instructions for installing OpenOCD (https://x-heep.readthedocs.io/en/latest/How_to/Debug.html).

To open those windows, run the following commands in order:
1. `make picocom`
2. `make openocd`
3. `make gdb PROJECT=PROJECT_NAME`. This application calls `make app` before running GDB, with the provided arguments (such as `PROJECT` or other options). 

In the GDB window, you can use the following commands:
- `load` to load the executable into memory. By default, `make gdb` runs load after connecting to the OpenOCD GDB server.
- `continue` to execute the loaded executable.
- `monitor reset halt` to reset all non-debug modules (including the CPU).

## Options

There are several options that can be tweaked if needed:

- **Memory size**: You can change the number of X-Heep HW memory banks by tweaking `MEMORY_BANKS` in the `Makefile`. If you run out of space for intermediate buffers or the input / output matrices, which are stored in the `StaticArenaAllocator`, you can increase its size by changing `StaticArenaAllocator::ARENA_SIZE` in `sw/external/memory/static_arena_allocator.h`.
- **Disable debug assertions**: To speed up operation, you can disable HEEPstor assertions (which disables every `HEEPSTOR_ASSERT`) by using `ENABLE_DEBUG_HEEPSTOR_ASSERTIONS=0`: `make run-fpga-com PROJECT=your_project_name ENABLE_DEBUG_HEEPSTOR_ASSERTIONS=0`.
- **Use software DNN layer operators instead of the systolic array**: `make run-fpga-com PROJECT=your_project_name USE_SOFTWARE_DNN_LAYER_OPERATORS=1`.
