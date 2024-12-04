<p align="left"><img src="docs/HEEPstor_logo.png" width="500"></p>

HEEPstor is an acceleration platform targeting ultra-low-power tensor computation built over the [X-HEEP](https://github.com/esl-epfl/x-heep) platform, extending it with a hybridly-quantized systolic array to optimize matrix-matrix multiply (GEMM) operations with FP32 activations and INT8 quantized-weights for neural networks. 

HEEPstor is integrated with the PyTorch Python Machine Learning to streamline the model to hardware-accelerated edge-AI workflow. HEEPstor can automatically convert a PyTorch module into a hardware-accelerated C X-Heep application using our hybridly-quantized systolic-array, automatically handling weight quantization, DNN layer operator execution and buffer management. HEEPstor also generates a quantized PyTorch model after weight-quantization, to easily evaluate the X-Heep model accuracy without needing to run the whole test on the embedded device.

# Capabilities

HEEPstor can currently:

- Automatically generate X-Heep C++ applications implementing a given PyTorch model (see below section for supported layer modules)
- Quantize the weights of a PyTorch model to INT8, and re-scale the results seamlessly to achieve both the accuracy of FP32 activations and the area, energy and memory savings of INT8 weights
- Generate an INT8 weight-quantized PyTorch model from a given PyTorch model to easily evaluate the accuracy of the embedded quantized model on a desktop Python C++  
- Generate an example `main.cpp` file performing an inference with given example data, and compare the output
against the PyTorch model golden output  
- Generate detailed per-layer inference performance reports on the embedded device
- Target either an accelerated hybridly-quantized systolic array or software
- Seamlessly support GPU-trained models

## Currently supported PyTorch layer modules

Currently, the following PyTorch layer modules are supported. The main module must be a `torch.nn.Sequential`, which contains one of the following layers:

- `nn.Linear`: with weights (which are quantized) and bias
- `nn.ReLU`

Additionally, an optional `Softmax` is supported at the end of the model to generate a probability distribution.

In the future, our plan is to add support for more layers, such as `nn.Conv2d`.

# Getting started

Due to its modular design, HEEPstor respects the X-HEEP workflow. As such, you can follow [X-HEEP's getting started](https://x-heep.readthedocs.io/en/latest/How_to/GettingStarted.html) to set up the environment. 

In this section, we will briefly go over the basic set-up. All of the make commands must be run inside the `X-Heep` conda environment. After installing the environment, you can activate it using `conda activate core-v-mini-mcu`.

Right now, the only supported FPGA is the Zynq UltraScale+ MPSoC ZCU104 Evaluation Kit (`zcu104`).

In order to build the HW and SW C++ applications, do:

1. Set the desired systolic array size in the config file `heepstor_cfg.hjson`
2. Run `make heepstor-gen` to regenerate the files which depend on `heepstor_cfg.hjson`
3. Run `make mcu-gen` to generate the MCU files, including the vendorized X-Heep. 
4. Run `make vivado-fpga` to perform synthesis and implementation to generate the bitstream for the FPGA.
5. Load the bitstream into the FPGA using Vivado Hardware Manager.
6. Run `make app PROJECT=your_project_name` to build the application in folder `sw/applications/your_project_name`.
7. Run `make run-fpga-com PROJECT=your_project_name` to load the application into the Flash.

In order to generate a C++ application from a PyTorch model, do:

1. Write your Python application in `python-dnn/apps`. Take a look at some examples such as `mnist-single_layer` or `mnist-multi_layer`. The HEEPstor implementation with PyTorch is in the `heepstorch` package, stored in `python-dnn/heepstorch`.
2. Install the prerrequisites into your Python installation. The prerrequisites can be found in `python-dnn/requirements.txt`.
3. Run the Python application by adding `PYTHONPATH=/your/absolute/path/to/python-dnn/`: `PYTHONPATH=/your/absolute/path/to/python-dnn/ python3 python-dnn/apps/your-app/main.py`. Alternatively, you can open the folder `python-dnn` in an IDE such as PyCharm, which will then handle PYTHONPATH. Run each Python app inside their respective folders, as most of them will download some datasets (such as MNIST). We recommend using PyCharm, which automatically takes care of this.

## Options

There are several options that can be tweaked if needed:

- Memory size: You can change the number of X-Heep HW memory banks by tweaking `MEMORY_BANKS` in the `Makefile`. If you run out of space for intermediate buffers or the input / output matrices, which are stored in the `StaticArenaAllocator`, you can increase it's size by changing `StaticArenaAllocator::ARENA_SIZE` in `sw/external/memory/static_arena_allocator.h`.
- Disable debug assertions: To speed-up operation, you can disable HEEPstor (which disables every `HEEPSTOR_ASSERT`) assertions by using `ENABLE_DEBUG_HEEPSTOR_ASSERTIONS=0`: `make run-fpga-com PROJECT=your_project_name ENABLE_DEBUG_HEEPSTOR_ASSERTIONS=0`.
- Use software DNN layer operators instead of the systolic array: `make run-fpga-com PROJECT=your_project_name USE_SOFTWARE_DNN_LAYER_OPERATORS=1`.

# Reference

You can learn more about the HEEPstor systolic array platform in: [Systolic Arrays and Structured Pruning Co-design for Efficient Transformers in Edge Systems](https://arxiv.org/abs/2411.10285).

```
@misc{palacios2024systolicarraysstructuredpruning,
      title={Systolic Arrays and Structured Pruning Co-design for Efficient Transformers in Edge Systems}, 
      author={Pedro Palacios and Rafael Medina and Jean-Luc Rouas and Giovanni Ansaloni and David Atienza},
      year={2024},
      eprint={2411.10285},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      url={https://arxiv.org/abs/2411.10285}, 
}
```