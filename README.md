<p align="left"><img src="docs/HEEPstor_logo.png" width="500"></p>

HEEPstor is an acceleration platform targeting ultra-low-power tensor computation built over the [X-HEEP](https://github.com/esl-epfl/x-heep) platform, extending it with a hybridly-quantized systolic array to optimize matrix-matrix multiply (GEMM) operations with FP32 activations and INT8 quantized-weights for neural networks. 

In the future, our goal is to integrate HEEPstor with a Machine Learning Framework such as PyTorch to streamline the model to hardware-accelerated edge-AI workflow.

# Getting started

Due to its modular design, HEEPstor respects the X-HEEP workflow. As such, you can follow [X-HEEP's getting started](https://x-heep.readthedocs.io/en/latest/How_to/GettingStarted.html) to set up the environment.

### TODO: Write basic set-up instructions
- Mention conda
- Mention supported FPGAs
- Set systolic array size in `heepstor.core`
- Application development
- `heepstor-gen`
- `mcu-gen`
- `vivado-fpga`
- `run_fpga`