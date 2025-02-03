import math
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional

import torch.nn
import numpy.typing as npt
import numpy as np
import copy
from enum import Enum
from dataclasses import dataclass

from sympy.codegen.cnodes import static

from heepstorch import quantization
from heepstorch.code_generator import CodeGenerator
from heepstorch import im2row


class NetworkMode(Enum):
    CONV = "conv"  # Single image, multiple channels
    LINEAR = "linear"  # Multiple batches possible


@dataclass
class ImageDimensions:
    height: int
    width: int

    def num_pixels(self) -> int:
        return self.height * self.width

    def validate(self):
        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"Invalid image dimensions: {self.height}x{self.width}")


class Module(ABC):
    @abstractmethod
    def forward_quantized(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Returns the result of running the forward pass of the equivalent torch module
         using the quantized weights
        """
        pass

    @abstractmethod
    def get_quantized_torch_module(self) -> torch.nn.Module:
        """
        Returns a torch module with the quantized parameters that performs the same
         behavior as the original module, except it operates with the quantized weights.
        """
        pass

    @abstractmethod
    def generate_model_parameters_c_code_constexpr_definitions(self) -> (str, str, str):
        """
        Returns three strings representing all necessary C code constexpr model parameters definitions (such as weights,
        quantization scaling, bias) for Heepstor execution.
        The first string of the tuple are the constexpr array declarations to be stored in the ModelParameters class,
        containing the actual value of the weights. The second string of the tuple are the wrappers of the arrays into
        Heepstor Matrix classes. The third string are the constexpr array definitions, without the declaration.
        """
        pass

    @abstractmethod
    def performs_inference_in_place(self) -> bool:
        """
        Returns True whenever the module can perform inference in place (that is, the input buffer and the output buffer
        can be the same). Returns False otherwise.
        """
        pass

    @abstractmethod
    def num_input_channels(self) -> int | None:
        """
        Returns the number of input features / channels required by this module. Returns None if it does not require an specific
        number of input features.
        """
        pass

    @abstractmethod
    def num_output_channels(self) -> int | None:
        """
        Returns the number of output features / channels required by this module. Returns None if it does not require an specific
        number of output features.
        """
        pass

    @abstractmethod
    def output_image_dimensions(self, input_dims: Optional[ImageDimensions], input_channels: int) -> Optional[
        ImageDimensions]:
        """Returns output image dimensions given input dimensions

        Args:
            input_channels:
        """
        pass

    @abstractmethod
    def network_mode(self) -> NetworkMode:
        """Returns whether this module operates in conv or linear mode"""
        pass

    @staticmethod
    def from_torch_module(m: torch.nn.Module, name: str) -> 'Module':
        """Convert a PyTorch module to a Module.

        Args:
            m: PyTorch module to convert
            name: String representing this module name.

        Returns:
            Module: Converted module

        Raises:
            ValueError: If the module type is not supported
        """

        # TODO: Implement Linear, ReLu, Softmax, Dropout, Sequential?
        # TODO: Then, implement Conv2d, BatchNorm2d, MaxPool2d.

        # For linear-only, let the matrix have [batch_size x num_features], processing multiple.
        # For conv, [num_input_channels x image_size], processing a single image at a time.
        if isinstance(m, torch.nn.Linear):
            return Linear(m, name)
        elif isinstance(m, torch.nn.ReLU):
            return ReLU(m, name)
        elif isinstance(m, torch.nn.Conv2d):
            return Conv2d(m, name)
        elif isinstance(m, torch.nn.Flatten):
            return Flatten(m, name)
        else:
            raise ValueError(f"Unsupported module type in Heepstorch: {type(m)}")

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of the module.
        """
        pass

    @abstractmethod
    def generate_inference_c_code(self, input_buffer_name: str, output_buffer_name: str) -> str:
        """
        Returns the C-code as a string to perform inference using the given input and output buffer names.
        """
        pass


class Linear(Module):
    """
    Linear layer of dimensions [DIM_IN, DIM_OUT]. DIM_IN is the dimensionality of the input and DIM_OUT
    is the dimensionality of the output.
    """

    def __init__(self, torch_module: torch.nn.Module, name: str):
        self.name = name

        self.DIM_IN = torch_module.in_features
        self.DIM_OUT = torch_module.out_features

        # Numpy array of shape [DIM_IN, DIM_OUT] with the weights
        self.weights_non_quantized = torch_module.weight.data.cpu().detach().numpy().transpose()
        self.weights_quantized, self.weight_scale = quantization.quantize(self.weights_non_quantized)

        # Numpy array containing the bias
        self.bias = torch_module.bias.cpu().detach().numpy()

        self.non_quantized_torch_module = torch_module
        self.quantized_torch_module = self.quantize_torch_module(torch_module)

    def performs_inference_in_place(self) -> bool:
        return False

    def num_input_channels(self) -> int | None:
        return self.DIM_IN

    def num_output_channels(self) -> int | None:
        return self.DIM_OUT

    def get_name(self) -> str:
        return self.name

    def output_image_dimensions(self, input_dims: Optional[ImageDimensions],
                                input_channels) -> Optional[ImageDimensions]:
        if input_dims is not None:
            raise ValueError("Linear layer expects no image dimensions")

        return None

    def network_mode(self) -> NetworkMode:
        return NetworkMode.LINEAR

    def quantize_torch_module(self, non_quantized_torch_module: torch.nn.Module) -> torch.nn.Module:
        quantized_module = copy.deepcopy(non_quantized_torch_module)

        fake_quantized_weights = quantization.fake_quantize(self.weights_non_quantized)

        quantized_module.weight.data = torch.from_numpy(fake_quantized_weights.transpose()).to(
            non_quantized_torch_module.weight.get_device())
        return quantized_module

    def forward_quantized(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return quantization.dequantize_matmult_result(x @ self.weights_quantized, self.weight_scale) + self.bias

    def get_quantized_torch_module(self) -> torch.nn.Module:
        return self.quantized_torch_module

    def weight_matrix_varname(self) -> str:
        return f'{self.name}_weights'

    def bias_matrix_varname(self) -> str:
        return f'{self.name}_bias'

    def weight_data_varname(self) -> str:
        return f'{self.name}_weight_data'

    def weight_scale_varname(self) -> str:
        return f'{self.name}_weight_scale'

    def bias_data_varname(self) -> str:
        return f'{self.name}_bias_data'

    def generate_model_parameters_c_code_constexpr_definitions(self) -> (str, str, str):
        assert self.weights_quantized.shape == (self.DIM_IN, self.DIM_OUT)

        c_code_weight_data, weight_arr_size = CodeGenerator.quantized_weights_to_packed_c_array(self.weights_quantized,
                                                                                                self.weight_data_varname())

        c_code_weight_scale = f'static constexpr float {self.weight_scale_varname()} = {self.weight_scale:.9g};'
        c_code_bias_data, bias_arr_size = CodeGenerator.bias_to_c_array(self.bias, self.bias_data_varname())

        model_parameter_definitions = '\n\n'.join([c_code_weight_scale, c_code_weight_data, c_code_bias_data])

        c_code_wrapper_weights = f'const PackedInt8Matrix {self.weight_matrix_varname()} = PackedInt8Matrix::from_const_pointer(ModelParameters::{self.weight_data_varname()}, {self.DIM_IN}, {self.DIM_OUT});'
        c_code_wrapper_bias = f'const Matrix<float> {self.bias_matrix_varname()} = Matrix<float>::from_const_pointer(ModelParameters::{self.bias_data_varname()}, 1, {bias_arr_size});'

        wrapper = '\n'.join([c_code_wrapper_weights, c_code_wrapper_bias])

        weight_data_declaration = f'constexpr uint32_t ModelParameters::{self.weight_data_varname()}[{weight_arr_size}];'
        bias_data_declaration = f'constexpr float ModelParameters::{self.bias_data_varname()}[{bias_arr_size}];'

        declarations = '\n'.join([weight_data_declaration, bias_data_declaration])

        return model_parameter_definitions, wrapper, declarations

    def generate_inference_c_code(self, input_buffer_name: str, output_buffer_name: str) -> str:
        return f'Linear::forward(systolic_array, {input_buffer_name}, {self.weight_matrix_varname()}, ModelParameters::{self.weight_scale_varname()}, {self.bias_matrix_varname()}, {output_buffer_name});'


class ReLU(Module):
    """
    Linear layer of dimensions [DIM_IN, DIM_OUT]. DIM_IN is the dimensionality of the input and DIM_OUT
    is the dimensionality of the output.
    """

    def __init__(self, torch_module: torch.nn.Module, name: str):
        self.name = name
        self.torch_module = torch_module
        self.quantized_torch_module = copy.deepcopy(torch_module)
        self.mode = None  # Will be set based on input

    def num_input_channels(self) -> int | None:
        return None

    def num_output_channels(self) -> int | None:
        return None

    def network_mode(self) -> NetworkMode:
        if self.mode is None:
            raise ValueError("Mode not yet determined: must process input first")
        return self.mode

    def output_image_dimensions(self, input_dims: Optional[ImageDimensions],
                                input_channels) -> Optional[ImageDimensions]:
        return input_dims  # Preserves dimensions

    def performs_inference_in_place(self) -> bool:
        return True

    def get_name(self) -> str:
        return self.name

    def forward_quantized(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.maximum(x, 0)

    def get_quantized_torch_module(self) -> torch.nn.Module:
        return self.quantized_torch_module

    def generate_model_parameters_c_code_constexpr_definitions(self) -> (str, str, str):
        return None, None, None

    def generate_inference_c_code(self, input_buffer_name: str, output_buffer_name: str) -> str:
        assert input_buffer_name == output_buffer_name
        return f'ReLU::forward({input_buffer_name});'


class Conv2d(Module):
    """
    2d convolutional layer.
    """

    def __init__(self, torch_module: torch.nn.Module, name: str):
        self.name = name
        self.torch_module = torch_module

        # For now, we only support padding=0, stride=1, dilation=1 and groups=1.
        assert torch_module.padding == (0, 0)
        assert torch_module.groups == 1
        assert torch_module.stride == (1, 1)
        assert torch_module.dilation == (1, 1)

        w = torch_module.weight.data.cpu().detach().numpy()

        # Number of output channels, input channels, kernel size
        self.C_OUT, self.C_IN, self.N, other_kernel_size = w.shape
        assert self.N == other_kernel_size

        # Kernel weight matrix of shape (C_IN * N * N, C_OUT), prepared for multiplication with im2row
        self.kernel_matrix_non_quantized = w.reshape(self.C_OUT, -1).T
        assert self.kernel_matrix_non_quantized.shape == (self.C_IN * self.N * self.N, self.C_OUT)

        self.kernel_matrix_quantized, self.kernel_scale = quantization.quantize(self.kernel_matrix_non_quantized)

        # Bias array of shape [C_OUT]
        self.bias = torch_module.bias.cpu().detach().numpy()

        self.quantized_torch_module = self.quantize_torch_module(torch_module)

        # This will be filled out in a later pass
        self.input_height = None
        self.input_width = None
        self.out_dims = None

    def quantize_torch_module(self, non_quantized_torch_module: torch.nn.Module) -> torch.nn.Module:
        quantized_module = copy.deepcopy(non_quantized_torch_module)

        fake_quantized_weights = quantization.fake_quantize(self.kernel_matrix_non_quantized)

        quantized_module.weight.data = torch.from_numpy(fake_quantized_weights.transpose()).reshape([self.C_OUT,
                                                                                                     self.C_IN, self.N,
                                                                                                     self.N]).to(
            non_quantized_torch_module.weight.get_device())
        return quantized_module

    def forward_quantized(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        H_W, NUM_CHANNELS = x.shape

        assert NUM_CHANNELS == self.C_IN

        # FIXME: For now, we assume in forward quantize that the images are square. To change this,
        #   more information about the height / width needs to be passed at this stage
        H = W = math.isqrt(H_W)
        assert H_W == H * W

        im2row_res = im2row.im2row(x, self.N, H, W)  # Shape: (H_out * W_out, C_in * N * N)

        matmul_res = quantization.dequantize_matmult_result(im2row_res @ self.kernel_matrix_quantized,
                                                            self.kernel_scale)
        res = matmul_res + self.bias
        return res

    def get_quantized_torch_module(self) -> torch.nn.Module:
        return self.quantized_torch_module

    def weight_matrix_varname(self) -> str:
        return f'{self.name}_kernel_weights'

    def bias_matrix_varname(self) -> str:
        return f'{self.name}_bias'

    def weight_data_varname(self) -> str:
        return f'{self.name}_kernel_weight_data'

    def weight_scale_varname(self) -> str:
        return f'{self.name}_kernel_weight_scale'

    def bias_data_varname(self) -> str:
        return f'{self.name}_bias_data'

    def generate_model_parameters_c_code_constexpr_definitions(self) -> (str, str, str):
        c_code_weight_data, weight_arr_size = CodeGenerator.quantized_weights_to_packed_c_array(
            self.kernel_matrix_quantized,
            self.weight_data_varname())

        c_code_weight_scale = f'static constexpr float {self.weight_scale_varname()} = {self.kernel_scale:.9g};'

        assert len(self.bias.shape) == 1
        c_code_bias_data, bias_arr_size = CodeGenerator.bias_to_c_array(self.bias, self.bias_data_varname())

        model_parameter_definitions = '\n\n'.join([c_code_weight_scale, c_code_weight_data, c_code_bias_data])

        weight_shape = self.kernel_matrix_quantized.shape
        assert len(weight_shape) == 2

        c_code_wrapper_weights = f'const PackedInt8Matrix {self.weight_matrix_varname()} = PackedInt8Matrix::from_const_pointer(ModelParameters::{self.weight_data_varname()}, {weight_shape[0]}, {weight_shape[1]});'
        c_code_wrapper_bias = f'const Matrix<float> {self.bias_matrix_varname()} = Matrix<float>::from_const_pointer(ModelParameters::{self.bias_data_varname()}, 1, {bias_arr_size});'

        wrapper = '\n'.join([c_code_wrapper_weights, c_code_wrapper_bias])

        weight_data_declaration = f'constexpr uint32_t ModelParameters::{self.weight_data_varname()}[{weight_arr_size}];'
        bias_data_declaration = f'constexpr float ModelParameters::{self.bias_data_varname()}[{bias_arr_size}];'

        declarations = '\n'.join([weight_data_declaration, bias_data_declaration])

        return model_parameter_definitions, wrapper, declarations

    def get_im2row_buffer_size(self) -> int:
        """
        Returns the number of floating point elements needed to hold the im2row matrix
        """
        return self.out_dims.num_pixels() * self.C_IN * self.N * self.N

    def output_image_dimensions(self, input_dims: ImageDimensions, input_channels: Optional[int]) -> ImageDimensions:
        if input_dims is None:
            raise ValueError("Conv2d requires input dimensions")

        if input_channels is not None:
            assert self.num_input_channels() == input_channels

        self.input_height = input_dims.height
        self.input_width = input_dims.width

        out_height = input_dims.height - self.N + 1
        out_width = input_dims.width - self.N + 1

        out_dims = ImageDimensions(out_height, out_width)

        self.out_dims = out_dims
        return out_dims

    def network_mode(self) -> NetworkMode:
        return NetworkMode.CONV

    def performs_inference_in_place(self) -> bool:
        return False

    def num_input_channels(self) -> int | None:
        return self.C_IN

    def num_output_channels(self) -> int | None:
        return self.C_OUT

    def get_name(self) -> str:
        return self.name

    @staticmethod
    def get_im2row_buffer_name() -> str:
        return 'im2row_buffer'

    def generate_inference_c_code(self, input_buffer_name: str, output_buffer_name: str) -> str:
        assert self.input_width is not None
        assert self.input_height is not None

        return f'Conv2d::forward(systolic_array, {input_buffer_name}, {self.weight_matrix_varname()}, ModelParameters::{self.weight_scale_varname()}, {self.bias_matrix_varname()}, {output_buffer_name}, {self.N}, {self.num_input_channels()}, {self.input_height}, {self.input_width}, {self.get_im2row_buffer_name()});'


class Flatten(Module):
    """
    Flatten a 2d image with channels to a one-dimensional array to use with fully-connected layers
    """

    def __init__(self, torch_module: torch.nn.Module, name: str):
        self.name = name
        self.torch_module = torch_module
        self.quantized_torch_module = copy.deepcopy(torch_module)

        # Will be set in a later pass
        self.input_dims = None
        self.input_channels = None
        self.output_features = None

    def forward_quantized(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # Flatten column by column into a 1-d array, and construct a row matrix from it (1 row, M columns).
        # The input is a matrix with one image channel in each column, and the output 1-d array has the flattened images
        #  of each channel concatenated side by side.
        return x.T.flatten().reshape([1, -1])

    def get_quantized_torch_module(self) -> torch.nn.Module:
        return self.quantized_torch_module

    def generate_model_parameters_c_code_constexpr_definitions(self) -> (str, str, str):
        return None, None, None

    def output_image_dimensions(self, input_dims: Optional[ImageDimensions], input_channels: int) -> None:
        """Computes output size and stores input configuration

        Args:
            input_channels:
        """
        if input_dims is None:
            raise ValueError(f"Flatten layer {self.name} requires input dimensions")

        # Store input configuration for dimension checking
        self.input_dims = input_dims

        if input_channels is not None:
            assert self.input_channels is None or self.input_channels == input_channels
            self.input_channels = input_channels

        if self.input_channels is None:
            raise ValueError(f"Flatten layer {self.name} requires input channels to be set first")

        # Output features is total number of elements: H * W * C
        self.output_features = input_dims.height * input_dims.width * self.input_channels

        # Flatten has no output image dimensions (LINEAR mode)
        return None

    def network_mode(self) -> NetworkMode:
        return NetworkMode.LINEAR

    def performs_inference_in_place(self) -> bool:
        return False

    def num_input_channels(self) -> int | None:
        return None  # Accepts any number of channels

    def num_output_channels(self) -> int | None:
        if self.output_features is None:
            raise ValueError(
                f"Flatten layer {self.name} output features not yet computed - call output_image_dimensions first")

        return self.output_features

    def get_name(self) -> str:
        return self.name

    def generate_inference_c_code(self, input_buffer_name: str, output_buffer_name: str) -> str:
        return f'Flatten::forward({input_buffer_name}, {output_buffer_name});'


class SequentialNetwork:
    def __init__(self, modules: OrderedDict[str, Module],
                 input_dimensions: Optional[ImageDimensions] = None,
                 input_channels: Optional[int] = None):
        self.modules = modules
        self.input_dimensions = input_dimensions
        self.input_channels = input_channels
        self.validate_network()

    @staticmethod
    def from_torch_sequential(
            seq: torch.nn.Sequential,
            input_height: Optional[int] = None,
            input_width: Optional[int] = None,
            input_channels: Optional[int] = None
    ) -> 'SequentialNetwork':
        """Create SequentialNetwork from PyTorch Sequential module

        Args:
            seq: PyTorch Sequential module
            input_height: Height of input images (required for conv networks)
            input_width: Width of input images (required for conv networks)
            input_channels: Number of input channels (required for conv networks)
        """
        # Determine if network requires image dimensions
        needs_dims = any(isinstance(m, (torch.nn.Conv2d, torch.nn.Flatten))
                         for m in seq.modules())

        if needs_dims:
            if any(x is None for x in (input_height, input_width, input_channels)):
                raise ValueError("Networks with Conv2d or Flatten layers require input dimensions")

            input_dimensions = ImageDimensions(
                height=input_height,
                width=input_width
            )
        else:
            input_dimensions = None

        # Create modules dictionary
        def sanitize_name(name: str) -> str:
            if name == '':
                return '_empty'
            name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
            name = re.sub(r'_+', '_', name)
            if name[0].isdigit():
                name = f"_{name}"
            return name

        sanitized_names = [sanitize_name(name) for name in seq._modules]
        assert len(sanitized_names) == len(set(sanitized_names)), \
            f'Duplicate names after sanitization: {sanitized_names}'

        return SequentialNetwork(
            OrderedDict([
                (sanitize_name(name), Module.from_torch_module(m, name))
                for (name, m) in seq._modules.items()
            ]),
            input_dimensions=input_dimensions,
            input_channels=input_channels
        )

    def get_quantized_torch_module(self):
        return torch.nn.Sequential(OrderedDict([
            (name, m.get_quantized_torch_module()) for name, m in self.modules.items()
        ]))

    def forward(self, x: npt.NDArray[np.float32]):
        for module in self.modules.values():
            x = module.forward_quantized(x)
        return x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_module_by_name(self, name: str):
        return self.modules[name]

    def supports_batching(self) -> bool:
        """Returns true when this network supports batch_sizes>1 for its inputs (i.e., when there are no CONV mode layers)
        """
        return all(
            m.network_mode() == NetworkMode.LINEAR
            for m in self.modules.values()
        )

    def validate_network(self):
        """Validates network architecture for compatibility constraints:
        1. If network has Conv2d/Flatten, must have input dimensions and channels
        2. No Conv2d layers after Flatten
        3. Channel counts match between consecutive layers
        4. Image dimensions are valid throughout conv layers
        5. Flatten gets proper channel count for output features
        """
        # First check if we need CONV mode
        has_conv_or_flatten = any(isinstance(m, (Conv2d, Flatten)) for m in self.modules.values())

        # Initialize network state
        if has_conv_or_flatten:
            if self.input_dimensions is None or self.input_channels is None:
                raise ValueError("Network contains Conv2d/Flatten but missing input dimensions or channels")
            current_mode = NetworkMode.CONV
            current_channels = self.input_channels
            current_dims = self.input_dimensions
        else:
            if self.input_dimensions is not None or self.input_channels is not None:
                raise ValueError("Network has no Conv2d/Flatten but input dimensions/channels were provided")
            current_mode = NetworkMode.LINEAR
            current_channels = None  # Will be set by first Linear layer
            current_dims = None

        for name, module in self.modules.items():
            # 1. Input channel validation
            if module.num_input_channels() is not None:
                if current_channels is None:
                    current_channels = module.num_input_channels()
                elif current_channels != module.num_input_channels():
                    raise ValueError(
                        f"Module {name} expects {module.num_input_channels()} "
                        f"channels/features but receives {current_channels}"
                    )
            # 2. Mode-specific validation and updates
            if isinstance(module, ReLU):
                module.mode = current_mode

            if current_mode == NetworkMode.CONV:
                if isinstance(module, Conv2d):
                    if current_dims is None:
                        raise ValueError(f"Conv2d layer {name} has no input dimensions")
                    current_dims = module.output_image_dimensions(current_dims, None)
                    if current_dims.height <= 0 or current_dims.width <= 0:
                        raise ValueError(
                            f"Invalid output dimensions {current_dims.height}x{current_dims.width} "
                            f"after Conv2d layer {name}"
                        )
                    current_channels = module.num_output_channels()
                elif isinstance(module, Flatten):
                    if current_dims is None or current_channels is None:
                        raise ValueError(f"Flatten layer {name} missing input dimensions or channels")
                    # Pass both dimensions and channels to compute output features
                    module.output_image_dimensions(current_dims, current_channels)
                    current_mode = NetworkMode.LINEAR
                    current_dims = None  # No more spatial dimensions in LINEAR mode
                    current_channels = module.num_output_channels()  # Get flattened feature count
                elif isinstance(module, Linear):
                    raise ValueError(f"Linear layer {name} cannot appear in CONV mode before Flatten")
            else:  # LINEAR mode
                if isinstance(module, Conv2d):
                    raise ValueError(f"Conv2d layer {name} cannot appear after Flatten")
                elif isinstance(module, Linear):
                    if current_channels is not None and module.num_input_channels() != current_channels:
                        raise ValueError(
                            f"Linear layer {name} expects {module.num_input_channels()} features "
                            f"but receives {current_channels}"
                        )
                    current_channels = module.num_output_channels()
                elif isinstance(module, Flatten):
                    raise ValueError(f"Flatten layer {name} cannot appear in LINEAR mode")

            # Update channels if module specifies output channels
            if module.num_output_channels() is not None:
                current_channels = module.num_output_channels()
