import math
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
import torch.nn
import numpy.typing as npt
import numpy as np
import copy

from networkx import is_isolate

from heepstorch import quantization
from heepstorch.code_generator import CodeGenerator
from heepstorch import im2row


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
    def num_input_features(self) -> int | None:
        """
        Returns the number of input features required by this module. Returns None if it does not require an specific
        number of input features.
        """
        pass

    @abstractmethod
    def num_output_features(self) -> int | None:
        """
        Returns the number of output features required by this module. Returns None if it does not require an specific
        number of output features.
        """
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

    def num_input_features(self) -> int | None:
        return self.DIM_IN

    def num_output_features(self) -> int | None:
        return self.DIM_OUT

    def get_name(self) -> str:
        return self.name

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

    def num_input_features(self) -> int | None:
        return None

    def num_output_features(self) -> int | None:
        return None

    def performs_inference_in_place(self) -> bool:
        return True

    def get_name(self) -> str:
        return self.name

    def forward_quantized(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.maximum(x, 0)

    def get_quantized_torch_module(self) -> torch.nn.Module:
        return self.quantized_torch_module

    def generate_model_parameters_c_code_constexpr_definitions(self) -> (str, str):
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

    def generate_model_parameters_c_code_constexpr_definitions(self) -> (str, str, str):
        raise NotImplementedError()

    def performs_inference_in_place(self) -> bool:
        raise NotImplementedError()

    def num_input_features(self) -> int | None:
        raise NotImplementedError()

    def num_output_features(self) -> int | None:
        raise NotImplementedError()

    def get_name(self) -> str:
        return self.name

    def generate_inference_c_code(self, input_buffer_name: str, output_buffer_name: str) -> str:
        raise NotImplementedError()


class Flatten(Module):
    """
    Flatten a 2d image with channels to a one-dimensional array to use with fully-connected layers
    """

    def __init__(self, torch_module: torch.nn.Module, name: str):
        self.name = name
        self.torch_module = torch_module
        self.quantized_torch_module = copy.deepcopy(torch_module)

    def forward_quantized(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # Flatten column by column into a 1-d array, and construct a row matrix from it (1 row, M columns).
        # The input is a matrix with one image channel in each column, and the output 1-d array has the flattened images
        #  of each channel concatenated side by side.
        return x.T.flatten().reshape([1, -1])

    def get_quantized_torch_module(self) -> torch.nn.Module:
        return self.quantized_torch_module

    def generate_model_parameters_c_code_constexpr_definitions(self) -> (str, str, str):
        raise NotImplementedError()

    def performs_inference_in_place(self) -> bool:
        raise NotImplementedError()

    def num_input_features(self) -> int | None:
        raise NotImplementedError()

    def num_output_features(self) -> int | None:
        raise NotImplementedError()

    def get_name(self) -> str:
        return self.name

    def generate_inference_c_code(self, input_buffer_name: str, output_buffer_name: str) -> str:
        raise NotImplementedError()


class SequentialNetwork:
    def __init__(self, modules: OrderedDict[str, Module]):
        self.modules = modules

    @staticmethod
    def from_torch_sequential(seq: torch.nn.Sequential) -> 'SequentialNetwork':
        def sanitize_name(name: str) -> str:
            # 0. If empty, return something
            if name == '':
                return '_empty'

            # 1. Replace spaces and special characters with underscores
            name = re.sub(r'[^a-zA-Z0-9_]', '_', name)

            # 2, Remove consecutive underscores
            name = re.sub(r'_+', '_', name)

            # 3. If starts with a number, add prefix underscore
            if name[0].isdigit():
                name = f"_{name}"

            return name

        # First check that sanitizing names keeps them unique
        sanitized_names = [sanitize_name(name) for name in seq._modules]
        assert len(sanitized_names) == len(
            set(sanitized_names)), f'After sanitizing, nn.Sequential are not unique! (After sanitization: {sanitized_names})'

        return SequentialNetwork(OrderedDict(
            [(sanitize_name(name), Module.from_torch_module(m, name)) for (name, m) in seq._modules.items()])
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
