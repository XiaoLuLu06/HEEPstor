import re
from abc import ABC, abstractmethod
from collections import OrderedDict
import torch.nn
import numpy.typing as npt
import numpy as np
import copy

from heepstorch import quantization


class HeepstorchModule(ABC):
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

    @staticmethod
    def from_torch_module(m: torch.nn.Module) -> 'HeepstorchModule':
        """Convert a PyTorch module to a HeepstorchModule.

        Args:
            m: PyTorch module to convert

        Returns:
            HeepstorchModule: Converted module

        Raises:
            ValueError: If the module type is not supported
        """

        # TODO: Implement Linear, ReLu, Softmax, Dropout, Sequential?
        # TODO: Then, implement Conv2d, BatchNorm2d, MaxPool2d.

        # For linear-only, let the matrix have [batch_size x num_features], processing multiple.
        # For conv, [num_input_channels x image_size], processing a single image at a time.
        if isinstance(m, torch.nn.Linear):
            return Linear(m)
        else:
            raise ValueError(f"Unsupported module type in Heepstorch: {type(m)}")


class Linear(HeepstorchModule):
    """
    Linear layer of dimensions [DIM_IN, DIM_OUT]. DIM_IN is the dimensionality of the input and DIM_OUT
    is the dimensionality of the output.
    """

    def __init__(self, torch_module: torch.nn.Module):
        self.DIM_IN = torch_module.in_features
        self.DIM_OUT = torch_module.out_features

        # Numpy array of shape [DIM_IN, DIM_OUT] with the weights
        self.weights_non_quantized = torch_module.weight.data.cpu().detach().numpy().transpose()
        self.weights_quantized, self.weight_scale = quantization.quantize(self.weights_non_quantized)

        # Numpy array containing the bias
        self.bias = torch_module.bias.cpu().detach().numpy()

        self.non_quantized_torch_module = torch_module
        self.quantized_torch_module = self.quantize_torch_module(torch_module)

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


class HeepstorchSequentialNetwork:
    def __init__(self, modules: OrderedDict[str, HeepstorchModule]):
        self.modules = modules

    @staticmethod
    def from_torch_sequential(seq: torch.nn.Sequential) -> 'HeepstorchSequentialNetwork':
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

        return HeepstorchSequentialNetwork(OrderedDict(
            [(sanitize_name(name), HeepstorchModule.from_torch_module(m)) for (name, m) in seq._modules.items()])
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
