from abc import ABC, abstractmethod
import torch.nn
import numpy.typing as npt
import numpy as np


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

    def __init__(self, torch_module: torch.nn.Module):
        self.non_quantized_torch_module = torch_module

    def forward_quantized(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        raise NotImplementedError

    def get_quantized_torch_module(self) -> torch.nn.Module:
        raise NotImplementedError
