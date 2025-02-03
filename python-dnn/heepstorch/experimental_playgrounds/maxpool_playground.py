import numpy as np
import torch
import torch.nn as nn


def maxpool2d(x, kernel_size):
    """
    Implements 2D max pooling with equal kernel size and stride, no padding

    Args:
        x (np.ndarray): Input array of shape (batch_size, channels, height, width)
        kernel_size (int): Size of the pooling window

    Returns:
        np.ndarray: Pooled output
    """
    batch_size, C_in, H, W = x.shape

    assert batch_size == 1

    # Calculate output dimensions
    out_height = (H - kernel_size) // kernel_size + 1
    out_width = (W - kernel_size) // kernel_size + 1

    input_cols = x.reshape(C_in, H * W).T

    # Initialize output array

    ##########################################################
    # Actual MaxPool2d code (the rest of the function is marshalling PyTorch data layout to Heepstorch)

    output_cols = np.zeros((out_height * out_width, C_in))

    # Perform max pooling
    for c in range(C_in):
        for i in range(out_height):
            for j in range(out_width):
                # Get starting positions for this window
                h_start = i * kernel_size
                w_start = j * kernel_size

                # Initialize max value for this window
                current_max = float('-inf')

                # Iterate over each element in the window
                for ki in range(kernel_size):
                    for kj in range(kernel_size):
                        h_idx = h_start + ki
                        w_idx = w_start + kj
                        val = input_cols[h_idx * W + w_idx, c]
                        current_max = max(current_max, val)

                output_cols[i * out_width + j, c] = current_max

    ##########################################################

    output = output_cols.T.reshape(1, C_in, out_height, out_width)
    return output


def test_maxpool2d():
    """Test the maxpool2d function against PyTorch's implementation"""

    # Create random input tensor
    batch_size, channels = 1, 3
    height, width = 10, 10
    kernel_size = 4

    # Generate random input
    x = np.random.randn(batch_size, channels, height, width)

    # Custom implementation
    custom_output = maxpool2d(x, kernel_size)

    # PyTorch implementation
    torch_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size, padding=0)
    torch_input = torch.FloatTensor(x)
    torch_output = torch_pool(torch_input).numpy()

    # Compare results
    max_diff = np.max(np.abs(custom_output - torch_output))
    print(f"Maximum difference: {max_diff:.10f}")

    is_close = np.allclose(custom_output, torch_output, rtol=1e-5, atol=1e-5)
    print(f"Outputs match: {is_close}")

    assert is_close


test_maxpool2d()
