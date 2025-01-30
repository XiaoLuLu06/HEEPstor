import numpy as np
import torch
import torch.nn as nn


def im2row(input, kernel_size, H, W):
    """
    Transform the input image into a 2D matrix using the im2col technique,
    but generate the transpose of the original im2col matrix (which we call im2row).
    This version uses explicit for loops instead of slicing and reshaping to be closer to the C code.

    Args:
        input (np.ndarray): Input image of shape (H*W, C_in). Each of the input channels is a row of the matrix.
        kernel_size (int): Size of the kernel (N).

    Returns:
        np.ndarray: Transformed 2D matrix of shape (H_out * W_out, C_in * N * N).
    """
    _, C_in = input.shape
    N = kernel_size
    H_out = H - N + 1
    W_out = W - N + 1

    # Initialize the output matrix
    res = np.zeros((H_out * W_out, C_in * N * N))

    # Iterate over each patch location (i, j)
    for i in range(H_out):
        for j in range(W_out):
            # Flatten the patch manually using for loops
            patch_idx = 0  # Index for the flattened patch
            for c in range(C_in):  # Iterate over input channels
                for ki in range(N):  # Iterate over kernel rows
                    for kj in range(N):  # Iterate over kernel columns
                        # Fill the patch element into the res matrix
                        res[i * W_out + j, patch_idx] = input[(i + ki) * W + (j + kj), c]
                        patch_idx += 1

    return res


def conv2d_matrix_multiplication_transpose(input, kernels):
    """
    Perform Conv2d operation using matrix multiplication with the input_cols @ kernel_matrix layout.

    Args:
        input (np.ndarray): Input image of shape (C_in, H, W).
        kernels (np.ndarray): Kernels of shape (C_out, C_in, N, N).

    Returns:
        np.ndarray: Output image of shape (C_out, H_out, W_out).
    """
    C_out, C_in, N, _ = kernels.shape
    _, H, W = input.shape
    H_out = H - N + 1
    W_out = W - N + 1

    # Reshape kernels into a 2D matrix (each column is a flattened kernel)
    kernel_matrix = kernels.reshape(C_out, -1).T  # Shape: (C_in * N * N, C_out)

    # Convert the input to columns: shape (H*W, C_in), where each column is one full image
    input_cols = input.reshape(C_in, H * W).T

    ############################################3
    # Actual convolution code (the rest of the function is marshalling PyTorch data layout to Heepstorch)

    # Transform input using im2row
    im2row_res = im2row(input_cols, N, H, W)  # Shape: (H_out * W_out, C_in * N * N)

    # Perform matrix multiplication
    output_cols = im2row_res @ kernel_matrix  # Shape: (H_out * W_out, C_out)

    ############################################3

    # Each column of output_cols is a channel of H_out * W_out.

    # Reshape the output into the final image shape
    output = output_cols.T.reshape(C_out, H_out, W_out)

    return output


# Test against PyTorch Conv2d
def test_conv2d_transpose():
    # Define input and kernels
    C_in, C_out, N = 3, 5, 3
    H, W = 10, 10
    input_np = np.random.randn(C_in, H, W).astype(np.float32)
    kernels_np = np.random.randn(C_out, C_in, N, N).astype(np.float32)

    # PyTorch Conv2d
    input_torch = torch.from_numpy(input_np).unsqueeze(0)  # Add batch dimension
    conv = nn.Conv2d(C_in, C_out, N, bias=False)
    conv.weight.data = torch.from_numpy(kernels_np)
    output_torch = conv(input_torch).squeeze(0).detach().numpy()  # Remove batch dimension

    # Manual Conv2d with transposed layout
    output_manual = conv2d_matrix_multiplication_transpose(input_np, kernels_np)

    # Compare results
    assert np.allclose(output_torch, output_manual, atol=1e-5), "Results do not match!"
    print("Max absolute difference:", np.max(np.abs(output_torch - output_manual)))
    print("Test passed! Results match.")


# Run the test
test_conv2d_transpose()
