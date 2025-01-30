import numpy as np


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
