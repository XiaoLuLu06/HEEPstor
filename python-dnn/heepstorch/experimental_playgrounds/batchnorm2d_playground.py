import numpy as np
import torch
import torch.nn as nn


def batchnorm2d_inference(x, gamma, beta, running_mean, running_var, eps=1e-5):
    """
    BatchNorm2d inference implementation

    Args:
        x (np.ndarray): Input of shape (N, C, H, W)
        gamma (np.ndarray): Scale parameter of shape (C,)
        beta (np.ndarray): Shift parameter of shape (C,)
        running_mean (np.ndarray): Running mean of shape (C,)
        running_var (np.ndarray): Running variance of shape (C,)
        eps (float): Small constant for numerical stability

    Returns:
        np.ndarray: Normalized output of shape (N, C, H, W)
    """
    N, C, H, W = x.shape

    # Reshape input for broadcasting (the same scale and shift values are used for all pixels in a given channel)
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)

    # Precompute combined scale and shift parameters
    scale = gamma / np.sqrt(running_var + eps)
    shift = beta - gamma * running_mean / np.sqrt(running_var + eps)

    # Apply combined transformation
    out = x_reshaped * scale + shift

    # Reshape back to (N, C, H, W)
    return out.reshape(N, H, W, C).transpose(0, 3, 1, 2)


def test_batchnorm2d():
    """Test BatchNorm2d inference implementation against PyTorch"""

    # Test parameters
    batch_size, channels, height, width = 4, 3, 8, 8
    eps = 1e-5

    # Create random input and parameters
    x = np.random.randn(batch_size, channels, height, width)
    gamma = np.random.randn(channels)
    beta = np.random.randn(channels)
    running_mean = np.random.randn(channels)
    running_var = np.abs(np.random.randn(channels))  # variance must be positive

    # Custom implementation
    custom_output = batchnorm2d_inference(x, gamma, beta, running_mean, running_var, eps)

    # PyTorch implementation
    torch_bn = nn.BatchNorm2d(channels, eps=eps)
    with torch.no_grad():
        # Set parameters manually
        torch_bn.weight.data = torch.FloatTensor(gamma)
        torch_bn.bias.data = torch.FloatTensor(beta)
        torch_bn.running_mean.data = torch.FloatTensor(running_mean)
        torch_bn.running_var.data = torch.FloatTensor(running_var)
        torch_bn.eval()  # set to evaluation mode

        torch_output = torch_bn(torch.FloatTensor(x)).numpy()

    # Compare results
    max_diff = np.max(np.abs(custom_output - torch_output))
    print(f"Maximum difference: {max_diff:.10f}")

    outputs_match = np.allclose(custom_output, torch_output, rtol=1e-5, atol=1e-5)
    print(f"Outputs match: {outputs_match}")

    assert outputs_match


test_batchnorm2d()
