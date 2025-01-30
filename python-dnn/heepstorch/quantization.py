from enum import Enum
import numpy as np
import numpy.typing as npt


# Quantizer to use. For now, only per_tensor_symmetric is supported for efficiency purposes.
class QuantizationScheme(Enum):
    per_tensor_symmetric = 1

    DEFAULT = per_tensor_symmetric


def quantize(
        matrix_fp32: npt.NDArray[np.float32],
        qscheme: QuantizationScheme = QuantizationScheme.DEFAULT
) -> (npt.NDArray[np.int8], np.float32):
    """
    Quantizes matrix x from fp32 to int8. In per_tensor_symmetric, it returns matrix_int8 and scaling factor s such that
        matrix_int8 * s = matrix_fp32
    Args:
        matrix_fp32: Matrix to quantize.
        qscheme: Quantization scheme to employ. Currently, only 'per_tensor_symmetric' is supported.
    Returns:
        A tuple with the int8 quantized matrix and the scaling factor s.
    """

    # For now, we only support per_tensor_symmetric
    assert qscheme == QuantizationScheme.per_tensor_symmetric

    max_abs_val = np.max(np.abs(matrix_fp32))
    scale = max_abs_val / 127.0

    # Note: Heepstor's hybrid fp32_int8 multiplier only supports integers in the range [-127, 127] due to
    #   using a sign-magnitude representation, so clamp to that range.
    matrix_int8 = np.clip(
        np.round(matrix_fp32 / scale),
        -127, 127
    ).astype(np.int8)

    return matrix_int8, scale


def dequantize(
        matrix_int8: npt.NDArray[np.int8],
        scale_factor: np.float32,
        qscheme: QuantizationScheme = QuantizationScheme.DEFAULT
) -> npt.NDArray[np.float32]:
    """
    Dequantizes a matrix from fp32 to int8, given the scaling factor.

    Args:
        matrix_int8: Matrix to dequantize.
        scale_factor: Scaling factor used during quantization.
        qscheme: Quantization scheme to employ. Currently, only 'per_tensor_symmetric' is supported.

    Returns:
        Dequantized fp32 matrix.
    """

    # For now, we only support per_tensor_symmetric
    assert qscheme == QuantizationScheme.per_tensor_symmetric

    return matrix_int8.astype(np.float32) * scale_factor


def dequantize_matmult_result(
        res_fp32: npt.NDArray[np.float32],
        scale_factor: np.float32,
        qscheme: QuantizationScheme = QuantizationScheme.DEFAULT
) -> npt.NDArray[np.float32]:
    """
    Dequantizes the result of multiplying res_fp32 = A_fp32 @ B_int8 using the provided scaling factor.
    Args:
        res_fp32: Resulting matrix of matrix multiplication to dequantize.
        scale_factor: Scaling factor used during quantization of B_int8.
        qscheme: Quantization scheme to employ. Currently, only 'per_tensor_symmetric' is supported.

    Returns:
        Dequantized fp32 matrix.
    """

    # For now, we only support per_tensor_symmetric
    assert qscheme == QuantizationScheme.per_tensor_symmetric
    return res_fp32 * scale_factor


def fake_quantize(
        matrix_fp32: npt.NDArray[np.float32],
        qscheme: QuantizationScheme = QuantizationScheme.DEFAULT
) -> npt.NDArray[np.float32]:
    """
    Quantizes and dequantizes a matrix, to simulate the precision lost due to quantization.
    Args:
        matrix_fp32: Matrix to dequantize and dequantize.
        qscheme: Quantization scheme to employ.

    Returns:
        matrix_fp32 after applying quantization and dequantization
    """
    matrix_int8, scaling_factor = quantize(matrix_fp32, qscheme)
    return dequantize(matrix_int8, scaling_factor, qscheme)
