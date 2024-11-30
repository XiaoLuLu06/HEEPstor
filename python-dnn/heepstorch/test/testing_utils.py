import numpy as np
import unittest

import numpy as np
import unittest


def assert_matrix_allclose(actual, expected, t: unittest.TestCase, rtol=1e-7, atol=1e-14, err_msg=""):
    """
    Assert that two matrices are approximately equal, suitable for quantization testing.
    Uses Frobenius norm to check overall matrix difference relative to matrix magnitude.

    Parameters:
    -----------
    actual : array_like
        The matrix to test (e.g., quantized and dequantized matrix)
    expected : array_like
        The expected matrix (e.g., original matrix)
    t : unittest.TestCase
        Test case instance for assertions
    rtol : float, optional
        Relative tolerance for matrix-wise difference (default: 1e-7)
    atol : float, optional
        Absolute tolerance for when expected values are very close to zero (default: 1e-14)
    err_msg : str, optional
        Custom error message to display on failure
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    # Check shapes match
    t.assertEqual(actual.shape, expected.shape,
                  f"Shapes don't match: {actual.shape} != {expected.shape}")

    # Compute matrix norms
    diff_norm = np.linalg.norm(actual - expected)
    expected_norm = np.linalg.norm(expected)

    # For matrices close to zero, use absolute tolerance
    if expected_norm < atol:
        t.assertTrue(diff_norm <= atol,
                     f"{err_msg}\nMatrix difference {diff_norm} exceeds absolute tolerance {atol}")
        return

    # Otherwise use relative tolerance on matrix norm
    relative_error = diff_norm / expected_norm
    t.assertTrue(relative_error <= rtol,
                 f"{err_msg}\nRelative matrix error {relative_error} exceeds relative tolerance {rtol}")
