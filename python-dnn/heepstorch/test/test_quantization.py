import unittest
import heepstorch.quantization as quant
import testing_utils as t
import numpy as np

from heepstorch.quantization import fake_quantize


class QuantizationTests(unittest.TestCase):
    def test_back_and_forth_quantization(self):
        np.random.seed(0)

        X_fp32 = np.random.randn(7, 9).astype(np.float32) * 10
        X_int8, scale_factor = quant.quantize(X_fp32)

        self.assertEqual(X_int8.dtype, np.int8)

        res_fp32 = quant.dequantize(X_int8, scale_factor)

        # print(X_fp32)
        # print(X_int8)
        # print(res_fp32)

        t.assert_matrix_allclose(res_fp32, X_fp32, self, rtol=0.01, atol=0.001)

    def test_dequantization_mat_mult(self):
        np.random.seed(1)

        A_fp32 = np.random.randn(17, 53).astype(np.float32) * 10
        B_fp32 = np.random.randn(53, 21).astype(np.float32) * 10

        B_int8, scale_factor = quant.quantize(B_fp32)

        res_dequantized = quant.dequantize_matmult_result(A_fp32 @ B_int8, scale_factor)
        res_actual = A_fp32 @ B_fp32

        # print(res_dequantized)
        # print(res_actual)

        t.assert_matrix_allclose(res_dequantized, res_actual, self, rtol=0.01, atol=0.001)

    def test_fake_quantize(self):
        np.random.seed(2)

        A_fp32 = np.random.randn(17, 53).astype(np.float32) * 10
        B_fp32 = np.random.randn(53, 21).astype(np.float32) * 10

        B_int8, scale_factor = quant.quantize(B_fp32)

        res_dequantized = quant.dequantize_matmult_result(A_fp32 @ B_int8, scale_factor)
        res_fake_dequantized = A_fp32 @ fake_quantize(B_fp32)

        t.assert_matrix_allclose(res_fake_dequantized, res_dequantized, self, rtol=1e-6, atol=1e-13)


if __name__ == '__main__':
    unittest.main()
