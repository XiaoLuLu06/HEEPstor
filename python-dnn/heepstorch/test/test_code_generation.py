import unittest
import heepstorch.quantization as quant
import heepstorch.code_generator as cg
import testing_utils as t
import numpy as np


class CodeGenerationTests(unittest.TestCase):
    def test_weight_packing_to_c_array(self):
        np.random.seed(0)

        X_fp32 = np.random.randn(7, 9).astype(np.float32) * 10
        X_int8, scale = quant.quantize(X_fp32)

        X_int8_c_array, size = cg.CodeGenerator.quantized_weights_to_packed_c_array(X_int8, 'test')
        print(X_int8)
        print(X_int8_c_array)
        print(f'Size: {size}')

    def test_bias_to_c_array(self):
        np.random.seed(0)

        def test(mu, sigma, name):
            bias = sigma * np.random.randn(13) + mu
            bias_c_array, size = cg.CodeGenerator.bias_to_c_array(bias, name)
            print(bias_c_array)
            print(f'\n{name}: {bias}')
            print(f'Size: {size}')

        test(0, 1, 'normal')
        test(0, 1e-10, 'small')
        test(5, 50, 'big')
        test(0, 1e10, 'huge')


if __name__ == '__main__':
    unittest.main()
