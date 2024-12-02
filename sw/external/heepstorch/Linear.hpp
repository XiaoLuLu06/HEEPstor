#pragma once

#include <drivers/systolic_array/systolic_array.h>
#include <math/matrix.h>
#include <math/packed_int8_matrix.h>

class Linear {
public:
    static void forward(SystolicArray& systolic_array, const Matrix<float>& activations, const PackedInt8Matrix& quantized_weights,
                        float quantized_weight_scale, const Matrix<float>& bias, Matrix<float>& out) {

        HEEPSTOR_ASSERT(bias.num_rows() == 1 && bias.num_cols() == quantized_weights.num_cols() &&
                        activations.num_cols() == quantized_weights.num_rows());

        // 1. Perform the matrix multiplication
        systolic_array.matrix_matrix_multiply(activations, quantized_weights, out);

        // 2. Apply the post-quantization scaling
        out *= quantized_weight_scale;

        // 3. Apply bias
        float* out_ptr = out.get_data();
        const float* bias_ptr = bias.get_data();

        for (int r = 0; r < out.num_rows(); ++r) {
            for (int c = 0; c < out.num_cols(); ++c) {
                *out_ptr += bias_ptr[c];
                out_ptr++;
            }
        }
    }
};