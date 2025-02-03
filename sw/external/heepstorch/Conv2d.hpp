#pragma once

#include <math/matrix.h>
#include <math/packed_int8_matrix.h>
#include <Linear.hpp>

class Conv2d {
public:
    static void forward(SystolicArray& systolic_array, const Matrix<float>& activations, const PackedInt8Matrix& quantized_kernel_weights,
                        float quantized_kernel_weight_scale, const Matrix<float>& bias, Matrix<float>& out, size_t kernel_size,
                        size_t num_input_channels, size_t height, size_t width) {

        HEEPSTOR_ASSERT(activations.num_rows() == height * width);
        HEEPSTOR_ASSERT(activations.num_cols() == num_input_channels);

        size_t height_out = height - kernel_size + 1;
        size_t width_out = width - kernel_size + 1;

        // Compute im2row transformation
        Matrix<float> im2row_res(height_out * width_out, num_input_channels * kernel_size * kernel_size);
        im2row(activations, im2row_res, kernel_size, num_input_channels, height, width);

        // Multiply the im2row transformed activations with the quantized pre-laid out kernel weights
        Linear::forward(systolic_array, im2row_res, quantized_kernel_weights, quantized_kernel_weight_scale, bias, out);
    }

private:
    static void im2row(const Matrix<float>& input, Matrix<float>& output, size_t kernel_size, size_t num_input_channels, size_t height,
                       size_t width) {
        HEEPSTOR_ASSERT(input.num_rows() == width * height);
        HEEPSTOR_ASSERT(output.num_cols() == num_input_channels * kernel_size * kernel_size);

        size_t height_out = height - kernel_size + 1;
        size_t width_out = width - kernel_size + 1;

        // Iterate over output positions
        for (size_t i = 0; i < height_out; i++) {
            for (size_t j = 0; j < width_out; j++) {
                size_t patch_idx = 0;  // Index for the flattened patch

                // Iterate over input channels
                for (size_t c = 0; c < num_input_channels; c++) {
                    // Iterate over kernel dimensions
                    for (size_t ki = 0; ki < kernel_size; ki++) {
                        for (size_t kj = 0; kj < kernel_size; kj++) {
                            // Calculate output and input indices
                            output(i * width_out + j, patch_idx) = input((i + ki) * width + (j + kj), c);
                            patch_idx++;
                        }
                    }
                }
            }
        }
    }
};