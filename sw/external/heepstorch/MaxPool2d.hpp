#pragma once

#include <math/matrix.h>
#include <limits>

class MaxPool2d {
public:
    static void forward(const Matrix<float>& in, Matrix<float>& out, size_t kernel_size, size_t input_height, size_t input_width) {
        HEEPSTOR_ASSERT(in.num_rows() == input_height * input_width);
        size_t NUM_INPUT_CHANNELS = in.num_cols();

        // Calculate output dimensions
        size_t out_height = (input_height - kernel_size) / kernel_size + 1;
        size_t out_width = (input_width - kernel_size) / kernel_size + 1;

        // Verify output shape
        HEEPSTOR_ASSERT(out.num_rows() == out_height * out_width);
        HEEPSTOR_ASSERT(out.num_cols() == NUM_INPUT_CHANNELS);

        // Perform max pooling
        for (size_t c = 0; c < NUM_INPUT_CHANNELS; c++) {
            for (size_t i = 0; i < out_height; i++) {
                for (size_t j = 0; j < out_width; j++) {
                    // Get starting positions for this window
                    size_t h_start = i * kernel_size;
                    size_t w_start = j * kernel_size;

                    // Initialize max value for this window
                    float current_max = -std::numeric_limits<float>::infinity();

                    // Iterate over each element in the window
                    for (size_t ki = 0; ki < kernel_size; ki++) {
                        for (size_t kj = 0; kj < kernel_size; kj++) {
                            size_t h_idx = h_start + ki;
                            size_t w_idx = w_start + kj;
                            float val = in(h_idx * input_width + w_idx, c);
                            current_max = std::max(current_max, val);
                        }
                    }

                    out( i * out_width + j, c) = current_max;
                }
            }
        }
    }
};