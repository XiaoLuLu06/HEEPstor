#pragma once

#include <math/matrix.h>

class Flatten {
public:
    static void forward(const Matrix<float>& input, Matrix<float>& output) {
        const size_t image_width_times_height = input.num_rows();
        const size_t num_input_channels = input.num_cols();

        HEEPSTOR_ASSERT(output.num_rows() == 1);
        HEEPSTOR_ASSERT(output.num_cols() == image_width_times_height * num_input_channels);

        // Flatten column by column into the output row matrix
        for (size_t channel = 0; channel < num_input_channels; channel++) {
            for (size_t i = 0; i < image_width_times_height; i++) {
                // Place each element from the input into its position in the flattened output
                output(0, channel * image_width_times_height + i) = input(i, channel);
            }
        }
    }
};