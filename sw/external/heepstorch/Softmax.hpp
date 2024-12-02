#pragma once

#include <math/floating_point_ops.h>
#include <math/matrix.h>
#include <algorithm>
#include <limits>

class Softmax {
public:
    // Performs in-place softmax across every row of the matrix
    void forward(Matrix<float>& m) {
        const size_t rows = m.num_rows();
        const size_t cols = m.num_cols();
        float* data = m.get_data();

        // Process each row independently
        for (size_t r = 0; r < rows; ++r) {
            float* row = data + (r * cols);

            // 1. Find maximum element in the row for numerical stability
            float max_val = std::numeric_limits<float>::lowest();
            for (size_t c = 0; c < cols; ++c) {
                max_val = std::max(max_val, row[c]);
            }

            // 2. Compute exp(x - max) for each element and sum
            float sum = 0.0f;
            for (size_t c = 0; c < cols; ++c) {
                const float val = fastexp(row[c] - max_val);
                row[c] = val;
                sum += val;
            }

            // 3. Normalize by dividing each element by the sum
            const float inv_sum = 1.0f / sum;
            for (size_t c = 0; c < cols; ++c) {
                row[c] *= inv_sum;
            }
        }
    }
};