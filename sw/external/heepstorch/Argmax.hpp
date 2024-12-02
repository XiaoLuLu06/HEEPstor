#pragma once

#include <math/matrix.h>
#include <cstddef>

class Argmax {
public:
    // Returns the index of the maximum value in the specified row
    static size_t forward(const Matrix<float>& m, size_t row) {
        const size_t cols = m.num_cols();
        HEEPSTOR_ASSERT(row < m.num_rows() && "Row index out of bounds");

        const float* row_data = m.get_data() + (row * cols);

        size_t max_idx = 0;
        float max_val = row_data[0];

        for (size_t c = 1; c < cols; ++c) {
            if (row_data[c] > max_val) {
                max_val = row_data[c];
                max_idx = c;
            }
        }

        return max_idx;
    }

    // Computes argmax for each row of the input matrix and stores results in out_indices
    //
    // Parameters:
    // - m: Input matrix of shape [N x C] where N is number of rows and C is number of columns
    // - out_indices: Output matrix of shape [1 x N] that will store the argmax index for each row.
    //               Must be a 1xN matrix where N matches the number of rows in m.
    //
    // The output matrix will contain the column index of the maximum value for each row of m.
    // For example, if m is:
    // [[1.0, 3.0, 2.0],
    //  [4.0, 1.0, 5.0]]
    // Then out_indices will be:
    // [1, 2]  // indicating max of first row is at column 1, max of second row is at column 2
    static void forward_batch(const Matrix<float>& m, Matrix<int>& out_indices) {
        const size_t rows = m.num_rows();
        const size_t cols = m.num_cols();
        const float* data = m.get_data();

        // Verify output matrix has correct shape
        HEEPSTOR_ASSERT(out_indices.num_rows() == 1 && out_indices.num_cols() == rows &&
                        "Output matrix must be 1xN where N is the number of rows in input matrix");

        int* indices = out_indices.get_data();

        for (size_t r = 0; r < rows; ++r) {
            const float* row_data = data + (r * cols);

            size_t max_idx = 0;
            float max_val = row_data[0];

            for (size_t c = 1; c < cols; ++c) {
                if (row_data[c] > max_val) {
                    max_val = row_data[c];
                    max_idx = c;
                }
            }

            indices[r] = static_cast<int>(max_idx);
        }
    }
};