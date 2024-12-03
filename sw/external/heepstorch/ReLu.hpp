#pragma once

#include <math/matrix.h>

class ReLU {
public:
    static void forward(Matrix<float>& m) {
        float* data = m.get_data();

        for (size_t i = 0; i < m.num_rows() * m.num_cols(); ++i) {
            // Branchless ReLU: x * (x >= 0)
            // x >= 0 evaluates to 1.0f if true, 0.0f if false
            data[i] *= (data[i] >= 0.0f);
        }
    }
};