#pragma once

#include <math/matrix.h>
#include <cstring>

class BatchNorm2d {
public:
    static void forward(Matrix<float>& m, const Matrix<float>& scale, const Matrix<float>& shift) {
        const size_t NUM_PIXELS = m.num_rows();
        const size_t NUM_CHANNELS = m.num_cols();

        HEEPSTOR_ASSERT(scale.num_rows() == 1);
        HEEPSTOR_ASSERT(scale.num_cols() == NUM_CHANNELS);

        HEEPSTOR_ASSERT(shift.num_rows() == 1);
        HEEPSTOR_ASSERT(shift.num_cols() == NUM_CHANNELS);

        for (size_t j = 0; j < NUM_CHANNELS; ++j) {
            for (size_t i = 0; i < NUM_PIXELS; ++i) {
                m(i, j) = m(i, j) * scale(0, j) + shift(0, j);
            }
        }
    }
};