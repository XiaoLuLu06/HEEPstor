#pragma once

#include <math/matrix.h>
#include <cstring>

class ReLU {
public:
    static void forward(Matrix<float>& m) {
        uint32_t* data_as_uint32 = reinterpret_cast<uint32_t*>(m.get_data());

        for (size_t i = 0; i < m.num_rows() * m.num_cols(); ++i) {
            // Branchless ReLU. Bitwise operation equivalent to  x * (x >= 0)

            // Negative -> 0x1 -> 0x00000000
            // Positive -> 0x0 -> 0xffffffff
            uint32_t mask = (data_as_uint32[i] >> 31) - 1;
            data_as_uint32[i] &= mask;
        }
    }
};