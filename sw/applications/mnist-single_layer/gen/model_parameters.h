#pragma once

#include <stdint.h>

class ModelParameters {
public:
    static constexpr float fc0_weight_scale = 0xdeadbeef;

    static constexpr uint32_t fc0_weight_data[28 * 28 * 10] = {1, 2, 3};
    static constexpr float fc0_bias_data[10] = {1, 2, 3};
};