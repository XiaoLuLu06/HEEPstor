#pragma once

#include <drivers/systolic_array/systolic_array.h>
#include <math/matrix.h>
#include <math/packed_int8_matrix_tile.h>
#include <heepstorch/Layers.hpp>
#include "model_parameters.h"

class Model {
    static constexpr size_t NUM_INPUT_FEATURES = 28 * 28;
    static constexpr size_t NUM_OUTPUT_FEATURES = 10;

    // inputs is a matrix of shape [BATCH_SIZE x NUM_INPUT_FEATURES], outputs is a matrix of shape [BATCH_SIZE x NUM_OUTPUT_FEATURES]
    static void infer(SystolicArray& systolic_array, const Matrix<float>& inputs, Matrix<float>& outputs) {
        HEEPSTOR_ASSERT(inputs.num_rows() == NUM_INPUT_FEATURES);
        HEEPSTOR_ASSERT(outputs.num_rows() == NUM_OUTPUT_FEATURES);
        HEEPSTOR_ASSERT(inputs.num_cols() == outputs.num_cols());

        // Wrap the parameters into matrices. Note: The ugly const cast is necessary because

        const PackedInt8Matrix fc0_weights = PackedInt8Matrix::from_const_pointer(ModelParameters::fc0_weight_data, 28 * 28, 10);
        const Matrix<float> fc0_bias = Matrix<float>::from_const_pointer(ModelParameters::fc0_bias_data, 1, 10);

        // TODO: Run everything

        // 1. fc0 layer
        Linear::forward(systolic_array, inputs, fc0_weights, ModelParameters::fc0_weight_scale, fc0_bias, outputs);

        // 2. Softmax layer (optional)
        Softmax::forward(outputs);
    }
};