#pragma once

#include <drivers/systolic_array/systolic_array.h>
#include <math/matrix.h>
#include <math/packed_int8_matrix_tile.h>
#include <heepstorch/Layers.hpp>
#include "model_parameters.hpp"

class Model {
public:
    static constexpr size_t NUM_INPUT_FEATURES = 144;
    static constexpr size_t NUM_OUTPUT_FEATURES = 10;

    // inputs is a matrix of shape [BATCH_SIZE x NUM_INPUT_FEATURES], outputs is a matrix of shape [BATCH_SIZE x NUM_OUTPUT_FEATURES]
    static void infer(SystolicArray& systolic_array, const Matrix<float>& inputs, Matrix<float>& outputs) {
        HEEPSTOR_ASSERT(inputs.num_rows() == NUM_INPUT_FEATURES);
        HEEPSTOR_ASSERT(outputs.num_rows() == NUM_OUTPUT_FEATURES);
        HEEPSTOR_ASSERT(inputs.num_cols() == outputs.num_cols());

        const size_t BATCH_SIZE = inputs.num_cols();

        //////////////////////////////////////////////
        //  Wrap the model parameters into matrices.
        //////////////////////////////////////////////

        // fc0: Linear
        const PackedInt8Matrix fc0_weights = PackedInt8Matrix::from_const_pointer(ModelParameters::fc0_weight_data, 144, 20);
        const Matrix<float> fc0_bias = Matrix<float>::from_const_pointer(ModelParameters::fc0_bias_data, 1, 20);

        // fc1: Linear
        const PackedInt8Matrix fc1_weights = PackedInt8Matrix::from_const_pointer(ModelParameters::fc1_weight_data, 20, 10);
        const Matrix<float> fc1_bias = Matrix<float>::from_const_pointer(ModelParameters::fc1_bias_data, 1, 10);

        //////////////////////////////////////////////
        //  Declare intermediate buffers
        //////////////////////////////////////////////

        Matrix<float> intermediate_buf_1(BATCH_SIZE, 20);

        //////////////////////////////////////////////
        //  Perform inference steps
        //////////////////////////////////////////////

        // 1. fc0: Linear
        Linear::forward(systolic_array, inputs, fc0_weights, ModelParameters::fc0_weight_scale, fc0_bias, intermediate_buf_1);

        // 2. relu0: ReLU
        ReLU::forward(intermediate_buf_1);

        // 3. fc1: Linear
        Linear::forward(systolic_array, intermediate_buf_1, fc1_weights, ModelParameters::fc1_weight_scale, fc1_bias, outputs);

        // 4. Final Softmax
        Softmax::forward(outputs);
    }
};