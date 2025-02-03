// DO NOT EDIT. File generated automatically, your changes will be lost when regenerated.
#pragma once

#include <drivers/systolic_array/systolic_array.h>
#include <math/matrix.h>
#include <math/packed_int8_matrix_tile.h>
#include <heepstorch/Layers.hpp>
#include <profiling/performance_timer.hpp>
#include "model_parameters.hpp"

class Model {
public:
    // Input configuration
    static constexpr size_t NUM_INPUT_FEATURES = 144;

    // Output configuration
    static constexpr size_t NUM_OUTPUT_FEATURES = 10;

    static void infer(SystolicArray& systolic_array, const Matrix<float>& inputs, Matrix<float>& outputs,
                     CheckpointPerformanceTimerDisplayConfig display_config) {

        //////////////////////////////////////////////
        // Validate input and output sizes
        //////////////////////////////////////////////

        HEEPSTOR_ASSERT(inputs.num_cols() == NUM_INPUT_FEATURES);
        HEEPSTOR_ASSERT(outputs.num_cols() == NUM_OUTPUT_FEATURES);
        HEEPSTOR_ASSERT(inputs.num_rows() == outputs.num_rows());

        const size_t batch_size = inputs.num_rows();

        //////////////////////////////////////////////
        //  Wrap the model parameters into matrices.
        //////////////////////////////////////////////

        CheckpointPerformanceTimer<4> performance_timer{display_config};
        performance_timer.reset();

        // fc0: Linear
        const PackedInt8Matrix fc0_weights = PackedInt8Matrix::from_const_pointer(ModelParameters::fc0_weight_data, 144, 20);
        const Matrix<float> fc0_bias = Matrix<float>::from_const_pointer(ModelParameters::fc0_bias_data, 1, 20);

        // fc1: Linear
        const PackedInt8Matrix fc1_weights = PackedInt8Matrix::from_const_pointer(ModelParameters::fc1_weight_data, 20, 10);
        const Matrix<float> fc1_bias = Matrix<float>::from_const_pointer(ModelParameters::fc1_bias_data, 1, 10);

        //////////////////////////////////////////////
        //  Declare intermediate buffers
        //////////////////////////////////////////////

        Matrix<float> intermediate_buf_1(batch_size, 20);

        //////////////////////////////////////////////
        //  Perform inference steps
        //////////////////////////////////////////////

        // 1. fc0: Linear
        Linear::forward(systolic_array, inputs, fc0_weights, ModelParameters::fc0_weight_scale, fc0_bias, intermediate_buf_1);
        performance_timer.checkpoint();

        // 2. relu0: ReLU
        ReLU::forward(intermediate_buf_1);
        performance_timer.checkpoint();

        // 3. fc1: Linear
        Linear::forward(systolic_array, intermediate_buf_1, fc1_weights, ModelParameters::fc1_weight_scale, fc1_bias, outputs);
        performance_timer.checkpoint();

        // 4. Final Softmax
        Softmax::forward(outputs);
        performance_timer.checkpoint();

        performance_timer.finalize({"fc0 (Linear)", "relu0 (ReLU)", "fc1 (Linear)", "Final Softmax"});
    }
};