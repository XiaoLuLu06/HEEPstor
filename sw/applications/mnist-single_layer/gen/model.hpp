#pragma once

#include <drivers/systolic_array/systolic_array.h>
#include <math/matrix.h>
#include <math/packed_int8_matrix_tile.h>
#include <heepstorch/Layers.hpp>
#include <profiling/performance_timer.hpp>
#include "model_parameters.hpp"

class Model {
public:
    static constexpr size_t NUM_INPUT_FEATURES = 144;
    static constexpr size_t NUM_OUTPUT_FEATURES = 10;

    // inputs is a matrix of shape [BATCH_SIZE x NUM_INPUT_FEATURES], outputs is a matrix of shape [BATCH_SIZE x NUM_OUTPUT_FEATURES]
    static void infer(SystolicArray& systolic_array, const Matrix<float>& inputs, Matrix<float>& outputs,
                      CheckpointPerformanceTimerDisplayConfig display_config) {
        HEEPSTOR_ASSERT(inputs.num_cols() == NUM_INPUT_FEATURES);
        HEEPSTOR_ASSERT(outputs.num_cols() == NUM_OUTPUT_FEATURES);
        HEEPSTOR_ASSERT(inputs.num_rows() == outputs.num_rows());

        const size_t BATCH_SIZE = inputs.num_rows();

        //////////////////////////////////////////////
        //  Wrap the model parameters into matrices.
        //////////////////////////////////////////////

        CheckpointPerformanceTimer<2> performance_timer{display_config};
        performance_timer.reset();

        // fc0: Linear
        const PackedInt8Matrix fc0_weights = PackedInt8Matrix::from_const_pointer(ModelParameters::fc0_weight_data, 144, 10);
        const Matrix<float> fc0_bias = Matrix<float>::from_const_pointer(ModelParameters::fc0_bias_data, 1, 10);

        //////////////////////////////////////////////
        //  Declare intermediate buffers
        //////////////////////////////////////////////



        //////////////////////////////////////////////
        //  Perform inference steps
        //////////////////////////////////////////////

        // 1. fc0: Linear
        Linear::forward(systolic_array, inputs, fc0_weights, ModelParameters::fc0_weight_scale, fc0_bias, outputs);
        performance_timer.checkpoint();

        // 2. Final Softmax
        Softmax::forward(outputs);
        performance_timer.checkpoint();

        performance_timer.finalize({"fc0 (Linear)", "Final Softmax"});
    }
};