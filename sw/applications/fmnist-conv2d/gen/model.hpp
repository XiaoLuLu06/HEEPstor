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
    static constexpr size_t NUM_INPUT_CHANNELS = 1;
    static constexpr size_t INPUT_HEIGHT = 14;
    static constexpr size_t INPUT_WIDTH = 14;

    // Output configuration
    static constexpr size_t NUM_OUTPUT_FEATURES = 10;

    static void infer(SystolicArray& systolic_array, const Matrix<float>& inputs, Matrix<float>& outputs,
                     CheckpointPerformanceTimerDisplayConfig display_config) {

        //////////////////////////////////////////////
        // Validate input and output sizes
        //////////////////////////////////////////////

        HEEPSTOR_ASSERT(inputs.num_rows() == INPUT_HEIGHT * INPUT_WIDTH);
        HEEPSTOR_ASSERT(inputs.num_cols() == NUM_INPUT_CHANNELS);
        HEEPSTOR_ASSERT(outputs.num_cols() == NUM_OUTPUT_FEATURES);
        HEEPSTOR_ASSERT(outputs.num_rows() == 1);

        const size_t batch_size = 1;

        //////////////////////////////////////////////
        //  Wrap the model parameters into matrices.
        //////////////////////////////////////////////

        CheckpointPerformanceTimer<7> performance_timer{display_config};
        performance_timer.reset();

        // conv1: Conv2d
        const PackedInt8Matrix conv1_kernel_weights = PackedInt8Matrix::from_const_pointer(ModelParameters::conv1_kernel_weight_data, 16, 16);
        const Matrix<float> conv1_bias = Matrix<float>::from_const_pointer(ModelParameters::conv1_bias_data, 1, 16);

        // conv2: Conv2d
        const PackedInt8Matrix conv2_kernel_weights = PackedInt8Matrix::from_const_pointer(ModelParameters::conv2_kernel_weight_data, 256, 8);
        const Matrix<float> conv2_bias = Matrix<float>::from_const_pointer(ModelParameters::conv2_bias_data, 1, 8);

        // fc1: Linear
        const PackedInt8Matrix fc1_weights = PackedInt8Matrix::from_const_pointer(ModelParameters::fc1_weight_data, 512, 10);
        const Matrix<float> fc1_bias = Matrix<float>::from_const_pointer(ModelParameters::fc1_bias_data, 1, 10);

        //////////////////////////////////////////////
        //  Declare intermediate buffers
        //////////////////////////////////////////////

        Matrix<float> intermediate_buf_1(11*11, 16);
        Matrix<float> intermediate_buf_2(8*8, 8);
        Matrix<float> intermediate_buf_3(batch_size, 512);

        //////////////////////////////////////////////
        //  Perform inference steps
        //////////////////////////////////////////////

        // 1. conv1: Conv2d
        Conv2d::forward(systolic_array, inputs, conv1_kernel_weights, ModelParameters::conv1_kernel_weight_scale, conv1_bias, intermediate_buf_1);
        performance_timer.checkpoint();

        // 2. relu1: ReLU
        ReLU::forward(intermediate_buf_1);
        performance_timer.checkpoint();

        // 3. conv2: Conv2d
        Conv2d::forward(systolic_array, intermediate_buf_1, conv2_kernel_weights, ModelParameters::conv2_kernel_weight_scale, conv2_bias, intermediate_buf_2);
        performance_timer.checkpoint();

        // 4. relu2: ReLU
        ReLU::forward(intermediate_buf_2);
        performance_timer.checkpoint();

        // 5. flatten: Flatten
        Flatten::forward(systolic_array, intermediate_buf_2, intermediate_buf_3);
        performance_timer.checkpoint();

        // 6. fc1: Linear
        Linear::forward(systolic_array, intermediate_buf_3, fc1_weights, ModelParameters::fc1_weight_scale, fc1_bias, outputs);
        performance_timer.checkpoint();

        // 7. Final Softmax
        Softmax::forward(outputs);
        performance_timer.checkpoint();

        performance_timer.finalize({"conv1 (Conv2d)", "relu1 (ReLU)", "conv2 (Conv2d)", "relu2 (ReLU)", "flatten (Flatten)", "fc1 (Linear)", "Final Softmax"});
    }
};