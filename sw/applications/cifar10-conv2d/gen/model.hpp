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
    static constexpr size_t NUM_INPUT_CHANNELS = 3;
    static constexpr size_t INPUT_HEIGHT = 26;
    static constexpr size_t INPUT_WIDTH = 26;

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

        //////////////////////////////////////////////
        //  Wrap the model parameters into matrices.
        //////////////////////////////////////////////

        CheckpointPerformanceTimer<13> performance_timer{display_config};
        performance_timer.reset();

        // conv1: Conv2d
        const PackedInt8Matrix conv1_kernel_weights = PackedInt8Matrix::from_const_pointer(ModelParameters::conv1_kernel_weight_data, 75, 16);
        const Matrix<float> conv1_bias = Matrix<float>::from_const_pointer(ModelParameters::conv1_bias_data, 1, 16);

        // batchnorm1: BatchNorm2d
        const Matrix<float> batchnorm1_scale = Matrix<float>::from_const_pointer(ModelParameters::batchnorm1_scale_data, 1, 16);
        const Matrix<float> batchnorm1_shift = Matrix<float>::from_const_pointer(ModelParameters::batchnorm1_shift_data, 1, 16);

        // conv2: Conv2d
        const PackedInt8Matrix conv2_kernel_weights = PackedInt8Matrix::from_const_pointer(ModelParameters::conv2_kernel_weight_data, 144, 24);
        const Matrix<float> conv2_bias = Matrix<float>::from_const_pointer(ModelParameters::conv2_bias_data, 1, 24);

        // batchnorm2: BatchNorm2d
        const Matrix<float> batchnorm2_scale = Matrix<float>::from_const_pointer(ModelParameters::batchnorm2_scale_data, 1, 24);
        const Matrix<float> batchnorm2_shift = Matrix<float>::from_const_pointer(ModelParameters::batchnorm2_shift_data, 1, 24);

        // conv3: Conv2d
        const PackedInt8Matrix conv3_kernel_weights = PackedInt8Matrix::from_const_pointer(ModelParameters::conv3_kernel_weight_data, 384, 6);
        const Matrix<float> conv3_bias = Matrix<float>::from_const_pointer(ModelParameters::conv3_bias_data, 1, 6);

        // batchnorm3: BatchNorm2d
        const Matrix<float> batchnorm3_scale = Matrix<float>::from_const_pointer(ModelParameters::batchnorm3_scale_data, 1, 6);
        const Matrix<float> batchnorm3_shift = Matrix<float>::from_const_pointer(ModelParameters::batchnorm3_shift_data, 1, 6);

        // fc1: Linear
        const PackedInt8Matrix fc1_weights = PackedInt8Matrix::from_const_pointer(ModelParameters::fc1_weight_data, 294, 10);
        const Matrix<float> fc1_bias = Matrix<float>::from_const_pointer(ModelParameters::fc1_bias_data, 1, 10);

        //////////////////////////////////////////////
        //  Declare intermediate buffers
        //////////////////////////////////////////////

        float* ping_buffer = StaticArenaAllocator::allocate_array<float>(7744);
        float* pong_buffer = StaticArenaAllocator::allocate_array<float>(9600);

        float* im2row_buffer = StaticArenaAllocator::allocate_array<float>(57600);

        Matrix<float> intermediate_buf_1(ping_buffer, 22*22, 16);
        Matrix<float> intermediate_buf_2(pong_buffer, 20*20, 24);
        Matrix<float> intermediate_buf_3(ping_buffer, 10*10, 24);
        Matrix<float> intermediate_buf_4(pong_buffer, 7*7, 6);
        Matrix<float> intermediate_buf_5(ping_buffer, 1, 294);

        //////////////////////////////////////////////
        //  Perform inference steps
        //////////////////////////////////////////////

        // 1. conv1: Conv2d
        Conv2d::forward(systolic_array, inputs, conv1_kernel_weights, ModelParameters::conv1_kernel_weight_scale, conv1_bias, intermediate_buf_1, 5, 3, 26, 26, im2row_buffer);
        performance_timer.checkpoint();

        // 2. batchnorm1: BatchNorm2d
        BatchNorm2d::forward(intermediate_buf_1, batchnorm1_scale, batchnorm1_shift);
        performance_timer.checkpoint();

        // 3. relu1: ReLU
        ReLU::forward(intermediate_buf_1);
        performance_timer.checkpoint();

        // 4. conv2: Conv2d
        Conv2d::forward(systolic_array, intermediate_buf_1, conv2_kernel_weights, ModelParameters::conv2_kernel_weight_scale, conv2_bias, intermediate_buf_2, 3, 16, 22, 22, im2row_buffer);
        performance_timer.checkpoint();

        // 5. batchnorm2: BatchNorm2d
        BatchNorm2d::forward(intermediate_buf_2, batchnorm2_scale, batchnorm2_shift);
        performance_timer.checkpoint();

        // 6. relu2: ReLU
        ReLU::forward(intermediate_buf_2);
        performance_timer.checkpoint();

        // 7. pool2: MaxPool2d
        MaxPool2d::forward(intermediate_buf_2, intermediate_buf_3, 2, 20, 20);
        performance_timer.checkpoint();

        // 8. conv3: Conv2d
        Conv2d::forward(systolic_array, intermediate_buf_3, conv3_kernel_weights, ModelParameters::conv3_kernel_weight_scale, conv3_bias, intermediate_buf_4, 4, 24, 10, 10, im2row_buffer);
        performance_timer.checkpoint();

        // 9. batchnorm3: BatchNorm2d
        BatchNorm2d::forward(intermediate_buf_4, batchnorm3_scale, batchnorm3_shift);
        performance_timer.checkpoint();

        // 10. relu3: ReLU
        ReLU::forward(intermediate_buf_4);
        performance_timer.checkpoint();

        // 11. flatten: Flatten
        Flatten::forward(intermediate_buf_4, intermediate_buf_5);
        performance_timer.checkpoint();

        // 12. fc1: Linear
        Linear::forward(systolic_array, intermediate_buf_5, fc1_weights, ModelParameters::fc1_weight_scale, fc1_bias, outputs);
        performance_timer.checkpoint();

        // 13. Final Softmax
        Softmax::forward(outputs);
        performance_timer.checkpoint();

        performance_timer.finalize({"conv1 (Conv2d)", "batchnorm1 (BatchNorm2d)", "relu1 (ReLU)", "conv2 (Conv2d)", "batchnorm2 (BatchNorm2d)", "relu2 (ReLU)", "pool2 (MaxPool2d)", "conv3 (Conv2d)", "batchnorm3 (BatchNorm2d)", "relu3 (ReLU)", "flatten (Flatten)", "fc1 (Linear)", "Final Softmax"});
    }
};