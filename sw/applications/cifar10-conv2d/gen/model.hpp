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
    static constexpr size_t INPUT_HEIGHT = 32;
    static constexpr size_t INPUT_WIDTH = 32;

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

        CheckpointPerformanceTimer<19> performance_timer{display_config};
        performance_timer.reset();

        // conv1: Conv2d
        const PackedInt8Matrix conv1_kernel_weights = PackedInt8Matrix::from_const_pointer(ModelParameters::conv1_kernel_weight_data, 27, 16);
        const Matrix<float> conv1_bias = Matrix<float>::from_const_pointer(ModelParameters::conv1_bias_data, 1, 16);

        // batchnorm1: BatchNorm2d
        const Matrix<float> batchnorm1_scale = Matrix<float>::from_const_pointer(ModelParameters::batchnorm1_scale_data, 1, 16);
        const Matrix<float> batchnorm1_shift = Matrix<float>::from_const_pointer(ModelParameters::batchnorm1_shift_data, 1, 16);

        // conv2: Conv2d
        const PackedInt8Matrix conv2_kernel_weights = PackedInt8Matrix::from_const_pointer(ModelParameters::conv2_kernel_weight_data, 144, 32);
        const Matrix<float> conv2_bias = Matrix<float>::from_const_pointer(ModelParameters::conv2_bias_data, 1, 32);

        // batchnorm2: BatchNorm2d
        const Matrix<float> batchnorm2_scale = Matrix<float>::from_const_pointer(ModelParameters::batchnorm2_scale_data, 1, 32);
        const Matrix<float> batchnorm2_shift = Matrix<float>::from_const_pointer(ModelParameters::batchnorm2_shift_data, 1, 32);

        // conv3: Conv2d
        const PackedInt8Matrix conv3_kernel_weights = PackedInt8Matrix::from_const_pointer(ModelParameters::conv3_kernel_weight_data, 288, 64);
        const Matrix<float> conv3_bias = Matrix<float>::from_const_pointer(ModelParameters::conv3_bias_data, 1, 64);

        // batchnorm3: BatchNorm2d
        const Matrix<float> batchnorm3_scale = Matrix<float>::from_const_pointer(ModelParameters::batchnorm3_scale_data, 1, 64);
        const Matrix<float> batchnorm3_shift = Matrix<float>::from_const_pointer(ModelParameters::batchnorm3_shift_data, 1, 64);

        // conv4: Conv2d
        const PackedInt8Matrix conv4_kernel_weights = PackedInt8Matrix::from_const_pointer(ModelParameters::conv4_kernel_weight_data, 576, 32);
        const Matrix<float> conv4_bias = Matrix<float>::from_const_pointer(ModelParameters::conv4_bias_data, 1, 32);

        // batchnorm4: BatchNorm2d
        const Matrix<float> batchnorm4_scale = Matrix<float>::from_const_pointer(ModelParameters::batchnorm4_scale_data, 1, 32);
        const Matrix<float> batchnorm4_shift = Matrix<float>::from_const_pointer(ModelParameters::batchnorm4_shift_data, 1, 32);

        // conv5: Conv2d
        const PackedInt8Matrix conv5_kernel_weights = PackedInt8Matrix::from_const_pointer(ModelParameters::conv5_kernel_weight_data, 288, 16);
        const Matrix<float> conv5_bias = Matrix<float>::from_const_pointer(ModelParameters::conv5_bias_data, 1, 16);

        // batchnorm5: BatchNorm2d
        const Matrix<float> batchnorm5_scale = Matrix<float>::from_const_pointer(ModelParameters::batchnorm5_scale_data, 1, 16);
        const Matrix<float> batchnorm5_shift = Matrix<float>::from_const_pointer(ModelParameters::batchnorm5_shift_data, 1, 16);

        // fc1: Linear
        const PackedInt8Matrix fc1_weights = PackedInt8Matrix::from_const_pointer(ModelParameters::fc1_weight_data, 1024, 10);
        const Matrix<float> fc1_bias = Matrix<float>::from_const_pointer(ModelParameters::fc1_bias_data, 1, 10);

        //////////////////////////////////////////////
        //  Declare intermediate buffers
        //////////////////////////////////////////////

        float* ping_buffer = StaticArenaAllocator::allocate_array<float>(14400);
        float* pong_buffer = StaticArenaAllocator::allocate_array<float>(25088);

        float* im2row_buffer = StaticArenaAllocator::allocate_array<float>(112896);

        Matrix<float> intermediate_buf_1(ping_buffer, 30*30, 16);
        Matrix<float> intermediate_buf_2(pong_buffer, 28*28, 32);
        Matrix<float> intermediate_buf_3(ping_buffer, 14*14, 32);
        Matrix<float> intermediate_buf_4(pong_buffer, 12*12, 64);
        Matrix<float> intermediate_buf_5(ping_buffer, 10*10, 32);
        Matrix<float> intermediate_buf_6(pong_buffer, 8*8, 16);
        Matrix<float> intermediate_buf_7(ping_buffer, 1, 1024);

        //////////////////////////////////////////////
        //  Perform inference steps
        //////////////////////////////////////////////

        // 1. conv1: Conv2d
        Conv2d::forward(systolic_array, inputs, conv1_kernel_weights, ModelParameters::conv1_kernel_weight_scale, conv1_bias, intermediate_buf_1, 3, 3, 32, 32, im2row_buffer);
        performance_timer.checkpoint();

        // 2. batchnorm1: BatchNorm2d
        BatchNorm2d::forward(intermediate_buf_1, batchnorm1_scale, batchnorm1_shift);
        performance_timer.checkpoint();

        // 3. relu1: ReLU
        ReLU::forward(intermediate_buf_1);
        performance_timer.checkpoint();

        // 4. conv2: Conv2d
        Conv2d::forward(systolic_array, intermediate_buf_1, conv2_kernel_weights, ModelParameters::conv2_kernel_weight_scale, conv2_bias, intermediate_buf_2, 3, 16, 30, 30, im2row_buffer);
        performance_timer.checkpoint();

        // 5. batchnorm2: BatchNorm2d
        BatchNorm2d::forward(intermediate_buf_2, batchnorm2_scale, batchnorm2_shift);
        performance_timer.checkpoint();

        // 6. relu2: ReLU
        ReLU::forward(intermediate_buf_2);
        performance_timer.checkpoint();

        // 7. pool2: MaxPool2d
        MaxPool2d::forward(intermediate_buf_2, intermediate_buf_3, 2, 28, 28);
        performance_timer.checkpoint();

        // 8. conv3: Conv2d
        Conv2d::forward(systolic_array, intermediate_buf_3, conv3_kernel_weights, ModelParameters::conv3_kernel_weight_scale, conv3_bias, intermediate_buf_4, 3, 32, 14, 14, im2row_buffer);
        performance_timer.checkpoint();

        // 9. batchnorm3: BatchNorm2d
        BatchNorm2d::forward(intermediate_buf_4, batchnorm3_scale, batchnorm3_shift);
        performance_timer.checkpoint();

        // 10. relu3: ReLU
        ReLU::forward(intermediate_buf_4);
        performance_timer.checkpoint();

        // 11. conv4: Conv2d
        Conv2d::forward(systolic_array, intermediate_buf_4, conv4_kernel_weights, ModelParameters::conv4_kernel_weight_scale, conv4_bias, intermediate_buf_5, 3, 64, 12, 12, im2row_buffer);
        performance_timer.checkpoint();

        // 12. batchnorm4: BatchNorm2d
        BatchNorm2d::forward(intermediate_buf_5, batchnorm4_scale, batchnorm4_shift);
        performance_timer.checkpoint();

        // 13. relu4: ReLU
        ReLU::forward(intermediate_buf_5);
        performance_timer.checkpoint();

        // 14. conv5: Conv2d
        Conv2d::forward(systolic_array, intermediate_buf_5, conv5_kernel_weights, ModelParameters::conv5_kernel_weight_scale, conv5_bias, intermediate_buf_6, 3, 32, 10, 10, im2row_buffer);
        performance_timer.checkpoint();

        // 15. batchnorm5: BatchNorm2d
        BatchNorm2d::forward(intermediate_buf_6, batchnorm5_scale, batchnorm5_shift);
        performance_timer.checkpoint();

        // 16. relu5: ReLU
        ReLU::forward(intermediate_buf_6);
        performance_timer.checkpoint();

        // 17. flatten: Flatten
        Flatten::forward(intermediate_buf_6, intermediate_buf_7);
        performance_timer.checkpoint();

        // 18. fc1: Linear
        Linear::forward(systolic_array, intermediate_buf_7, fc1_weights, ModelParameters::fc1_weight_scale, fc1_bias, outputs);
        performance_timer.checkpoint();

        // 19. Final Softmax
        Softmax::forward(outputs);
        performance_timer.checkpoint();

        performance_timer.finalize({"conv1 (Conv2d)", "batchnorm1 (BatchNorm2d)", "relu1 (ReLU)", "conv2 (Conv2d)", "batchnorm2 (BatchNorm2d)", "relu2 (ReLU)", "pool2 (MaxPool2d)", "conv3 (Conv2d)", "batchnorm3 (BatchNorm2d)", "relu3 (ReLU)", "conv4 (Conv2d)", "batchnorm4 (BatchNorm2d)", "relu4 (ReLU)", "conv5 (Conv2d)", "batchnorm5 (BatchNorm2d)", "relu5 (ReLU)", "flatten (Flatten)", "fc1 (Linear)", "Final Softmax"});
    }
};