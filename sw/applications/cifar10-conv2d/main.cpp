// DO NOT EDIT. File generated automatically, your changes will be lost when regenerated.
#include <stdio.h>
#include "drivers/fpu.hpp"
#include "gen/model.hpp"
#include "profiling/performance_timer.hpp"

void perform_test_inference() {
    auto systolic_array = SystolicArray::get_default();

    ///////////////////////////////////////////
    // 1. Define input and output buffers
    ///////////////////////////////////////////

    const Matrix<float> inputs(Model::INPUT_HEIGHT * Model::INPUT_WIDTH, Model::NUM_INPUT_CHANNELS);

    HEEPSTOR_ASSERT(inputs.num_rows() == Model::INPUT_HEIGHT * Model::INPUT_WIDTH);
    HEEPSTOR_ASSERT(inputs.num_cols() == Model::NUM_INPUT_CHANNELS);
    Matrix<float> outputs(1, Model::NUM_OUTPUT_FEATURES);

    ///////////////////////////////////////////
    // 2. Perform inference
    ///////////////////////////////////////////

    printf("Performing inference...\n");

    Model::infer(systolic_array, inputs, outputs, CheckpointPerformanceTimerDisplayConfig::Seconds);

    ///////////////////////////////////////////
    // 3. Check the output
    ///////////////////////////////////////////

    printf("Inference complete!\n");

    printf("Computed result:\n");
    outputs.print();

    Matrix<int> output_prediction(1, 1);
    Argmax::forward_batch(outputs, output_prediction);

    printf("Computed Prediction: \n");
    output_prediction.print();
}

int main(int argc, char* argv[]) {
    // 1. Enable RISC-V CPU Floating Point Unit
    FloatingPointUnit::enable();

    // 2. Print banner
    printf("\n");
    printf("====================================\n");
    printf("Hello from HEEPstor! \n");
    printf("PROJECT: cifar10-conv2d\n");
#if USE_SOFTWARE_DNN_LAYER_OPERATORS
    printf("USING SOFTWARE DNN OPERATORS\n");
#else
    printf("SYSTOLIC_ARRAY_SIZE=%u\n", SystolicArray::SIZE);
#endif
    printf("MODE: Convolutional (single image)\n");
    printf("====================================\n");
    printf("\n");

    // 3. Perform test inference
    perform_test_inference();

    // 4. Print arena allocator usage
    {
        printf("\nARENA ALLOCATOR: \n");
        printf("Available bytes: %d, Used bytes: %d\n", StaticArenaAllocator::available_bytes(), StaticArenaAllocator::used_bytes());
        StaticArenaAllocator::reset();
    }

    // 5. Print assertion reminder
#if ENABLE_DEBUG_HEEPSTOR_ASSERTIONS
    printf("NOTE: Disable heepstor assert for better performance\n");
#endif
}