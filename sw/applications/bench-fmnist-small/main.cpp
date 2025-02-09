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

    // Because linear layers support batch_size>1, let's showcase it with batch_size = 9
    const size_t batch_size = 8;

    Matrix<float> inputs(batch_size, Model::NUM_INPUT_FEATURES);
    // Normally, the input would come from sensors, simulate it as filling it with random data

    inputs.fill_random(-2, 2);

    Matrix<float> outputs(batch_size, Model::NUM_OUTPUT_FEATURES);

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

    Matrix<int> output_prediction(1, batch_size);
    Argmax::forward_batch(outputs, output_prediction);

    printf("Computed Prediction: \n");
    output_prediction.print();
}

int main(int argc, char* argv[]) {
    // 1. Enable RISC-V CPU Floating Point Unit
    FloatingPointUnit::enable();

    // 2. Print banner with project name  systolic array size used by the system.
    printf("\n");
    printf("====================================\n");
    printf("Hello from HEEPstor! \n");
    printf("PROJECT: bench-fmnist-small\n");
#if USE_SOFTWARE_DNN_LAYER_OPERATORS
    printf("USING SOFTWARE DNN OPERATORS\n");
#else
    printf("SYSTOLIC_ARRAY_SIZE=%u\n", SystolicArray::SIZE);
#endif
    printf("====================================\n");
    printf("\n");

    // 3. Perform a test inference
    perform_test_inference();

    // 4. Print arena allocator (memory where non-static matrices are stored) usage information
    {
        printf("\nARENA ALLOCATOR: \n");
        printf("Available bytes: %d, Used bytes: %d\n", StaticArenaAllocator::available_bytes(), StaticArenaAllocator::used_bytes());

        // NOTE: You may clear the arena allocator after inference to recover the used memory for other inferences.
        StaticArenaAllocator::reset();
    }

    // 5. Print a reminder to disable Heepstor asserts when functionality has been established to be
    //  correct in order to avoid expensive checks during inference.
#if ENABLE_DEBUG_HEEPSTOR_ASSERTIONS
    printf("NOTE: Disable heepstor assert for better performance\n");
#endif
}