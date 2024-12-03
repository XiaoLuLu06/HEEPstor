#include <stdio.h>
#include "drivers/fpu.hpp"
#include "gen/model.hpp"

void perform_test_inference() {
    auto systolic_array = SystolicArray::get_default();

    ///////////////////////////////////////////
    // 1. Define input and output buffers
    ///////////////////////////////////////////

    const Matrix<float> inputs = $INPUT_MATRIX;

    HEEPSTOR_ASSERT(inputs.num_cols() == Model::NUM_INPUT_FEATURES);

    size_t batch_size = inputs.num_rows();
    Matrix<float> outputs(batch_size, Model::NUM_OUTPUT_FEATURES);

    ///////////////////////////////////////////
    // 2. Perform inference
    ///////////////////////////////////////////

    printf("Performing inference...\n");

    Model::infer(systolic_array, inputs, outputs);

    ///////////////////////////////////////////
    // 3. Check the output
    ///////////////////////////////////////////

    printf("Inference complete!\n");

    printf("Computed result:\n");
    outputs.print();

    const Matrix<float> expected_outputs = $EXPECTED_OUTPUT_MATRIX;

    HEEPSTOR_ASSERT(outputs.num_rows() == batch_size);
    HEEPSTOR_ASSERT(outputs.num_cols() == Model::NUM_OUTPUT_FEATURES);

    printf("Expected output: \n");
    expected_outputs.print();

    auto relative_error_percentage = outputs.relative_error(expected_outputs) * 100.0f;
    printf("Relative error: ");
    printFloat(relative_error_percentage);
    printf("%%\n");

    Matrix<int> output_predictions(1, outputs.num_cols());
    Argmax::forward_batch(outputs, output_predictions);

    printf("Computed Predictions: \n");
    output_predictions.print();

    printf("Expected Predictions: \n");
    printf("$EXPECTED_PREDICTIONS\n");

    printf("True label values: \n");
    printf("$TRUE_LABEL_VALUES\n");
}

int main(int argc, char* argv[]) {
    // 1. Enable RISC-V CPU Floating Point Unit
    FloatingPointUnit::enable();

    // 2. Print the systolic array size used by the system.
    printf("SYSTOLIC_ARRAY_SIZE=%u\n", SystolicArray::SIZE);

    // 3. Perform a test inference
    perform_test_inference();

    // 4. Print arena allocator (memory where non-static matrices are stored) usage information
    {
        printf("\n ARENA ALLOCATOR: \n");
        printf("Available bytes: %d, Used bytes: %d\n", StaticArenaAllocator::available_bytes(), StaticArenaAllocator::used_bytes());

        // NOTE: You may clear the arena allocator after inference to recover the used memory for other inferences.
        StaticArenaAllocator::reset();
    }

    // 5. Print a reminder to disable Heepstor asserts when functionality has been established to be
    //  correct in order to avoid expensive checks during inference.
    // TODO: Disable heepstor assert for better performance
    printf("NOTE: Disable heepstor assert for better performance\n");
}