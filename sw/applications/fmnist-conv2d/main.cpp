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

    const Matrix<float> inputs = {
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.809330463f},
        {-0.804296553f},
        {-0.750486076f},
        {-0.784681797f},
        {-0.807073891f},
        {-0.799262702f},
        {-0.686781406f},
        {-0.764571071f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.803949356f},
        {-0.695460498f},
        {-0.0485196933f},
        {-0.285459399f},
        {-0.587145209f},
        {-0.515282273f},
        {-0.0943454206f},
        {-0.641178906f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810024798f},
        {-0.809330463f},
        {-0.808636129f},
        {-0.807421088f},
        {-0.774093211f},
        {-0.206132337f},
        {0.555546343f},
        {0.811580122f},
        {0.772176921f},
        {0.736592591f},
        {0.58783263f},
        {-0.582260191f},
        {-0.810198367f},
        {-0.809677601f},
        {-0.807768226f},
        {-0.804296553f},
        {-0.802734375f},
        {-0.782945991f},
        {-0.393427521f},
        {0.312184095f},
        {0.596511662f},
        {0.863654733f},
        {0.961207926f},
        {0.947147787f},
        {0.769399583f},
        {-0.377333999f},
        {-0.807222664f},
        {-0.803255022f},
        {-0.802907884f},
        {-0.779127121f},
        {-0.693030417f},
        {-0.334756762f},
        {0.289791971f},
        {0.471358925f},
        {0.705347776f},
        {0.825466692f},
        {0.915035069f},
        {0.89455235f},
        {0.884484589f},
        {0.129129246f},
        {-0.731640041f},
        {-0.526912212f},
        {-0.363571405f},
        {-0.169159308f},
        {0.173318312f},
        {0.401058137f},
        {0.498264164f},
        {0.635741234f},
        {0.78762573f},
        {0.892643034f},
        {0.907918155f},
        {0.855669975f},
        {0.934649825f},
        {0.514580905f},
        {-0.00636402704f},
        {0.36634171f},
        {0.411993802f},
        {0.427269071f},
        {0.468408048f},
        {0.539576769f},
        {0.639907241f},
        {0.809149861f},
        {0.996271551f},
        {1.0523386f},
        {1.20057774f},
        {1.16308403f},
        {1.20769465f},
        {0.62170589f},
        {-0.287071228f},
        {0.135477453f},
        {0.457298815f},
        {0.699966729f},
        {0.718366444f},
        {0.748743236f},
        {0.849073768f},
        {0.683823526f},
        {0.413035393f},
        {0.406786382f},
        {1.1625632f},
        {1.26272011f},
        {1.09243608f},
        {0.32770735f},
        {-0.810198367f},
        {-0.792319357f},
        {-0.689558744f},
        {-0.57551527f},
        {-0.52847445f},
        {-0.518406689f},
        {-0.540451646f},
        {-0.655015886f},
        {-0.791798651f},
        {-0.779300749f},
        {-0.576556742f},
        {-0.556941986f},
        {-0.5869717f},
        {-0.682441831f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f},
        {-0.810198367f}
    };
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

    const Matrix<float> expected_outputs = {
        {1.00963211e-06f, 2.00501404e-11f, 1.46545972e-07f, 1.62643996e-08f, 3.09954942e-08f, 0.0104884617f, 1.90180015e-06f, 0.00980267208f, 0.000988187385f, 0.978717566f}
    };

    printf("Expected output: \n");
    expected_outputs.print();

    auto relative_error_percentage = outputs.relative_error(expected_outputs) * 100.0f;
    printf("Relative error: ");
    printFloat(relative_error_percentage);
    printf("%%\n");

    Matrix<int> output_prediction(1, 1);
    Argmax::forward_batch(outputs, output_prediction);

    printf("Computed Prediction: \n");
    output_prediction.print();

    printf("Expected Prediction: \n");
    printf("[9]\n");

    printf("True label value: \n");
    printf("[9]\n");
}

int main(int argc, char* argv[]) {
    // 1. Enable RISC-V CPU Floating Point Unit
    FloatingPointUnit::enable();

    // 2. Print banner
    printf("\n");
    printf("====================================\n");
    printf("Hello from HEEPstor! \n");
    printf("PROJECT: fmnist-conv2d\n");
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
        printf("Available bytes: %d, Used bytes: %d\n",
               StaticArenaAllocator::available_bytes(),
               StaticArenaAllocator::used_bytes());
        StaticArenaAllocator::reset();
    }

    // 5. Print assertion reminder
#if ENABLE_DEBUG_HEEPSTOR_ASSERTIONS
    printf("NOTE: Disable heepstor assert for better performance\n");
#endif
}