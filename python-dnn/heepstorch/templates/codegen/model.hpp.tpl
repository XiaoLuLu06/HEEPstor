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
$INPUT_INFO

    // Output configuration
$OUTPUT_INFO

    static void infer(SystolicArray& systolic_array, const Matrix<float>& inputs, Matrix<float>& outputs,
                     CheckpointPerformanceTimerDisplayConfig display_config) {

        //////////////////////////////////////////////
        // Validate input and output sizes
        //////////////////////////////////////////////

$VALIDATION_CODE

        //////////////////////////////////////////////
        //  Wrap the model parameters into matrices.
        //////////////////////////////////////////////

        CheckpointPerformanceTimer<$NUM_LAYERS> performance_timer{display_config};
        performance_timer.reset();

$MODEL_PARAMETER_WRAPPERS

        //////////////////////////////////////////////
        //  Declare intermediate buffers
        //////////////////////////////////////////////

$INTERMEDIATE_BUFFER_DECLARATIONS

        //////////////////////////////////////////////
        //  Perform inference steps
        //////////////////////////////////////////////

$INFERENCE_STEPS

        performance_timer.finalize({$PERF_TIMER_LAYER_LIST});
    }
};