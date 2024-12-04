#pragma once

#include <drivers/systolic_array/systolic_array.h>
#include <math/matrix.h>
#include <math/packed_int8_matrix_tile.h>
#include <heepstorch/Layers.hpp>
#include "model_parameters.hpp"

class Model {
public:
    static constexpr size_t NUM_INPUT_FEATURES = $NUM_INPUT_FEATURES;
    static constexpr size_t NUM_OUTPUT_FEATURES = $NUM_OUTPUT_FEATURES;

    // inputs is a matrix of shape [BATCH_SIZE x NUM_INPUT_FEATURES], outputs is a matrix of shape [BATCH_SIZE x NUM_OUTPUT_FEATURES]
    static void infer(SystolicArray& systolic_array, const Matrix<float>& inputs, Matrix<float>& outputs) {
        HEEPSTOR_ASSERT(inputs.num_cols() == NUM_INPUT_FEATURES);
        HEEPSTOR_ASSERT(outputs.num_cols() == NUM_OUTPUT_FEATURES);
        HEEPSTOR_ASSERT(inputs.num_rows() == outputs.num_rows());

        const size_t BATCH_SIZE = inputs.num_rows();

        //////////////////////////////////////////////
        //  Wrap the model parameters into matrices.
        //////////////////////////////////////////////

$MODEL_PARAMETER_WRAPPERS

        //////////////////////////////////////////////
        //  Declare intermediate buffers
        //////////////////////////////////////////////

$INTERMEDIATE_BUFFER_DECLARATIONS

        //////////////////////////////////////////////
        //  Perform inference steps
        //////////////////////////////////////////////

$INFERENCE_STEPS
    }
};