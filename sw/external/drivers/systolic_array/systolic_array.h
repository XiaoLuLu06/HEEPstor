#pragma once

#include <cstdint>

#include "mmio.h"
#include "core_v_mini_mcu.h"
#include "systolic_array_def.h"

class SystolicArray {
public:
    constexpr static size_t SIZE = SYSTOLIC_ARRAY_SIZE;

    static SystolicArray get_default();

    SystolicArray(mmio_region_t base_addr);

    // TODO: In the future, maybe add a little piece of hardware to convert 2's complement uint8_t to sign + magnitude when
    //  loading weights, to avoid the software overhead.
     
    // four_packed_weights contains four 8-bit *sign+magnitude* packed weights. 
    //
    // Weight loading is done in a delay-line fashion.
    // Weights are input in the systolic array in reverse order. 
    // On cmd==CMD_WRITE_WEIGHTS, the weights are shifted to the right (and to the-bottom if it's the last row), and input_weight is loaded into the first weight. 
    // In order to load all the weights, load the last weight, then the second-to-last, and so on.
    // As weights are 8-bit, but weight_input is 32-bit, we load 4 weights at a time.
    // The assignment is the following:
    //      weights_reg[0][0] <= weight_input[31:24];
    //      weights_reg[0][1] <= weight_input[23:16];
    //      weights_reg[0][2] <= weight_input[15:8];
    //      weights_reg[0][3] <= weight_input[7:0];
    void write_weights(uint32_t four_packed_weights);

    // Load activation into the input shift registers at position IDX, but do not perform any computation yet (not all inputs are loaded)
    // Returns the activation at position IDX of output shift register.
    float stream(uint32_t idx, float activation);

    // Load activation into the input shift registers at position IDX, and perform a computation in the systolic array (shifting all the values). 
    // activation is therefore the last activation (IDX should be SYSTOLIC_ARRAY_SIZE-1 if loading activations starting from 0).
    // Returns the activation at position IDX of output shift register.
    float queue(uint32_t idx, float activation);

    // Multiplies two matrices lhs (size MxN) and rhs (size NxP) using the systolic array. The sizes are
    //  expressed as number of elements, but the weight matrix (rhs) must have already been compressed into 32-bit words.
    // IMPORTANT: The size of the RHS must be the size of the systolic array. If bigger matrices want to be used,
    //  then the big matrix multiplication has to be blocked
    void matrix_matrix_multiply(float* lhs, uint32_t* rhs, float* out, size_t M, size_t N, size_t P);

private:
    mmio_region_t base_addr;

    enum class Command {
        // Write weights into the systolic array        
        WRITE_WEIGHTS  = 0b00,

        // Load inputs into the FIFOs (shift registers), but do not perform any computation yet (not all inputs are loaded)
        QUEUE = 0b01,

        // Perform a computation in the systolic array (and shift all values)
        STREAM = 0b10,

        // Do nothing
        NONE = 0b11
    };
    
    __attribute__((always_inline)) uint32_t read_32_bits(ptrdiff_t offset) {
        return mmio_region_read32(base_addr, offset);
    }

    __attribute__((always_inline)) void write_32_bits(ptrdiff_t offset, uint32_t val) {
        mmio_region_write32(base_addr, offset, val);
    }
};