// TODO: Create driver

#include "systolic_array.h"
#include <cstdio>
#include <cstring>
#include "heepstor.h"
#include "heepstor_assert.h"
#include "systolic_array_def.h"

SystolicArray::SystolicArray(mmio_region_t base_addr) : base_addr(base_addr) {}

SystolicArray SystolicArray::get_default() {
    return SystolicArray{mmio_region_from_addr(SYSTOLIC_ARRAY_START_ADDRESS)};
}

void SystolicArray::write_weights(uint32_t four_packed_weights) {
    uint32_t cmd = static_cast<uint32_t>(Command::WRITE_WEIGHTS);
    write_32_bits(cmd << 18, four_packed_weights);
}

float SystolicArray::stream(uint32_t idx, float activation) {
    uint32_t cmd = static_cast<uint32_t>(Command::STREAM);

    uint32_t raw_activation_bits;
    std::memcpy(&raw_activation_bits, &activation, sizeof(raw_activation_bits));

    write_32_bits((cmd << 18) | (idx << 2), raw_activation_bits);
    uint32_t raw_res = read_32_bits(0);

    float res;
    std::memcpy(&res, &raw_res, sizeof(res));
    return res;
}

float SystolicArray::queue(uint32_t idx, float activation) {
    uint32_t cmd = static_cast<uint32_t>(Command::QUEUE);

    uint32_t raw_activation_bits;
    std::memcpy(&raw_activation_bits, &activation, sizeof(raw_activation_bits));

    write_32_bits((cmd << 18) | (idx << 2), raw_activation_bits);
    uint32_t raw_res = read_32_bits(0);

    float res;
    std::memcpy(&res, &raw_res, sizeof(res));
    return res;
}

template <typename T>
constexpr T ceil_div(T x, T y) {
    return (x + y - 1) / y;
}

// idx is the index (as element)
__attribute__((always_inline)) inline bool should_stream_systolic_array(size_t idx) {
    return idx % SYSTOLIC_ARRAY_SIZE == (SYSTOLIC_ARRAY_SIZE - 1);
}

void SystolicArray::matrix_matrix_multiply(const MatrixTile<float>& lhs, const PackedInt8Matrix& rhs, MatrixTile<float>& out) {
    const size_t M = lhs.num_rows();
    const size_t N = lhs.num_cols();
    const size_t P = rhs.num_cols();

    // Verify input matrix dimensions match for multiplication
    HEEPSTOR_ASSERT(lhs.num_cols() == rhs.num_rows() && "LHS columns must match RHS rows");

    // Verify output matrix dimensions
    HEEPSTOR_ASSERT(out.num_rows() == M && "Output matrix rows must match LHS rows");
    HEEPSTOR_ASSERT(out.num_cols() == P && "Output matrix cols must match RHS cols");

    ////////////////////////////////////////////////////
    /// 1. Set the RHS weights in the systolic array
    ////////////////////////////////////////////////////
    HEEPSTOR_ASSERT(N == SYSTOLIC_ARRAY_SIZE);
    HEEPSTOR_ASSERT(P == SYSTOLIC_ARRAY_SIZE);

    // Get raw packed weights and calculate array length
    const uint32_t* weights_ptr = rhs.get_packed_data();
    uint32_t weight_array_len = ceil_div((int)(N * P), WEIGHTS_PER_BUS);

    // Write weights in reverse order
    for (int i = weight_array_len - 1; i >= 0; --i) {
        write_weights(weights_ptr[i]);
    }

    //////////////////////////////////////////////////////////////
    /// 2. Stream the lhs values into the systolic array inputs
    //////////////////////////////////////////////////////////////
    HEEPSTOR_ASSERT(SYSTOLIC_ARRAY_SIZE >= 4);
    HEEPSTOR_ASSERT(SYSTOLIC_ARRAY_SIZE % 4 == 0);

    auto out_iterator = out.begin();

    uint32_t systolic_array_first_stream_with_valid_output = 2 * SYSTOLIC_ARRAY_SIZE;
    size_t num_streams_to_systolic_array = 0;

    auto stream_or_queue_and_store_result_if_valid = [&](size_t& idx, bool should_stream, float input_value) {
        // If the output is valid, store the result
        bool isOutputValid = num_streams_to_systolic_array >= systolic_array_first_stream_with_valid_output;
        float res;

        if (should_stream) {
            // printf("Streaming at idx=%d ", idx);
            // printFloat(input_value);
            // printf("\n");

            // Stream
            res = stream(idx, input_value);
            num_streams_to_systolic_array++;
            idx = 0;
        } else {
            // printf("Queuing at idx=%d ", idx);
            // printFloat(input_value);
            // printf("\n");

            // Queue
            res = queue(idx, input_value);
            idx++;
        }

        if (isOutputValid) {
            HEEPSTOR_ASSERT(out_iterator != out.end());
            *out_iterator = res;
            ++out_iterator;
        }
    };

    size_t idx = 0;

    // Iterate through the whole input matrix
    for (auto in_iterator = lhs.begin(); in_iterator != lhs.end(); ++in_iterator) {
        stream_or_queue_and_store_result_if_valid(idx, in_iterator.is_at_last_column(), *in_iterator);
    }

    //////////////////////////////////////////////////////////////
    /// 3. Stream out the remaining values
    //////////////////////////////////////////////////////////////

    // Stream remaining values
    while (out_iterator != out.end()) {
        bool should_stream = idx == SYSTOLIC_ARRAY_SIZE - 1;
        stream_or_queue_and_store_result_if_valid(idx, should_stream, 0.0f);
    }
}

void SystolicArray::matrix_matrix_multiply(const Matrix<float>& lhs, const PackedInt8Matrix& rhs, Matrix<float>& out) {
    // TODO: Actually implement tiling here
    auto lhs_tile = lhs.as_tile();
    auto rhs_tile = out.as_tile();

    matrix_matrix_multiply(lhs_tile, rhs, rhs_tile);
}
