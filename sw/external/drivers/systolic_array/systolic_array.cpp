// TODO: Create driver

#include "systolic_array.h"
#include <algorithm>
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

void SystolicArray::matrix_matrix_multiply(const MatrixTile<float>& lhs, const PackedInt8MatrixTile& rhs, MatrixTile<float>& out) {
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
    HEEPSTOR_ASSERT(N <= SYSTOLIC_ARRAY_SIZE);
    HEEPSTOR_ASSERT(P <= SYSTOLIC_ARRAY_SIZE);

    // printf("Multiplying...\n");

    // First write as many rows of 0s as the difference between rhs.num_blocks_rows and SYSTOLIC_ARRAY_SIZE (bottom padding)
    for (int i = 0; i < SYSTOLIC_ARRAY_SIZE - rhs.num_blocks_rows(); ++i) {
        for (int j = 0; j < SYSTOLIC_ARRAY_SIZE; j += 4) {
            // printf("Writing bottom padding 0x0 \n");
            write_weights(0);
        }
    }

    // printf("Bottom padding done!\n");
    // printf("Stride: %d\n", rhs.get_stride());

    const uint32_t* last_col_ptr = rhs.last_packed_pointer();

    // Now, write the actual weights of the tile
    for (int i = 0; i < rhs.num_blocks_rows(); ++i) {
        const uint32_t* ptr = last_col_ptr;

        // First, write as many 0s as the difference between rhs.num_block_cols and SYSTOLIC_ARRAY_SIZE/4 (right padding)
        for (int j = 0; j < SYSTOLIC_ARRAY_SIZE / 4 - rhs.num_blocks_cols(); ++j) {
            // printf("Writing right padding 0x0 \n");
            write_weights(0);
        }

        // Now actually write the weights
        for (int j = 0; j < rhs.num_blocks_cols(); ++j) {
            // printf("Writing weight %#08X (ptr=%#08X) \n", *ptr, (uint32_t)ptr);
            write_weights(*ptr);
            ptr--;
        }

        // Move to the previous column
        last_col_ptr = rhs.move_pointer_one_row_up(last_col_ptr);
    }

    //////////////////////////////////////////////////////////////
    /// 2. Stream the lhs values into the systolic array inputs
    //////////////////////////////////////////////////////////////
    HEEPSTOR_ASSERT(SYSTOLIC_ARRAY_SIZE >= 4);
    HEEPSTOR_ASSERT(SYSTOLIC_ARRAY_SIZE % 4 == 0);

    auto out_iterator = out.begin();

    uint32_t systolic_array_first_stream_with_valid_output = 2 * SYSTOLIC_ARRAY_SIZE;
    size_t num_streams_to_systolic_array = 0;

    auto stream_or_queue_and_store_result_if_valid = [&](size_t& idx, float input_value) {
        // If the output is valid, store the result
        bool isOutputValid = num_streams_to_systolic_array >= systolic_array_first_stream_with_valid_output;
        float res;

        int prev_idx = idx;

        if (idx == SYSTOLIC_ARRAY_SIZE - 1) {
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

        // TODO: If Matrix tile size is systolic array, don't need to check prev_idx < out.num_cols(). Maybe fast path?
        if (isOutputValid && prev_idx < out.num_cols()) {
            HEEPSTOR_ASSERT(out_iterator != out.end());
            // printf("Saving output at idx=%d: ", idx);
            // printFloat(res);
            // printf("\n");

            *out_iterator += res;
            ++out_iterator;
        }
    };

    size_t idx = 0;

    // Iterate through the whole input matrix

    if (N == SYSTOLIC_ARRAY_SIZE) {
        // Fast path
        for (auto in_iterator = lhs.begin(); in_iterator != lhs.end(); ++in_iterator) {
            stream_or_queue_and_store_result_if_valid(idx, *in_iterator);
        }
    } else {
        // Need to do more special case handling
        for (auto in_iterator = lhs.begin(); in_iterator != lhs.end(); ++in_iterator) {
            bool is_last_column = in_iterator.is_at_last_column();
            if (is_last_column) {
                stream_or_queue_and_store_result_if_valid(idx, *in_iterator);
                for (int i = 0; i < SYSTOLIC_ARRAY_SIZE - N; ++i) {
                    stream_or_queue_and_store_result_if_valid(idx, 0);
                }
            } else {
                stream_or_queue_and_store_result_if_valid(idx, *in_iterator);
            }
        }
    }

    //////////////////////////////////////////////////////////////
    /// 3. Stream out the remaining values
    //////////////////////////////////////////////////////////////

    // Stream remaining values
    while (out_iterator != out.end()) {
        bool should_stream = idx == SYSTOLIC_ARRAY_SIZE - 1;
        stream_or_queue_and_store_result_if_valid(idx, 0.0f);
    }
}

void SystolicArray::matrix_matrix_multiply(const Matrix<float>& lhs, const PackedInt8Matrix& rhs, Matrix<float>& out) {
    // Verify input matrix dimensions match for multiplication
    HEEPSTOR_ASSERT(lhs.num_cols() == rhs.num_rows() && "LHS columns must match RHS rows");
    HEEPSTOR_ASSERT(out.num_rows() == lhs.num_rows() && "Output matrix rows must match LHS rows");
    HEEPSTOR_ASSERT(out.num_cols() == rhs.num_cols() && "Output matrix cols must match RHS cols");

    const size_t M = lhs.num_rows();  // Number of rows in LHS
    const size_t N = lhs.num_cols();  // Number of columns in LHS / rows in RHS
    const size_t P = rhs.num_cols();  // Number of columns in RHS

    // Calculate number of tiles needed for RHS in both dimensions
    const size_t N_tiles = (N + SYSTOLIC_ARRAY_SIZE - 1) / SYSTOLIC_ARRAY_SIZE;  // RHS rows
    const size_t P_tiles = (P + SYSTOLIC_ARRAY_SIZE - 1) / SYSTOLIC_ARRAY_SIZE;  // RHS cols

    // Initialize output matrix to zeros since we'll be accumulating results
    out.fill(0.0f);

    // Iterate over RHS 2D tiles
    for (size_t n = 0; n < N_tiles; n++) {
        const size_t n_start = n * SYSTOLIC_ARRAY_SIZE;
        const size_t n_size = std::min(static_cast<size_t>(SYSTOLIC_ARRAY_SIZE), N - n_start);

        for (size_t p = 0; p < P_tiles; p++) {
            const size_t p_start = p * SYSTOLIC_ARRAY_SIZE;
            const size_t p_size = std::min(static_cast<size_t>(SYSTOLIC_ARRAY_SIZE), P - p_start);

            // Get current RHS tile
            auto rhs_tile = rhs.get_tile(n_start, p_start, n_size, p_size);

            // Process all rows of LHS against this RHS tile
            auto lhs_tile = lhs.get_tile(0, n_start, M, n_size);
            auto out_tile = out.get_tile(0, p_start, M, p_size);

            // printf("Multiplying LHS(%d:%d, %d:%d) * RHS(%d:%d, %d:%d) -> OUT(%d:%d, %d:%d)\n", 0, M - 1, n_start, n_start + n_size - 1, n_start,
            //        n_start + n_size - 1, p_start, p_start + p_size - 1, 0, M - 1, p_start, p_start + p_size - 1);

            // Perform multiplication
            matrix_matrix_multiply(lhs_tile, rhs_tile, out_tile);
        }
    }
}
