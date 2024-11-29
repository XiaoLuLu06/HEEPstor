#pragma once

#include <cstddef>
#include <cstdint>
#include "heepstor_assert.h"
#include "math/random_number_generator.h"
#include "memory/static_arena_allocator.h"
#include "packed_int8_matrix_tile.h"

class PackedInt8Matrix {
private:
    uint32_t* packed_data;
    size_t rows;
    size_t cols;

    // Number of int8_t values that can fit in one uint32_t
    static constexpr size_t INT8_PER_UINT32 = 4;

    // Helper to calculate the padded number of columns (smallest multiple of 4 >= r)
    static constexpr size_t num_cols_after_padding(size_t c) { return (c + INT8_PER_UINT32 - 1) & ~(INT8_PER_UINT32 - 1); }

    // Helper to set all padding values in a row to zero, as expected by the systolic array
    void clear_row_padding(size_t row) {
        const size_t num_cols_padded = num_cols_after_padding(cols);

        // Clear padding values in this row
        for (size_t c = cols; c < num_cols_padded; ++c) {
            const size_t linear_idx = row * num_cols_padded + c;
            const size_t uint32_idx = linear_idx / INT8_PER_UINT32;
            const size_t byte_pos = linear_idx % INT8_PER_UINT32;
            const uint32_t shift = (3 - byte_pos) * 8;

            packed_data[uint32_idx] &= ~(0xFFu << shift);
        }
    }

public:
    PackedInt8Matrix(uint32_t* data, size_t r, size_t c) : packed_data(data), rows(r), cols(c) {
        HEEPSTOR_ASSERT(data != nullptr && "Packed data pointer cannot be null");
        HEEPSTOR_ASSERT(r > 0 && c > 0 && "Matrix dimensions must be positive");
    }

    // Get a specific uint32_t containing 4 packed values. The index is in uint32_t units (i.e., goes over 4 int8_t).
    uint32_t get_packed_uint32(size_t packed_idx) const {
        const size_t required_uint32 = required_storage(rows, cols);
        HEEPSTOR_ASSERT(packed_idx < required_uint32 && "Packed index out of bounds");
        return packed_data[packed_idx];
    }

    // Extract an int8_t value at given row and column
    int8_t operator()(size_t r, size_t c) const {
        HEEPSTOR_ASSERT(r < rows && c < cols && "Matrix indices out of bounds");

        // Calculate linear index in row-major order with padded columns
        const size_t num_cols_padded = num_cols_after_padding(cols);
        const size_t linear_idx = r * num_cols_padded + c;

        // Calculate which uint32_t contains our value and the byte position within it
        const size_t uint32_idx = linear_idx / INT8_PER_UINT32;
        const size_t byte_pos = linear_idx % INT8_PER_UINT32;
        const uint32_t shift = (3 - byte_pos) * 8;

        return static_cast<int8_t>((packed_data[uint32_idx] >> shift) & 0xff);
    }

    // Helper to calculate required uint32_t storage for given dimensions
    static constexpr size_t required_storage(size_t r, size_t c) {
        const size_t num_cols_padded = num_cols_after_padding(c);
        // Total number of int8_t elements needed with padding
        const size_t total_elements = r * num_cols_padded;

        HEEPSTOR_ASSERT(num_cols_padded % INT8_PER_UINT32 == 0);
        return total_elements / INT8_PER_UINT32;
    }

    static PackedInt8Matrix allocate(size_t rows, size_t cols) {
        HEEPSTOR_ASSERT(rows > 0 && cols > 0 && "Matrix dimensions must be positive");

        const size_t required_uint32 = required_storage(rows, cols);
        uint32_t* packed_storage = StaticArenaAllocator::allocate_array<uint32_t>(required_uint32);

        // Initialize all storage to 0 to ensure padding is zero
        for (size_t i = 0; i < required_uint32; ++i) {
            packed_storage[i] = 0;
        }

        return PackedInt8Matrix(packed_storage, rows, cols);
    }

    static PackedInt8Matrix allocate_from_int8_list(std::initializer_list<std::initializer_list<int8_t>> init) {
        const size_t r = init.size();
        HEEPSTOR_ASSERT(r > 0 && "Empty initializer list");
        const size_t c = init.begin()->size();
        HEEPSTOR_ASSERT(c > 0 && "Empty row in initializer list");

        const size_t num_uint32 = required_storage(r, c);
        uint32_t* packed_storage = StaticArenaAllocator::allocate_array<uint32_t>(num_uint32);

        // Initialize all storage to 0 to ensure padding is zero
        for (size_t i = 0; i < num_uint32; ++i) {
            packed_storage[i] = 0;
        }

        // Pack values row by row
        const size_t num_cols_padded = num_cols_after_padding(c);
        size_t row_idx = 0;
        for (const auto& row : init) {
            HEEPSTOR_ASSERT(row.size() == c && "All rows must have the same length");
            size_t col_idx = 0;
            for (const auto& val : row) {
                const size_t linear_idx = row_idx * num_cols_padded + col_idx;
                const size_t uint32_idx = linear_idx / INT8_PER_UINT32;
                const size_t byte_pos = linear_idx % INT8_PER_UINT32;
                const uint32_t shift = (3 - byte_pos) * 8;

                packed_storage[uint32_idx] |= (static_cast<uint32_t>(val) & 0xFF) << shift;
                col_idx++;
            }
            row_idx++;
        }

        return PackedInt8Matrix(packed_storage, r, c);
    }

    void fill(int8_t value) {
        const uint32_t packed_pattern = static_cast<uint32_t>(value) * 0x01010101u;
        const size_t num_cols_padded = num_cols_after_padding(cols);

        for (size_t r = 0; r < rows; ++r) {
            // Fill the actual data for this row
            const size_t row_start = r * num_cols_padded / INT8_PER_UINT32;
            const size_t row_end = (r * num_cols_padded + cols - 1) / INT8_PER_UINT32 + 1;

            for (size_t i = row_start; i < row_end; ++i) {
                packed_data[i] = packed_pattern;
            }

            // Clear the padding area
            clear_row_padding(r);
        }
    }

    void fill_random(int8_t min_val, int8_t max_val, RandomNumberGenerator& rng) {
        const size_t num_cols_padded = num_cols_after_padding(cols);

        for (size_t r = 0; r < rows; ++r) {
            // Fill actual data
            for (size_t c = 0; c < cols; ++c) {
                const int8_t rand_val = static_cast<int8_t>(rng.rand_int_range(min_val, max_val));

                const size_t linear_idx = r * num_cols_padded + c;
                const size_t uint32_idx = linear_idx / INT8_PER_UINT32;
                const size_t byte_pos = linear_idx % INT8_PER_UINT32;
                const uint32_t shift = (3 - byte_pos) * 8;

                packed_data[uint32_idx] &= ~(0xFFu << shift);
                packed_data[uint32_idx] |= (static_cast<uint32_t>(rand_val) & 0xFF) << shift;
            }

            // Clear the padding area
            clear_row_padding(r);
        }
    }

    void fill_diag(int8_t value) {
        HEEPSTOR_ASSERT(rows == cols && "Matrix must be square for diagonal fill");

        // First zero out the entire matrix
        const size_t required_uint32 = required_storage(rows, cols);
        for (size_t i = 0; i < required_uint32; ++i) {
            packed_data[i] = 0;
        }

        // Then set diagonal elements
        const size_t num_cols_padded = num_cols_after_padding(cols);
        for (size_t i = 0; i < rows; i++) {
            const size_t linear_idx = i * num_cols_padded + i;
            const size_t uint32_idx = linear_idx / INT8_PER_UINT32;
            const size_t byte_pos = linear_idx % INT8_PER_UINT32;
            const uint32_t shift = (3 - byte_pos) * 8;

            packed_data[uint32_idx] |= (static_cast<uint32_t>(value) & 0xFF) << shift;
        }
    }

    PackedInt8MatrixTile get_tile(size_t start_row, size_t start_col, size_t tile_rows, size_t tile_cols) {
        HEEPSTOR_ASSERT(start_row + tile_rows <= rows && start_col + tile_cols <= cols);
        return PackedInt8MatrixTile(packed_data, rows, cols, start_row, start_col, tile_rows, tile_cols);
    }

    const PackedInt8MatrixTile get_tile(size_t start_row, size_t start_col, size_t tile_rows, size_t tile_cols) const {
        HEEPSTOR_ASSERT(start_row + tile_rows <= rows && start_col + tile_cols <= cols);
        return PackedInt8MatrixTile(packed_data, rows, cols, start_row, start_col, tile_rows, tile_cols);
    }

    PackedInt8MatrixTile as_tile() { return PackedInt8MatrixTile(packed_data, rows, cols, 0, 0, rows, cols); }

    const PackedInt8MatrixTile as_tile() const { return PackedInt8MatrixTile(packed_data, rows, cols, 0, 0, rows, cols); }

    // Getters
    size_t num_rows() const { return rows; }
    size_t num_cols() const { return cols; }
    const uint32_t* get_packed_data() const { return packed_data; }
};