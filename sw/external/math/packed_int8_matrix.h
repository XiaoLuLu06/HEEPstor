#pragma once

#include <cstddef>
#include <cstdint>
#include "heepstor_assert.h"
#include "math/random_number_generator.h"
#include "memory/static_arena_allocator.h"

class PackedInt8Matrix {
private:
    uint32_t* packed_data;
    size_t rows;
    size_t cols;

    static constexpr size_t INT8_PER_UINT32 = sizeof(int32_t) / sizeof(int8_t);

public:
    PackedInt8Matrix(uint32_t* data, size_t r, size_t c) : packed_data(data), rows(r), cols(c) {
        HEEPSTOR_ASSERT(data != nullptr && "Packed data pointer cannot be null");
        HEEPSTOR_ASSERT(r > 0 && c > 0 && "Matrix dimensions must be positive");
    }

    // Get a specific uint32_t containing 4 packed values
    uint32_t get_packed_uint32(size_t packed_idx) const {
        const size_t total_elements = rows * cols;
        const size_t required_uint32 = (total_elements + INT8_PER_UINT32 - 1) / INT8_PER_UINT32;
        HEEPSTOR_ASSERT(packed_idx < required_uint32 && "Packed index out of bounds");
        return packed_data[packed_idx];
    }

    // Extract an int8_t value at given row and column
    int8_t operator()(size_t r, size_t c) const {
        HEEPSTOR_ASSERT(r < rows && c < cols && "Matrix indices out of bounds");

        const size_t int8_idx = r * cols + c;

        // Calculate which uint32_t contains our value and the byte position within it
        const size_t packed_idx = int8_idx / INT8_PER_UINT32;
        const size_t byte_pos = int8_idx % INT8_PER_UINT32;

        const uint32_t shift = (3 - byte_pos) * 8;

        return static_cast<int8_t>((packed_data[packed_idx] >> shift) & 0xff);
    }

    // Helper to calculate required uint32_t storage for given dimensions
    static constexpr size_t required_storage(size_t r, size_t c) {
        const size_t total_elements = r * c;
        return (total_elements + INT8_PER_UINT32 - 1) / INT8_PER_UINT32;
    }

    static PackedInt8Matrix allocate(size_t rows, size_t cols) {
        HEEPSTOR_ASSERT(rows > 0 && cols > 0 && "Matrix dimensions must be positive");

        const size_t required_uint32 = required_storage(rows, cols);
        uint32_t* packed_storage = StaticArenaAllocator::allocate_array<uint32_t>(required_uint32);

        return PackedInt8Matrix(packed_storage, rows, cols);
    }

    static PackedInt8Matrix allocate_from_int8_list(std::initializer_list<std::initializer_list<int8_t>> init) {
        // Check dimensions
        const size_t r = init.size();
        HEEPSTOR_ASSERT(r > 0 && "Empty initializer list");
        const size_t c = init.begin()->size();
        HEEPSTOR_ASSERT(c > 0 && "Empty row in initializer list");

        const size_t num_uint32 = required_storage(r, c);

        uint32_t* packed_storage = StaticArenaAllocator::allocate_array<uint32_t>(num_uint32);

        // Initialize all storage to 0
        for (size_t i = 0; i < num_uint32; ++i) {
            packed_storage[i] = 0;
        }

        // Pack values
        size_t linear_idx = 0;
        for (const auto& row : init) {
            HEEPSTOR_ASSERT(row.size() == c && "All rows must have the same length");
            for (const auto& val : row) {
                // Calculate which uint32_t and byte position to pack into
                const size_t packed_idx = linear_idx / INT8_PER_UINT32;
                const size_t byte_pos = linear_idx % INT8_PER_UINT32;
                const uint32_t shift = (3 - byte_pos) * 8;

                packed_storage[packed_idx] |= (static_cast<uint32_t>(val) & 0xFF) << shift;
                linear_idx++;
            }
        }

        return PackedInt8Matrix(packed_storage, r, c);
    }

    // Fill all elements with the same value
    void fill(int8_t value) {
        // Create the packed pattern by replicating the value in all 4 byte positions
        uint32_t packed_pattern = 0;
        for (size_t byte_pos = 0; byte_pos < INT8_PER_UINT32; ++byte_pos) {
            const uint32_t shift = (3 - byte_pos) * 8;  // MSB first
            packed_pattern |= (static_cast<uint32_t>(value) & 0xFF) << shift;
        }

        const size_t required_uint32 = required_storage(rows, cols);

        for (size_t i = 0; i < required_uint32; ++i) {
            packed_data[i] = packed_pattern;
        }
    }

    // Fill with random values in range [min_val, max_val]
    void fill_random(int8_t min_val, int8_t max_val, RandomNumberGenerator& rng) {
        const size_t total_elements = rows * cols;

        for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
            const int8_t rand_val = static_cast<int8_t>(rng.rand_int_range(min_val, max_val));

            // Calculate which uint32_t and byte position to pack into
            const size_t packed_idx = linear_idx / INT8_PER_UINT32;
            const size_t byte_pos = linear_idx % INT8_PER_UINT32;
            const uint32_t shift = (3 - byte_pos) * 8;

            // Clear the target byte position and set the new value
            packed_data[packed_idx] &= ~(0xFFu << shift);  // Clear existing byte
            packed_data[packed_idx] |= (static_cast<uint32_t>(rand_val) & 0xFF) << shift;
        }
    }

    void fill_diag(int8_t value) {
        HEEPSTOR_ASSERT(rows == cols && "Matrix must be square for diagonal fill");

        // For each diagonal element
        for (size_t i = 0; i < rows; i++) {
            const size_t linear_idx = i * cols + i;

            // Calculate which uint32_t and byte position to pack into
            const size_t packed_idx = linear_idx / INT8_PER_UINT32;
            const size_t byte_pos = linear_idx % INT8_PER_UINT32;
            const uint32_t shift = (3 - byte_pos) * 8;

            packed_data[packed_idx] &= ~(0xFFu << shift);
            packed_data[packed_idx] |= (static_cast<uint32_t>(value) & 0xFF) << shift;
        }
    }

    // Getters
    size_t num_rows() const { return rows; }
    size_t num_cols() const { return cols; }
    const uint32_t* get_packed_data() const { return packed_data; }
};