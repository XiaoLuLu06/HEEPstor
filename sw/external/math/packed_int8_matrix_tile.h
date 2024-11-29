#pragma once

#include <cstddef>
#include <cstdint>
#include "heepstor_assert.h"

class PackedInt8MatrixTile {
private:
    const uint32_t* data;

    size_t start_row_idx;
    size_t start_col_idx;
    size_t num_tile_rows;
    size_t num_tile_cols;

    size_t matrix_cols;  // Full matrix columns (needed for stride calculation)
    size_t uint32_stride;

    static constexpr size_t INT8_PER_UINT32 = 4;

    // Helper to ensure positions are aligned to INT8_PER_UINT32
    static constexpr bool is_aligned(size_t value) { return (value % INT8_PER_UINT32) == 0; }

    // Helper to calculate the padded number of columns (smallest multiple of 4 >= r)
    static constexpr size_t num_cols_after_padding(size_t cols) { return (cols + INT8_PER_UINT32 - 1) & ~(INT8_PER_UINT32 - 1); }

public:
    PackedInt8MatrixTile(const uint32_t* data, size_t num_matrix_rows, size_t num_matrix_cols, size_t start_row_idx, size_t start_col_idx,
                         size_t num_tile_rows, size_t num_tile_cols)
        : data(data),
          start_row_idx(start_row_idx),
          start_col_idx(start_col_idx),
          num_tile_rows(num_tile_rows),
          num_tile_cols(num_tile_cols),
          matrix_cols(num_matrix_cols),
          uint32_stride(num_cols_after_padding(num_matrix_cols) / 4) {

        HEEPSTOR_ASSERT(data != nullptr && "Data pointer cannot be null");
        HEEPSTOR_ASSERT(is_aligned(start_row_idx) && is_aligned(start_col_idx) && "Tile start position must be aligned to 4");
        HEEPSTOR_ASSERT(start_row_idx + num_tile_rows <= num_matrix_rows && start_col_idx + num_tile_cols <= num_matrix_cols &&
                        "Tile dimensions exceed matrix bounds");
    }

    __attribute__((always_inline)) const uint32_t* last_packed_pointer() const {
        const size_t last_row_offset = (start_row_idx + num_tile_rows - 1) * uint32_stride;
        const size_t last_col_block = start_col_idx / INT8_PER_UINT32 + (num_blocks_cols() - 1);

        return data + last_row_offset + last_col_block;
    }

    __attribute__((always_inline)) const uint32_t* move_pointer_one_row_up(const uint32_t* ptr) const { return ptr - uint32_stride; }

    // Get the number of uint32_t blocks in rows and columns
    __attribute__((always_inline)) size_t num_blocks_rows() const { return num_tile_rows; }
    __attribute__((always_inline)) size_t num_blocks_cols() const { return (num_tile_cols + INT8_PER_UINT32 - 1) / INT8_PER_UINT32; }

    __attribute__((always_inline)) size_t get_stride() const { return uint32_stride; }

    __attribute__((always_inline)) size_t num_rows() const { return num_tile_rows; }
    __attribute__((always_inline)) size_t num_cols() const { return num_tile_cols; }
};