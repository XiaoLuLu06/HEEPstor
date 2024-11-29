#pragma once

#include <cstddef>
#include "heepstor_assert.h"

template <typename T>
class MatrixTile {
private:
    T* data;
    size_t start_row_idx;
    size_t start_col_idx;

    size_t num_tile_rows;
    size_t num_tile_cols;

public:
    // Iterator used to fastly access the matrix tile in row-major order. Used in the systolic
    //  array to simplify the tile multiplication kernel.
    template <bool IsConst>
    class IteratorImpl {
    private:
        using DataPtr = typename std::conditional<IsConst, const T*, T*>::type;
        DataPtr current_ptr;
        DataPtr row_start_ptr;
        size_t stride;
        size_t current_col;
        size_t tile_cols;

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = typename std::conditional<IsConst, const T*, T*>::type;
        using reference = typename std::conditional<IsConst, const T&, T&>::type;

        __attribute__((always_inline)) IteratorImpl(typename std::conditional<IsConst, const MatrixTile&, MatrixTile&>::type t, size_t row = 0,
                                                    size_t col = 0)
            : current_ptr(t.data + (t.start_row_idx + row) * t.stride + (t.start_col_idx + col)),
              row_start_ptr(t.data + (t.start_row_idx + row) * t.stride + t.start_col_idx),
              stride(t.stride),
              current_col(col),
              tile_cols(t.num_tile_cols) {}

        __attribute__((always_inline)) reference operator*() { return *current_ptr; }
        __attribute__((always_inline)) pointer operator->() { return current_ptr; }

        __attribute__((always_inline)) IteratorImpl& operator++() {
            ++current_ptr;
            if (++current_col >= tile_cols) {
                current_col = 0;
                row_start_ptr += stride;
                current_ptr = row_start_ptr;
            }
            return *this;
        }

        __attribute__((always_inline)) IteratorImpl operator++(int) {
            IteratorImpl tmp = *this;
            ++(*this);
            return tmp;
        }

        __attribute__((always_inline)) bool operator==(const IteratorImpl& other) const { return current_ptr == other.current_ptr; }
        __attribute__((always_inline)) bool operator!=(const IteratorImpl& other) const { return !(*this == other); }
        __attribute__((always_inline)) bool is_at_last_column() const { return current_col == tile_cols - 1; }
    };

    using Iterator = IteratorImpl<false>;
    using ConstIterator = IteratorImpl<true>;

    MatrixTile(T* data, size_t num_matrix_rows, size_t num_matrix_cols, size_t start_row_idx, size_t start_col_idx, size_t num_tile_rows,
               size_t num_tile_cols)
        : data(data),
          start_row_idx(start_row_idx),
          start_col_idx(start_col_idx),
          num_tile_rows(num_tile_rows),
          num_tile_cols(num_tile_cols),
          stride(num_matrix_cols) {
        HEEPSTOR_ASSERT(start_row_idx + num_tile_rows <= num_matrix_rows);
        HEEPSTOR_ASSERT(start_col_idx + num_tile_cols <= num_matrix_cols);
    }

    __attribute__((always_inline)) Iterator begin() { return Iterator(*this); }
    __attribute__((always_inline)) Iterator end() { return Iterator(*this, num_tile_rows, 0); }

    __attribute__((always_inline)) ConstIterator begin() const { return ConstIterator(*this); }
    __attribute__((always_inline)) ConstIterator end() const { return ConstIterator(*this, num_tile_rows, 0); }

    __attribute__((always_inline)) size_t num_rows() const { return num_tile_rows; }
    __attribute__((always_inline)) size_t num_cols() const { return num_tile_cols; }

    size_t get_num_tile_rows() const { return num_tile_rows; }
    size_t get_num_tile_cols() const { return num_tile_cols; }

private:
    size_t stride;
};