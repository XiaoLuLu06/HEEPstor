#pragma once
#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include "floating_point_ops.h"
#include "heepstor_assert.h"
#include "static_arena_allocator.h"

template <typename T>
class Matrix {
private:
    T* data;
    size_t rows;
    size_t cols;
    bool is_static_data;

    static void validate_dimensions(size_t r, size_t c) { HEEPSTOR_ASSERT(r > 0 && c > 0 && "Matrix dimensions must be positive"); }

public:
    // Constructor for arena-allocated data
    Matrix(size_t r, size_t c) : rows(r), cols(c), is_static_data(false) {
        validate_dimensions(r, c);
        data = StaticArenaAllocator::allocate_array<T>(r * c);
    }

    // Constructor with initializer list
    Matrix(std::initializer_list<std::initializer_list<T>> init) : rows(init.size()), cols(init.begin()->size()), is_static_data(false) {
        validate_dimensions(rows, cols);
        data = StaticArenaAllocator::allocate_array<T>(rows * cols);

        size_t i = 0;
        for (const auto& row : init) {
            HEEPSTOR_ASSERT(row.size() == cols && "All rows must have the same length");
            for (const auto& val : row) {
                data[i++] = val;
            }
        }
    }

    // Constructor for static data
    Matrix(T* static_data, size_t r, size_t c) : data(static_data), rows(r), cols(c), is_static_data(true) {
        validate_dimensions(r, c);
        HEEPSTOR_ASSERT(static_data != nullptr && "Static data pointer cannot be null");
    }

    // Prevent copying
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;

    // Allow moving
    Matrix(Matrix&& other) noexcept : data(other.data), rows(other.rows), cols(other.cols), is_static_data(other.is_static_data) {
        other.data = nullptr;
        other.rows = other.cols = 0;
    }

    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            data = other.data;
            rows = other.rows;
            cols = other.cols;
            is_static_data = other.is_static_data;
            other.data = nullptr;
            other.rows = other.cols = 0;
        }
        return *this;
    }

    // Destructor doesn't free arena memory
    ~Matrix() = default;

    // Element access
    T& operator()(size_t r, size_t c) {
        HEEPSTOR_ASSERT(r < rows && c < cols && "Matrix indices out of bounds");
        return data[r * cols + c];
    }

    const T& operator()(size_t r, size_t c) const {
        HEEPSTOR_ASSERT(r < rows && c < cols && "Matrix indices out of bounds");
        return data[r * cols + c];
    }

    // In-place addition
    Matrix& operator+=(const Matrix& other) {
        HEEPSTOR_ASSERT(rows == other.rows && cols == other.cols && "Matrix dimensions must match for addition");

        for (size_t i = 0; i < rows * cols; ++i) {
            data[i] += other.data[i];
        }
        return *this;
    }

    // Regular addition using in-place operator
    Matrix operator+(const Matrix& other) const {
        Matrix result(rows, cols);
        result = std::move(*this);
        result += other;
        return result;
    }

    // Software matrix multiplication
    Matrix multiply_software(const Matrix& other) const {
        HEEPSTOR_ASSERT(cols == other.rows && "Invalid dimensions for multiplication");

        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                T sum = T();
                for (size_t k = 0; k < cols; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    void print() const {
        printf("[");
        for (size_t i = 0; i < rows; i++) {
            printf("[");
            for (size_t j = 0; j < cols; j++) {
                // By default we'll print it as an integer, except if it's a matrix of float
                if constexpr (std::is_same<T, float>::value) {
                    printFloat((*this)(i, j));
                } else {
                    printf("%d", (*this)(i, j));
                }
                if (j < cols - 1)
                    printf(", ");
            }
            printf("]");
            if (i < rows - 1)
                printf("\n ");
        }
        printf("]\n");
    }

    void fill(const T& value) {
        for (size_t i = 0; i < rows * cols; ++i) {
            data[i] = value;
        }
    }

    // Getters
    size_t num_rows() const { return rows; }
    size_t num_cols() const { return cols; }
    T* get_data() { return data; }
    const T* get_data() const { return data; }
    bool is_static() const { return is_static_data; }
};