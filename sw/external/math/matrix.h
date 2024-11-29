#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include "floating_point_ops.h"
#include "heepstor_assert.h"
#include "matrix_tile.h"
#include "packed_int8_matrix.h"
#include "random_number_generator.h"
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

    // Software matrix multiplication with uint8_t packed matrix
    Matrix multiply_software_with_packed(const PackedInt8Matrix& rhs) {
        static_assert(std::is_same<T, float>::value, "PackedInt8Matrix multiplication only supported with float matrices");
        HEEPSTOR_ASSERT(cols == rhs.num_rows() && "Invalid dimensions for multiplication");

        Matrix<T> result(rows, rhs.num_cols());

        multiply_software_with_packed(rhs, result);
        return result;
    }

    void multiply_software_with_packed(const PackedInt8Matrix& rhs, Matrix& out) {
        static_assert(std::is_same<T, float>::value, "PackedInt8Matrix multiplication only supported with float matrices");
        HEEPSTOR_ASSERT(cols == rhs.num_rows() && "Invalid dimensions for multiplication");

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < rhs.num_cols(); ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < cols; ++k) {
                    sum += (*this)(i, k) * static_cast<float>(rhs(k, j));
                }
                out(i, j) = sum;
            }
        }
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

    // Fills the matrix with random float values in [min_val, max_val) using a provided random number generator.
    //
    // This method is only available for matrices of type float. Attempting to call it on
    // matrices of other types will result in a compilation error.
    template <typename U = T>
    typename std::enable_if<std::is_same<U, float>::value>::type fill_random(float min_val, float max_val, RandomNumberGenerator& rng) {
        for (size_t i = 0; i < rows * cols; ++i) {
            data[i] = rng.rand_float_uniform_range(min_val, max_val);
        }
    }

    // Fills the matrix with random float values in [min_val, max_val) using a new random number generator
    //  initialized with the default seed.
    //
    // This method is only available for matrices of type float. Attempting to call it on
    // matrices of other types will result in a compilation error.
    template <typename U = T>
    typename std::enable_if<std::is_same<U, float>::value>::type fill_random(float min_val, float max_val) {
        RandomNumberGenerator rng;  // Create with default seed
        fill_random(min_val, max_val, rng);
    }

    // Creates a square identity matrix of the given size.
    static Matrix identity(size_t size) {
        Matrix result(size, size);
        result.fill(T());

        for (size_t i = 0; i < size; ++i) {
            result(i, i) = T(1);
        }
        return result;
    }

    // Creates a matrix filled with zeros
    static Matrix zeros(size_t rows, size_t cols) {
        Matrix result(rows, cols);
        result.fill(T());
        return result;
    }

    // Creates a matrix of size rows x cols where the elements
    //  are increasing sequences of integers, for example for a 3x4 matrix:
    //  [[1,  2,  3,  4]
    //   [5,  6,  7,  8]
    //   [9, 10, 11, 12]]
    static Matrix sequence(size_t rows, size_t cols) {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows * cols; ++i) {
            result.data[i] = static_cast<T>(i + 1);
        }
        return result;
    }

    // Computes the maximum relative error between elements of the matrices: max(abs((a - b) / (a + eps)))
    // Uses an epsilon value in the denominator to handle zero reference values.
    T relative_error(const Matrix& ground_truth) const {
        HEEPSTOR_ASSERT(rows == ground_truth.rows && cols == ground_truth.cols && "Matrix dimensions must match for relative error computation");

        // Define epsilon based on the type T
        const T eps = std::is_same<T, float>::value ? 1e-6f : std::is_same<T, double>::value ? 1e-10 : T(1);

        T max_error = T{};
        for (size_t i = 0; i < rows * cols; ++i) {
            // Add epsilon to denominator to handle zero values
            T rel_err = std::abs((data[i] - ground_truth.data[i]) / (std::abs(data[i]) + eps));
            max_error = std::max(max_error, rel_err);
        }

        return max_error;
    }

    MatrixTile<T> get_tile(size_t start_row, size_t start_col, size_t tile_rows, size_t tile_cols) {
        HEEPSTOR_ASSERT(start_row + tile_rows <= rows && start_col + tile_cols <= cols);
        return MatrixTile<T>(data, rows, cols, start_row, start_col, tile_rows, tile_cols);
    }

    const MatrixTile<T> get_tile(size_t start_row, size_t start_col, size_t tile_rows, size_t tile_cols) const {
        HEEPSTOR_ASSERT(start_row + tile_rows <= rows && start_col + tile_cols <= cols);
        return MatrixTile<T>(data, rows, cols, start_row, start_col, tile_rows, tile_cols);
    }

    MatrixTile<T> as_tile() { return MatrixTile<T>(data, rows, cols, 0, 0, rows, cols); }

    const MatrixTile<T> as_tile() const { return MatrixTile<T>(data, rows, cols, 0, 0, rows, cols); }

    // Getters
    size_t num_rows() const { return rows; }
    size_t num_cols() const { return cols; }
    T* get_data() { return data; }
    const T* get_data() const { return data; }
    bool is_static() const { return is_static_data; }
};