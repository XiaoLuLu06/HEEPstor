#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "csr.h"
#include "matrix.h"

#include "floating_point_ops.h"
#include "heepstor.h"
#include "heepstor_assert.h"
#include "packed_int8_matrix.h"
#include "static_arena_allocator.h"
#include "systolic_array.h"

float __attribute__((noinline)) float_add(float a, float b) {
    return a + b;
}

void enable_floating_point_unit() {
    CSR_SET_BITS(CSR_REG_MSTATUS, 0x1 << 13);
}

void run_fp32_test_suite(volatile float a, volatile float b) {
    volatile float res = float_add(a, b);

    printf("add: ");
    printFloat(res);
    printf("\n");

    printf("std exp: ");
    printFloat(__builtin_expf(res));
    printf("\n");

    printf("fastfloorf: ");
    printFloat(fastfloorf(res));
    printf("\n");

    printf("fastexp2: ");
    printFloat(fastexp2(res));
    printf("\n");

    printf("fastexp: ");
    printFloat(fastexp(res));
    printf("\n");
}

// Returns max_relative_error_percentage
float run_random_tests(int m, int n, int p, int num_tests, float min_val, float max_val, RandomNumberGenerator rng) {
    SystolicArray systolic_array = SystolicArray::get_default();

    Matrix<float> lhs(m, n);

    Matrix<float> res_hw(m, p);
    Matrix<float> res_sw(m, p);

    auto packed_weights = PackedInt8Matrix::allocate(n, p);

    float relative_error_percentage_sum = 0;
    float relative_error_percentage_max = -1;

    for (int i = 0; i < num_tests; ++i) {
        printf("\n########### TEST NUM %d ###########\n\n", i);

        res_hw.fill(0);
        lhs.fill_random(min_val, max_val, rng);

        // printf("LHS (#%d): \n", i);
        // lhs.print();

        packed_weights.fill_random(-127, 127, rng);

        systolic_array.matrix_matrix_multiply(lhs, packed_weights, res_hw);
        lhs.multiply_software_with_packed(packed_weights, res_sw);

        printf("Res HW (#%d): \n", i);
        res_hw.print();

        printf("Res SW (#%d): \n", i);
        res_sw.print();

        auto relative_error_percentage = res_hw.relative_error(res_sw) * 100.0f;
        printf("Relative error (#%d): ", i);
        printFloat(relative_error_percentage);
        printf("%%\n");

        relative_error_percentage_sum += relative_error_percentage;
        relative_error_percentage_max = std::max(relative_error_percentage_max, relative_error_percentage);
    }

    printf("\n ########## STATISTICAL RESULTS #############\n\n");
    printf("Average relative error: ");
    printFloat(relative_error_percentage_sum / num_tests);
    printf("%%\n");
    printf("Max relative error: ");
    printFloat(relative_error_percentage_max);
    printf("%%\n");

    return relative_error_percentage_max;
}

void test_systolic_array_size_4() {
    SystolicArray systolic_array = SystolicArray::get_default();

    Matrix<float> lhs{
        {1.0, 2.0, 3.0, 4.0},      //
        {5.0, 6.0, 7.0, 8.0},      //
        {9.0, 10.0, 11.0, 12.0},   //
        {13.0, 14.0, 15.0, 16.0},  //
        {17.0, 18.0, 19.0, 20.0},
    };

    printf("LHS: \n");
    lhs.print();

    // auto packed_weights = PackedInt8Matrix::allocate_from_int8_list({{1, 0, 0, 0},  //
    //                                                                  {0, -1, 0, 0},
    //                                                                  {0, 0, 1, 0},
    //                                                                  {0, 0, 0, 1}});

    auto packed_weights = PackedInt8Matrix::allocate(4, 4);
    packed_weights.fill_diag(-2);

    Matrix<float> res(5, 4);
    res.fill(0);

    systolic_array.matrix_matrix_multiply(lhs, packed_weights, res);

    auto res_sw = lhs.multiply_software_with_packed(packed_weights);

    printf("Res: \n");
    res.print();

    printf("Res SW: \n");
    res_sw.print();

    auto relative_error_percentage = res.relative_error(res_sw) * 100.0f;
    printf("Relative error: ");
    printFloat(relative_error_percentage);
    printf("%%\n");
}

void test_systolic_array_size_8() {
    SystolicArray systolic_array = SystolicArray::get_default();

    Matrix<float> lhs{
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
        {9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
    };

    printf("LHS: \n");
    lhs.print();

    auto packed_weights = PackedInt8Matrix::allocate_from_int8_list({
        {1, 0, 0, 0, 0, 0, 0, 0},  //
        {0, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 0, 1},
    });

    Matrix<float> res(2, 8);
    res.fill(0);

    systolic_array.matrix_matrix_multiply(lhs, packed_weights, res);

    auto res_sw = lhs.multiply_software_with_packed(packed_weights);

    printf("Res: \n");
    res.print();

    printf("Res SW: \n");
    res_sw.print();

    auto relative_error_percentage = res.relative_error(res_sw) * 100.0f;
    printf("Relative error: ");
    printFloat(relative_error_percentage);
    printf("%%\n");
}

int main(int argc, char* argv[]) {
    printf("\n");
    printf("====================================\n");
    printf("Hello from HEEPstor! \n");
    printf("SYSTOLIC_ARRAY_SIZE=%u\n", SystolicArray::SIZE);
    printf("====================================\n");
    printf("\n");

    // 1. Enable the Floating-Point Unit
    {
        printf("Enabling FPU...\n");
        enable_floating_point_unit();
        printf("FPU enabled!\n");
    }

    // 2. Test floating poinit operations
    {
        printf("\n");
        run_fp32_test_suite(1.2, 3.14);
        printf("\n");
        run_fp32_test_suite(-5.6, 0.7);
        printf("\n");
    }

    // 3. Use the systolic array peripheral
    {
        // test_systolic_array_size_8();

        RandomNumberGenerator rng;

        float total_max_relative_error_percentage = -1;

        auto t = [&](int m, int n, int p, int num_runs) {
            total_max_relative_error_percentage =
                std::max(total_max_relative_error_percentage, run_random_tests(m, n, p, num_runs, -1000, 1000, rng));
        };

        // t(15, 4, 4, 5);
        // t(6, 4, 4, 5);
        // t(3, 4, 4, 5);
        // t(2, 4, 4, 5);
        // t(3, 1, 2, 1);

        t(8, 8, 8, 2);
        t(10, 9, 7, 2);
        t(8, 9, 10, 2);
        t(8, 9, 9, 2);
        t(13, 30, 41, 2);

        // Test small matrices
        // for (int i = 1; i <= 4; ++i) {
        //     for (int j = 1; j <= 4; ++j) {
        //         t(3, i, j, 3);
        //         t(4, i, j, 3);
        //         t(10, i, j, 3);
        //     }
        // }

        printf("\n================================================ \n");
        printf("All random tests max relative err: ");
        printFloat(total_max_relative_error_percentage);
        printf("%%\n================================================ \n");

        // TODO: Disable HEEPSTOR assert, make it a Makefile option
    }

    // 4. Report arena allocator statistics
    {
        printf("\n ARENA ALLOCATOR: \n");
        printf("Available bytes: %d, Used bytes: %d\n", StaticArenaAllocator::available_bytes(), StaticArenaAllocator::used_bytes());
    }

    printf("TODO: Disable HEEPSTOR assert for performance!\n");

    // TODO: Test two's complement to sign+magnitude. Test that -128 saturates to -127.
    // TODO: Add copyright notices + my name to the files.

    return EXIT_SUCCESS;
}
