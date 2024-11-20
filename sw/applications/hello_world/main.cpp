#include <stdio.h>
#include <stdlib.h>
#include "csr.h"
#include "matrix.h"

#include "floating_point_ops.h"
#include "heepstor.h"
#include "heepstor_assert.h"
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

    uint32_t weights[] = {
        0x01000000,
        0x00010000,
        0x00000100,
        0x00000001,
    };

    Matrix<float> res(5, 4);
    res.fill(0);

    systolic_array.matrix_matrix_multiply(lhs.get_data(), weights, res.get_data(), 5, 4, 4);

    printf("Res: \n");
    res.print();
}

void test_systolic_array_size_8() {
    SystolicArray systolic_array = SystolicArray::get_default();

    Matrix<float> lhs{
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
        {9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
    };

    printf("LHS: \n");
    lhs.print();

    uint32_t weights[] = {
        0x81000000, 0x00000000,  //
        0x00010000, 0x00000000,  //
        0x00000100, 0x00000000,  //
        0x00000001, 0x00000000,  //
        0x00000000, 0x01000000,  //
        0x00000000, 0x00010000,  //
        0x00000000, 0x00000100,  //
        0x00000000, 0x00000001,
    };

    Matrix<float> res(2, 8);
    res.fill(0);

    systolic_array.matrix_matrix_multiply(lhs.get_data(), weights, res.get_data(), 2, 8, 8);

    printf("Res: \n");
    res.print();
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
        test_systolic_array_size_4();

        // TODO: Disable HEEPSTOR assert, make it a Makefile option
    }

    // 4. Report arena allocator statistics
    {
        printf("\n ARENA ALLOCATOR: \n");
        printf("Available bytes: %d, Used bytes: %d\n", StaticArenaAllocator::available_bytes(), StaticArenaAllocator::used_bytes());
    }

    // TODO: Test two's complement to sign+magnitude. Test that -128 saturates to -127.
    // TODO: Add copyright notices + my name to the files.

    return EXIT_SUCCESS;
}
