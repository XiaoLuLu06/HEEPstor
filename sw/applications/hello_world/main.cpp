#include <stdio.h>
#include <stdlib.h>
#include "csr.h"

#include "systolic_array.h"
#include "heepstor.h"
#include "floating_point_ops.h"
#include "heepstor_assert.h"

float __attribute__ ((noinline)) float_add(float a, float b){
    return a + b;
}

void enable_floating_point_unit() {
    CSR_SET_BITS(CSR_REG_MSTATUS, 0x1 << 13);
}

static const float pwr10[] = {1e36,  1e33,  1e30,  1e27,  1e24,  1e21,  1e18,
                              1e15,  1e12,  1e9,   1e6,   1e3,   1e0,   1e-3,
                              1e-6,  1e-9,  1e-12, 1e-15, 1e-18, 1e-21, 1e-24,
                              1e-27, 1e-30, 1e-33, 1e-36};

void ftoa(float f, char *buf, size_t bufsiz) {
  char sign[2] = {0};
  uint32_t p;

  if (f < 0) {
    f = -f;
    sign[0] = '-';
  }

  if (f == 0)
    p = 12;
  else
    for (p = 0; p < sizeof(pwr10) / sizeof(pwr10[0]) - 1; p++)
      if (f >= pwr10[p])
        break;
  uint32_t exponent = 36 - 3 * p;
  float mantissa = f / pwr10[p];

  mantissa += 0.00005; // 4 digit precision
  uint32_t digits = mantissa;
  uint32_t fraction = (mantissa - digits) * 10000.0;
  snprintf(buf, bufsiz, "%s%d.%04de%d", sign, digits, fraction, exponent);
}

void printFloat(float x) {
  static char fbuf[16];
  ftoa(x, fbuf, sizeof(fbuf));
  printf("%s", fbuf);
}

void run_fp32_test_suite(volatile float a, volatile float b) {
      volatile float res = float_add(a,  b);

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

template<size_t M, size_t N>
void print_matrix(float m[M][N]) {
  printf("[");
  for (size_t i = 0; i < M; i++) {
      printf("[");
      
      for (size_t j = 0; j < N; j++) {
          printFloat(m[i][j]);
          if (j < N-1) printf(", ");
      }
      printf("]");
      if (i < M-1) printf("\n ");
  }
  printf("]\n");
}

void test_systolic_array_size_4() {
  SystolicArray systolic_array = SystolicArray::get_default();      

  float lhs[5][4] = {
    { 1.0, 2.0, 3.0, 4.0 }, 
    { 5.0, 6.0, 7.0, 8.0 }, 
    { 9.0, 10.0, 11.0, 12.0 }, 
    { 13.0, 14.0, 15.0, 16.0 },
    { 17.0, 18.0, 19.0, 20.0 }, 
  };

  printf("LHS: \n");
  print_matrix<5, 4>(lhs);

  uint32_t weights[] = {
      0x01000000,
      0x00010000,
      0x00000100,
      0x00000001,
  };

  // uint32_t weights[] = {
  //     0x01000000,
  //     0x00020000,
  //     0x00000300,
  //     0x00000004,
  // };

  float res[5][4] = {0};
  systolic_array.matrix_matrix_multiply(&lhs[0][0], weights, &res[0][0], 2, 4, 4);
  
  printf("Res: \n");
  print_matrix<5, 4>(res); 
}

void test_systolic_array_size_8() {
  SystolicArray systolic_array = SystolicArray::get_default();      

  float lhs[2][8] = {
    { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 }, 
    { 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 },
  };

  printf("LHS: \n");
  print_matrix<2, 8>(lhs);

  uint32_t weights[] = {
      0x81000000, 0x00000000,
      0x00010000, 0x00000000,
      0x00000100, 0x00000000,
      0x00000001, 0x00000000,
      0x00000000, 0x01000000,
      0x00000000, 0x00010000,
      0x00000000, 0x00000100,
      0x00000000, 0x00000001
  };

  // uint32_t weights[] = {
  //     0x01000000,
  //     0x00020000,
  //     0x00000300,
  //     0x00000004,
  // };

  float res[2][8] = {0};
  systolic_array.matrix_matrix_multiply(&lhs[0][0], weights, &res[0][0], 2, 8, 8);
  
  printf("Res: \n");
  print_matrix<2, 8>(res); 
}

int main(int argc, char *argv[])
{
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
    }
    
    return EXIT_SUCCESS;
}

