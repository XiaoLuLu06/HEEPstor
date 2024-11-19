#include <stdio.h>
#include <stdlib.h>
#include "csr.h"

#include "systolic_array.h"
#include "heepstor.h"
#include "floating_point_ops.h"

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
      
      printf("std exp:");
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

int main(int argc, char *argv[])
{
    printf("Hello, heepstor!");

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
      SystolicArray systolic_array {mmio_region_from_addr(SYSTOLIC_ARRAY_START_ADDRESS)};
      systolic_array.write(0x12340, 23);

      uint32_t res = systolic_array.read(0x42);
      printf("Read result: 0x%x\n", res);
    }
    
    return EXIT_SUCCESS;
}

