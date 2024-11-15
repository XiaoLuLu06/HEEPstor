#include <stdio.h>
#include <stdlib.h>
#include "csr.h"

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

int main(int argc, char *argv[])
{
    char fbuf[16];

    printf("Enabling FPU...\n");
    enable_floating_point_unit();
    printf("FPU enabled!\n");

    /* write something to stdout */
    while(1)
    {
        volatile float res = float_add(1.2,  3.14);
        ftoa(res, fbuf, sizeof(fbuf));
        printf("hello world otro! float: %s\n", fbuf);

        // printf("hola mundo, pedro \n");    
    }
    return EXIT_SUCCESS;
}

