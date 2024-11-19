#include <cstdint>
#include <cstdio>
#include <cstring>

#include "floating_point_ops.h"

/* compute 2**p, for p in [-126, 128). Maximum relative error: 5.04e-5; RMS error: 1.03e-5 */
// From https://stackoverflow.com/questions/65554112/fast-double-exp2-function-in-c
float fastexp2(float p) {
    constexpr int FP32_MIN_EXPO = -126;  // exponent of minimum binary32 normal
    constexpr int FP32_MANT_BITS = 23;   // number of stored mantissa (significand) bits
    constexpr int FP32_EXPO_BIAS = 127;  // binary32 exponent bias

    p = (p < FP32_MIN_EXPO) ? FP32_MIN_EXPO : p;  // clamp below

    /* 2**p = 2**(w+z), with w an integer and z in [0, 1) */
    int32_t w = fastfloorf(p);  // integral part
    float z = p - w;            // fractional part

    /* approximate 2**z-1 for z in [0, 1) */
    float approx = -0x1.6e7592p+2f + 0x1.bba764p+4f / (0x1.35ed00p+2f - z) - 0x1.f5e546p-2f * z;

    /* assemble the exponent and mantissa components into final result */
    int32_t resi = ((1 << FP32_MANT_BITS) * ((w + FP32_EXPO_BIAS) + approx));

    float res;
    memcpy(&res, &resi, sizeof(res));

    return res;
}

// From https://stackoverflow.com/questions/47025373/fastest-implementation-of-the-natural-exponential-function-using-sse
float fastexp(float p) {
    // LOG2_INV = 1 / log(2.0)
    constexpr float LOG2_INV = 1.44269504;
    return fastexp2(p * LOG2_INV);
}

// Computes the largest integer value not greater than x. Ignores NaN and infinity.
int32_t fastfloorf(float x) {
    int32_t result;

    __asm__ volatile("fcvt.w.s %0, %1, rdn"  // Convert float to int, rounding down
                     : "=r"(result)          // Output: result in integer register
                     : "f"(x)                // Input: single-precision float in floating-point register
                     :                       // No clobbered registers
    );

    return result;
}

static const float pwr10[] = {1e36, 1e33, 1e30, 1e27,  1e24,  1e21,  1e18,  1e15,  1e12,  1e9,   1e6,   1e3,  1e0,
                              1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18, 1e-21, 1e-24, 1e-27, 1e-30, 1e-33, 1e-36};

void ftoa(float f, char* buf, size_t bufsiz) {
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

    mantissa += 0.00005;  // 4 digit precision
    uint32_t digits = mantissa;
    uint32_t fraction = (mantissa - digits) * 10000.0;
    snprintf(buf, bufsiz, "%s%d.%04de%d", sign, digits, fraction, exponent);
}

void printFloat(float x) {
    static char fbuf[16];
    ftoa(x, fbuf, sizeof(fbuf));
    printf("%s", fbuf);
}