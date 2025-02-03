#pragma once

#include <cstdint>

// Computes the largest integer value not greater than x
int32_t fastfloorf(float x);

/* Compute 2**p, for p in [-126, 128). Maximum relative error: 5.04e-5; RMS error: 1.03e-5 */
float fastexp2(float p);

float fastexp(float p);

void printFloat(float x);