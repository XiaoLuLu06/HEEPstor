// Copyright EPFL contributors.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// TODO: Update ifdef
#ifndef CGRA_X_HEEP_H_
#define CGRA_X_HEEP_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include "core_v_mini_mcu.h"

// TODO: I think we don't need the two following defines for EXT_XBAR
// #define EXT_XBAR_NMASTER 0
// #define EXT_XBAR_NSLAVE  1

#define SYSTOLIC_ARRAY_START_ADDRESS (EXT_SLAVE_START_ADDRESS + 0x000000)
#define SYSTOLIC_ARRAY_MEM_SIZE 0x100000
#define SYSTOLIC_ARRAY_END_ADDRESS (SYSTOLIC_ARRAY_START_ADDRESS + SYSTOLIC_ARRAY_MEM_SIZE)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // CGRA_X_HEEP_H_
