// Copyright EPFL contributors.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef _SYSTOLIC_ARRAY_DEF_H_
#define _SYSTOLIC_ARRAY_DEF_H_

#define SYSTOLIC_ARRAY_SIZE ${systolic_array_size}

#define BUS_SIZE 32

// Activations are fp32
#define ACTIVATION_SIZE 32

// Weights are int8_t
#define WEIGHT_SIZE 8

// How many activations can fit in a single bus transaction
#define ACTIVATIONS_PER_BUS_TRANSACTION (BUS_SIZE / ACTIVATION_SIZE)

// How many Weights can fit in a single bus transaction
#define WEIGHTS_PER_BUS_TRANSACTION (BUS_SIZE / WEIGHT_SIZE)

#endif _SYSTOLIC_ARRAY_DEF_H_