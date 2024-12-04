#pragma once

#include <cassert>
#include "heepstor_defs.h"

#if ENABLE_DEBUG_HEEPSTOR_ASSERTIONS
#define HEEPSTOR_ASSERT(x) assert(x)
#else
#define HEEPSTOR_ASSERT(x)
#endif