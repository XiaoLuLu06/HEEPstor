#include "fpu.hpp"
#include "csr.h"

void FloatingPointUnit::enable() {
    CSR_SET_BITS(CSR_REG_MSTATUS, 0x1 << 13);
}