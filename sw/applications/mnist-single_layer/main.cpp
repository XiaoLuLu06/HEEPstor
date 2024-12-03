#include <stdio.h>
#include "csr.h"
#include "gen/model.hpp"

void enable_floating_point_unit() {
    CSR_SET_BITS(CSR_REG_MSTATUS, 0x1 << 13);
}

int main(int argc, char* argv[]) {
    enable_floating_point_unit();
    printf("Model test");
}