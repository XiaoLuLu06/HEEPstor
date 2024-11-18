#pragma once

#include <cstdint>

#include "mmio.h"
#include "core_v_mini_mcu.h"

class SystolicArray {
private:
    mmio_region_t base_addr;
public:
    SystolicArray(mmio_region_t base_addr);

    void write(ptrdiff_t offset, uint32_t val);
    uint32_t read(ptrdiff_t offset);
};