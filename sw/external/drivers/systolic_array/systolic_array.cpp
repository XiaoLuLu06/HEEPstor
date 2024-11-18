// TODO: Create driver 

#include "systolic_array.h"

SystolicArray::SystolicArray(mmio_region_t base_addr)
    : base_addr(base_addr)
{
}

void SystolicArray::write(ptrdiff_t offset, uint32_t val)
{
    mmio_region_write32(base_addr, offset, val);
}

uint32_t SystolicArray::read(ptrdiff_t offset)
{
    return mmio_region_read32(base_addr, offset);
}
