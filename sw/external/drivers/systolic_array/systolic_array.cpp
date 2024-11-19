// TODO: Create driver 

#include "systolic_array.h"
#include <cstring>

SystolicArray::SystolicArray(mmio_region_t base_addr)
    : base_addr(base_addr)
{
}

void SystolicArray::write_weights(uint32_t four_packed_weights) {
    uint32_t cmd = static_cast<uint32_t>(Command::WRITE_WEIGHTS);
    write_32_bits(cmd << 18, four_packed_weights);
}

float SystolicArray::stream(uint32_t idx, float activation)
{
    uint32_t cmd = static_cast<uint32_t>(Command::STREAM);

    uint32_t raw_activation_bits;
    std::memcpy(&raw_activation_bits, &activation, sizeof(raw_activation_bits));
    
    write_32_bits((cmd << 18) | (idx << 2), raw_activation_bits);
    uint32_t raw_res = read_32_bits(0);

    float res;    
    std::memcpy(&res, &raw_res, sizeof(res));
    return res;
}

float SystolicArray::queue(uint32_t idx, float activation)
{
    uint32_t cmd = static_cast<uint32_t>(Command::QUEUE);

    uint32_t raw_activation_bits;
    std::memcpy(&raw_activation_bits, &activation, sizeof(raw_activation_bits));
    
    write_32_bits((cmd << 18) | (idx << 2), raw_activation_bits);
    uint32_t raw_res = read_32_bits(0);

    float res;    
    std::memcpy(&res, &raw_res, sizeof(res));
    return res;
}
