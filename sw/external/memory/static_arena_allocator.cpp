#include "static_arena_allocator.h"

uint8_t StaticArenaAllocator::buffer[StaticArenaAllocator::ARENA_SIZE];
size_t StaticArenaAllocator::currentOffset = 0;