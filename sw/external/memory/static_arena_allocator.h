#pragma once

#include <heepstor_assert.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

// TODO: In the future, maybe statically arrange the buffers.
// TODO: Study whether this is a better approach than just using malloc.
class StaticArenaAllocator {
private:
    static constexpr size_t ARENA_SIZE = 256 * 1024;  // 256 KiB
    alignas(std::max_align_t) static uint8_t buffer[ARENA_SIZE];
    static size_t currentOffset;

public:
    template <typename T>
    static T* allocate_array(size_t count, size_t alignment = alignof(std::max_align_t)) {
        size_t size = count * sizeof(T);
        size_t padding = (alignment - (currentOffset & (alignment - 1))) & (alignment - 1);

        // Ensure that we are not out of memory
        HEEPSTOR_ASSERT((currentOffset + padding + size < ARENA_SIZE) && "ERROR: Out of memory!");

        T* ptr = reinterpret_cast<T*>(buffer + currentOffset + padding);
        currentOffset += size + padding;
        return ptr;
    }

    static void reset() { currentOffset = 0; }

    static void* allocate(size_t size, size_t alignment = alignof(std::max_align_t)) {
        size_t padding = (alignment - (currentOffset & (alignment - 1))) & (alignment - 1);

        // Ensure that we are not out of memory
        HEEPSTOR_ASSERT((currentOffset + padding + size < ARENA_SIZE) && "ERROR: Out of memory!");

        void* ptr = buffer + currentOffset + padding;
        currentOffset += size + padding;
        return ptr;
    }

    template <typename T, typename... Args>
    static T* create(Args&&... args) {
        void* ptr = allocate(sizeof(T), alignof(T));
        return new (ptr) T(std::forward<Args>(args)...);
    }

    template <typename T>
    static void destroy(T* ptr) {
        if (ptr) {
            ptr->~T();
            // Memory is not freed until reset()
        }
    }

    static size_t available_bytes() { return ARENA_SIZE - currentOffset; }

    static size_t used_bytes() { return currentOffset; }
};