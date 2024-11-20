#include <cstdint>

// Based on Xoshiro128+ (https://prng.di.unimi.it/)
class RandomNumberGenerator {
private:
    // Xoshiro128+ state (four 32-bit integers)
    uint32_t state[4];

    // Helper function for bit rotation
    static inline uint32_t rotl(const uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }

    // Generate next random value using Xoshiro128+
    uint32_t next_int() {
        const uint32_t result = state[0] + state[3];
        const uint32_t t = state[1] << 9;

        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];

        state[2] ^= t;
        state[3] = rotl(state[3], 11);

        return result;
    }

    // Fast implementation of modulo for powers of 2
    static uint32_t fast_mod2(uint32_t value, uint32_t ceiling) { return value & (ceiling - 1); }

    // Check if number is power of 2
    static bool is_power_of_2(uint32_t x) { return x && !(x & (x - 1)); }

    // Initialize the state using splitmix64 algorithm
    void init_state(uint64_t seed) {
        uint64_t z = seed;
        for (int i = 0; i < 4; i++) {
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z = z ^ (z >> 31);
            state[i] = static_cast<uint32_t>(z);
        }
    }

public:
    explicit RandomNumberGenerator(uint64_t initial_seed = 5489ULL) { set_seed(initial_seed); }

    uint32_t rand_int() { return next_int(); }

    // Generate random integer in range [min, max]
    int32_t rand_int_range(int32_t min, int32_t max) {
        uint32_t range = static_cast<uint32_t>(max - min + 1);
        uint32_t result;

        if (is_power_of_2(range)) {
            result = fast_mod2(next_int(), range);
        } else {
            // Reject values that would create modulo bias
            uint32_t limit = (UINT32_MAX - range + 1) % range;
            do {
                result = next_int();
            } while (result < limit);
            result %= range;
        }

        return static_cast<int32_t>(result) + min;
    }

    // Generate random float in range [0, 1)
    float rand_float_uniform() {
        // Use 23 bits for mantissa to match float precision
        return (next_int() >> 9) * 0x1.0p-23f;
    }

    // Generate random float in range [min, max)
    float rand_float_uniform_range(float min, float max) { return min + rand_float_uniform() * (max - min); }

    // Reseed the generator
    void set_seed(uint64_t new_seed) { init_state(new_seed); }
};