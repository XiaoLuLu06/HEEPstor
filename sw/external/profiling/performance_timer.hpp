#pragma once

#include <cstdint>
#include <cstdio>
#include "core_v_mini_mcu.h"
#include "rv_timer.h"
#include "rv_timer_regs.h"
#include "soc_ctrl.h"

enum class CheckpointPerformanceTimerDisplayConfig {
    None,  // Don't print anything
    Microseconds,
    Milliseconds,
    Seconds
};

template <size_t N>
class CheckpointPerformanceTimer {
private:
    using TimePoint = uint64_t;  // Number of cycles
    using DisplayConfig = CheckpointPerformanceTimerDisplayConfig;

    DisplayConfig display_config;

    TimePoint start_time;
    TimePoint last_checkpoint;
    TimePoint durations[N];

    size_t current_checkpoint_idx;
    uint32_t freq_hz;
    rv_timer_t timer;

    TimePoint get_current_time() {
        uint64_t cycle_count;
        rv_timer_result_t result = rv_timer_counter_read(&timer, 0, &cycle_count);
        HEEPSTOR_ASSERT(result == kRvTimerOk && "Timer read failed");
        return cycle_count;
    }

    uint64_t compute_duration_us(TimePoint end, TimePoint start) {
        // Convert from number of cycles to us using frequency
        uint64_t tick_diff = end - start;
        return (tick_diff * 1000000ULL) / freq_hz;
    }

    void print_duration(uint64_t duration_us) {
        if (display_config == DisplayConfig::None)
            return;

        switch (display_config) {
            case DisplayConfig::Microseconds: {
                printf("%lu µs\n", duration_us);
                break;
            }
            case DisplayConfig::Milliseconds: {
                printf("%lu ms\n", duration_us / 1000ULL);
                break;
            }
            case DisplayConfig::Seconds: {
                printFloat(static_cast<float>(duration_us) / 1000000.0f);
                printf(" s\n");
                break;
            }
            default: {
                HEEPSTOR_ASSERT(false && "Not implemented!");
                break;
            }
        }
    }

public:
    explicit CheckpointPerformanceTimer(DisplayConfig config = DisplayConfig::Microseconds) : display_config(config), current_checkpoint_idx(0) {
        // Get system frequency
        soc_ctrl_t soc_ctrl;
        soc_ctrl.base_addr = mmio_region_from_addr((uintptr_t)SOC_CTRL_START_ADDRESS);
        freq_hz = soc_ctrl_get_frequency(&soc_ctrl);

        // Initialize timer
        rv_timer_config_t timer_cfg = {
            .hart_count = RV_TIMER_PARAM_N_HARTS,
            .comparator_count = RV_TIMER_PARAM_N_TIMERS,
        };
        mmio_region_t timer_base = {
            .base = (void*)RV_TIMER_AO_START_ADDRESS,
        };

        rv_timer_result_t result = rv_timer_init(timer_base, timer_cfg, &timer);
        HEEPSTOR_ASSERT(result == kRvTimerOk && "Timer initialization failed");

        rv_timer_tick_params_t tick_params;

        rv_timer_approximate_tick_params_result_t result_tick_params = rv_timer_approximate_tick_params(freq_hz, freq_hz, &tick_params);
        HEEPSTOR_ASSERT(result_tick_params == kRvTimerOk && "Timer tick params calculation failed");

        result = rv_timer_set_tick_params(&timer, 0, tick_params);
        HEEPSTOR_ASSERT(result == kRvTimerOk && "Timer tick params set failed");

        result = rv_timer_counter_set_enabled(&timer, 0, kRvTimerEnabled);
        HEEPSTOR_ASSERT(result == kRvTimerOk && "Timer enable failed");

        start_time = get_current_time();
        last_checkpoint = start_time;

        for (size_t i = 0; i < N; ++i) {
            durations[i] = 0;
        }
    }

    void reset() {
        rv_timer_result_t result = rv_timer_reset(&timer);
        HEEPSTOR_ASSERT(result == kRvTimerOk && "Timer reset failed");

        result = rv_timer_counter_set_enabled(&timer, 0, kRvTimerEnabled);
        HEEPSTOR_ASSERT(result == kRvTimerOk && "Timer enable failed");

        for (size_t i = 0; i < N; ++i) {
            durations[i] = 0;
        }
        current_checkpoint_idx = 0;
        start_time = get_current_time();
        last_checkpoint = start_time;
    }

    void checkpoint() {
        HEEPSTOR_ASSERT(current_checkpoint_idx < N && "Too many checkpoints");

        TimePoint now = get_current_time();
        durations[current_checkpoint_idx] = now - last_checkpoint;
        current_checkpoint_idx++;
        last_checkpoint = now;
    }

    void finalize(std::initializer_list<const char*> labels) {
        HEEPSTOR_ASSERT(labels.size() == N && "Number of labels must match number of checkpoints");
        HEEPSTOR_ASSERT(current_checkpoint_idx == N && "Not all checkpoints were recorded");

        if (display_config == DisplayConfig::None)
            return;

        TimePoint now = get_current_time();
        uint64_t total = compute_duration_us(now, start_time);

        printf("\nPerformance Summary:\n");
        printf("-------------------------------\n");

        auto label_it = labels.begin();
        for (size_t i = 0; i < N; ++i, ++label_it) {
            uint64_t duration_us = compute_duration_us(start_time + durations[i], start_time);
            printf("%s: ", *label_it);
            fflush(stdout);
            print_duration(duration_us);
        }

        printf("-------------------------------\n");
        printf("Total time: ");
        fflush(stdout);

        print_duration(total);
        printf("\n");
    }

    uint64_t elapsed_microseconds() const { return compute_duration_us(get_current_time(), start_time); }

    uint64_t checkpoint_duration(size_t index) const {
        HEEPSTOR_ASSERT(index < N && "Checkpoint index out of bounds");
        return compute_duration_us(start_time + durations[index], start_time);
    }

    uint64_t checkpoint_ticks(size_t index) const {
        HEEPSTOR_ASSERT(index < N && "Checkpoint index out of bounds");
        return durations[index];
    }
};