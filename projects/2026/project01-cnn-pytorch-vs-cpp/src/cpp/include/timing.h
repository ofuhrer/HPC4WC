#ifndef TIMING_H
#define TIMING_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#endif

typedef struct {
    uint64_t os_start_nsec;
    uint64_t os_stop_nsec;
} timer_context_t;

typedef struct {
    int fd;
} pmu_ctx_t;

/* ============================================================================
 * Compiler barrier
 * Prevents compiler reordering across timing boundaries.
 * ============================================================================ */
static inline void _compiler_barrier(void) {
    __asm__ volatile("" ::: "memory");
}

/* ============================================================================
 * OS timer
 * ============================================================================ */
static inline uint64_t _get_os_nsec(void) {
#ifdef __APPLE__
    // macOS High-resolution fallback using Mach Absolute Time
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    return (mach_absolute_time() * timebase.numer) / timebase.denom;
#else
    // Linux high-resolution timers
    struct timespec ts;
#if defined(CLOCK_MONOTONIC_RAW)
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
#endif
}


/* ============================================================================
 * Timer controls
 * ============================================================================ */
static inline void start_timer(timer_context_t* ctx) {
    _compiler_barrier();
    ctx->os_start_nsec = _get_os_nsec();
    _compiler_barrier();
}

static inline void stop_timer(timer_context_t* ctx) {
    _compiler_barrier();
    ctx->os_stop_nsec = _get_os_nsec();
    _compiler_barrier();
}


/* ============================================================================
 * Elapsed time calculators
 * ============================================================================ */
static inline double get_elapsed_os_sec(const timer_context_t* ctx) {
    return (double)(ctx->os_stop_nsec - ctx->os_start_nsec) / 1e9;
}

static inline int _cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static inline double compute_median(double *arr, int n) {
    qsort(arr, n, sizeof(double), _cmp_double);
    if (n % 2 == 0) {
        return (arr[n/2 - 1] + arr[n/2]) / 2.0;
    } else {
        return arr[n/2];
    }
}


/* =========================================================================== 
 * PMU (Performance Monitoring Unit) controls
 * ============================================================================
*/
#ifdef __linux__
static inline long _perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                              int cpu, int group_fd, unsigned long flags) {
    return syscall(SYS_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static inline int pmu_cycles_init(pmu_ctx_t *ctx) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type           = PERF_TYPE_HARDWARE;
    pe.size           = sizeof(pe);
    pe.config         = PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled       = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv     = 1;

    ctx->fd = _perf_event_open(&pe, 0 /* self */, -1 /* any cpu */, -1, 0);
    if (ctx->fd == -1) {
        fprintf(stderr, "perf_event_open failed: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

static inline void pmu_cycles_start(pmu_ctx_t *ctx) {
    ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(ctx->fd, PERF_EVENT_IOC_ENABLE, 0);
}

static inline uint64_t pmu_cycles_stop(pmu_ctx_t *ctx) {
    ioctl(ctx->fd, PERF_EVENT_IOC_DISABLE, 0);
    uint64_t val;
    if (read(ctx->fd, &val, sizeof(val)) == -1) {
        return 0;
    }
    return val;
}

static inline void pmu_cycles_close(pmu_ctx_t *ctx) {
    close(ctx->fd);
}

#else
// macOS fallback
static inline int pmu_cycles_init(pmu_ctx_t *ctx) {
    (void)ctx;
    return -1; 
}

static inline void pmu_cycles_start(pmu_ctx_t *ctx) {
    (void)ctx;
}

static inline uint64_t pmu_cycles_stop(pmu_ctx_t *ctx) {
    (void)ctx;
    return 0; 
}

static inline void pmu_cycles_close(pmu_ctx_t *ctx) {
    (void)ctx;
}
#endif

#endif
