/* Copyright (C) 2025 Davide Santangelo
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *  *  Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *
 *  *  Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file vsize.c
 * @brief Implementation of the vsize library for efficient array processing.
 */

#include "vsize.h" // Library header
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h> // POSIX threads
#include <string.h>
#include <stdbool.h> // For bool
#include <stdint.h>  // For uintptr_t

// --- Internal Configurations ---

// Define SIMD usage based on architecture
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h> // AVX/AVX2/AVX512 intrinsics
#include <cpuid.h>     // For __get_cpuid_max, __cpuid_count (GCC/Clang specific)
#define ARCH_X86 1
#define HAS_SIMD 1
#elif defined(__ARM_NEON)
#include <arm_neon.h> // NEON intrinsics
#define ARCH_ARM_NEON 1
#define HAS_SIMD 1
#else
#define ARCH_X86 0
#define ARCH_ARM_NEON 0
#define HAS_SIMD 0 // No recognized SIMD support
#endif

// Maximum number of threads for parallel computation
#define VSIZE_MAX_THREADS 8
// Common cache line size in bytes
#define VSIZE_CACHE_LINE_SIZE 64
// Minimum elements per thread to justify parallel overhead
#define MIN_ELEMENTS_PER_THREAD 1024
// Alignment requirement for AVX/AVX2 (256-bit = 32 bytes)
#define VSIZE_AVX_ALIGNMENT 32

// --- Internal Runtime CPU Feature Detection ---

// Static variables to cache CPU feature checks
#if ARCH_X86
static bool avx_supported = false;
static bool avx2_supported = false;
// static bool avx512f_supported = false; // Placeholder for AVX512
#endif
// static bool neon_supported = false;    // Placeholder for NEON (usually compile-time)
static bool features_checked = false;

#if ARCH_X86
// Helper function to check CPU features using CPUID (GCC/Clang specific)
static void vsize_internal_check_cpu_features_x86(void)
{
    if (features_checked)
        return;

    unsigned int eax, ebx, ecx, edx;
    unsigned int max_level = __get_cpuid_max(0, NULL);

    // Check for AVX (Function 1, ECX bit 28)
    if (max_level >= 1)
    {
        __cpuid(1, eax, ebx, ecx, edx);
        if (ecx & (1 << 28))
        {
            avx_supported = true;
        }
    }

    // Check for AVX2 (Function 7, Sub-leaf 0, EBX bit 5)
    if (max_level >= 7)
    {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        if (ebx & (1 << 5))
        {
            avx2_supported = true;
        }
        // Placeholder: Check for AVX512F (Function 7, Sub-leaf 0, EBX bit 16)
        // if (ebx & (1 << 16)) {
        //     avx512f_supported = true;
        // }
    }

    features_checked = true;
    // printf("[vsize debug] AVX: %d, AVX2: %d\n", avx_supported, avx2_supported);
}
#elif ARCH_ARM_NEON
// On ARM, NEON support is typically checked at compile time via __ARM_NEON
static void vsize_internal_check_cpu_features_arm(void)
{
    if (features_checked)
        return;
#ifdef __ARM_NEON
    // neon_supported = true; // If needed for runtime logic
#endif
    features_checked = true;
}
#else
// No specific runtime checks needed for architectures without known SIMD here
static void vsize_internal_check_cpu_features_other(void)
{
    features_checked = true;
}
#endif

// Unified function to trigger the check
static void vsize_internal_ensure_features_checked(void)
{
#if ARCH_X86
    vsize_internal_check_cpu_features_x86();
#elif ARCH_ARM_NEON
    vsize_internal_check_cpu_features_arm();
#else
    vsize_internal_check_cpu_features_other();
#endif
}

// --- Internal Data Structures (for Multithreading) ---

// Structure for int
typedef struct
{
    const int *array_start;
    size_t num_elements;
    int64_t partial_sum;
    int thread_id;
} VsizeThreadDataInt;

// Structure for float
typedef struct
{
    const float *array_start;
    size_t num_elements;
    double partial_sum; // Use double for intermediate sum precision
    int thread_id;
} VsizeThreadDataFloat;

// Structure for double
typedef struct
{
    const double *array_start;
    size_t num_elements;
    double partial_sum;
    int thread_id;
} VsizeThreadDataDouble;

// --- Internal Worker Functions (static) ---

// Worker for int
static void *process_sum_chunk_int(void *arg)
{
    VsizeThreadDataInt *data = (VsizeThreadDataInt *)arg;
    data->partial_sum = 0;
    for (size_t i = 0; i < data->num_elements; ++i)
    {
        data->partial_sum += data->array_start[i];
    }
    return NULL;
}

// Worker for float
static void *process_sum_chunk_float(void *arg)
{
    VsizeThreadDataFloat *data = (VsizeThreadDataFloat *)arg;
    data->partial_sum = 0.0;
    for (size_t i = 0; i < data->num_elements; ++i)
    {
        data->partial_sum += data->array_start[i];
    }
    return NULL;
}

// Worker for double
static void *process_sum_chunk_double(void *arg)
{
    VsizeThreadDataDouble *data = (VsizeThreadDataDouble *)arg;
    data->partial_sum = 0.0;
    for (size_t i = 0; i < data->num_elements; ++i)
    {
        data->partial_sum += data->array_start[i];
    }
    return NULL;
}

// --- Public Function Implementations ---

// --- Integer Functions ---

/**
 * @brief Parallel sum of an integer array using POSIX threads.
 */
int64_t vsize_parallel_sum_int(const int *array, size_t total_elements)
{
    if (!array || total_elements == 0)
        return 0;

    int num_threads = total_elements / MIN_ELEMENTS_PER_THREAD;
    if (num_threads > VSIZE_MAX_THREADS)
        num_threads = VSIZE_MAX_THREADS;
    if (num_threads < 1)
        num_threads = 1;

    if (num_threads == 1)
    {
        int64_t sum = 0;
        for (size_t i = 0; i < total_elements; ++i)
            sum += array[i];
        return sum;
    }

    pthread_t threads[VSIZE_MAX_THREADS];
    VsizeThreadDataInt thread_data[VSIZE_MAX_THREADS];
    size_t elements_per_thread = total_elements / num_threads;
    size_t remaining_elements = total_elements % num_threads;
    size_t current_offset = 0;
    int active_threads = 0;

    for (int i = 0; i < num_threads; ++i)
    {
        size_t chunk_size = elements_per_thread + ((size_t)i < remaining_elements ? 1 : 0);
        if (chunk_size == 0)
            continue;

        thread_data[i].array_start = array + current_offset;
        thread_data[i].num_elements = chunk_size;
        thread_data[i].partial_sum = 0;
        thread_data[i].thread_id = i;

        int rc = pthread_create(&threads[active_threads], NULL, process_sum_chunk_int, &thread_data[i]);
        if (rc)
        {
            fprintf(stderr, "Error [vsize]: Failed to create thread %d, code: %d\n", i, rc);
            thread_data[i].num_elements = 0; // Mark as failed
        }
        else
        {
            active_threads++;
        }
        current_offset += chunk_size;
    }

    if (active_threads == 0)
    {
        fprintf(stderr, "Warning [vsize]: No worker threads created. Falling back to sequential.\n");
        int64_t sum = 0;
        for (size_t i = 0; i < total_elements; ++i)
            sum += array[i];
        return sum;
    }

    for (int i = 0; i < active_threads; ++i)
    {
        pthread_join(threads[i], NULL); // Error handling omitted for brevity
    }

    int64_t total_sum = 0;
    for (int i = 0; i < num_threads; ++i)
    {
        if (thread_data[i].num_elements > 0)
        {
            total_sum += thread_data[i].partial_sum;
        }
    }
    return total_sum;
}

/**
 * @brief Sum of an integer array using AVX2 SIMD instructions if available.
 */
int64_t vsize_simd_sum_int(const int *array, size_t total_elements)
{
    if (!array || total_elements == 0)
        return 0;

    vsize_internal_ensure_features_checked(); // Ensure CPU features are checked

    int64_t total_sum = 0;

#if ARCH_X86
    if (avx2_supported && total_elements >= 8) // Check runtime support and size
    {
        size_t num_vectors = total_elements / 8;
        size_t start_index = num_vectors * 8;
        bool is_aligned = ((uintptr_t)array % VSIZE_AVX_ALIGNMENT) == 0;

        __m256i sum_vec_low64 = _mm256_setzero_si256();
        __m256i sum_vec_high64 = _mm256_setzero_si256();

        for (size_t i = 0; i < num_vectors; ++i)
        {
            // Load 8 integers (aligned or unaligned)
            __m256i data_vec;
            if (is_aligned)
            {
                data_vec = _mm256_load_si256((__m256i const *)(array + i * 8));
            }
            else
            {
                data_vec = _mm256_loadu_si256((__m256i const *)(array + i * 8));
            }

            __m128i data_low128 = _mm256_castsi256_si128(data_vec);
            __m128i data_high128 = _mm256_extracti128_si256(data_vec, 1);
            __m256i data_low64 = _mm256_cvtepi32_epi64(data_low128);
            __m256i data_high64 = _mm256_cvtepi32_epi64(data_high128);
            sum_vec_low64 = _mm256_add_epi64(sum_vec_low64, data_low64);
            sum_vec_high64 = _mm256_add_epi64(sum_vec_high64, data_high64);
        }

        int64_t sums64[4];
        _mm256_storeu_si256((__m256i *)sums64, sum_vec_low64);
        total_sum += sums64[0] + sums64[1] + sums64[2] + sums64[3];
        _mm256_storeu_si256((__m256i *)sums64, sum_vec_high64);
        total_sum += sums64[0] + sums64[1] + sums64[2] + sums64[3];

        // Process remaining elements
        for (size_t i = start_index; i < total_elements; ++i)
        {
            total_sum += array[i];
        }
        return total_sum; // Return AVX2 result
    }
    // Placeholder: Add SSE implementation here if AVX2 not supported but SSE is
    // else if (sse_supported && total_elements >= 4) { ... }

#elif ARCH_ARM_NEON
    // Placeholder: Add NEON implementation here
    // if (neon_supported && total_elements >= 4) { ... }
#endif

    // Fallback scalar implementation if no suitable SIMD available or array too small
    // printf("Note [vsize]: Using scalar loop for vsize_simd_sum_int.\n");
    for (size_t i = 0; i < total_elements; ++i)
    {
        total_sum += array[i];
    }
    return total_sum;
}

/**
 * @brief Sum of an integer array using cache-friendly sequential access.
 */
int64_t vsize_cache_friendly_sum_int(const int *array, size_t total_elements)
{
    if (!array || total_elements == 0)
        return 0;

    int64_t total_sum = 0;
    const size_t element_size = sizeof(int);
    size_t elements_per_cacheline = (VSIZE_CACHE_LINE_SIZE / element_size);
    if (elements_per_cacheline == 0)
        elements_per_cacheline = 1;

    for (size_t i = 0; i < total_elements; i += elements_per_cacheline)
    {
        size_t end_chunk = (i + elements_per_cacheline > total_elements) ? total_elements : i + elements_per_cacheline;
        for (size_t j = i; j < end_chunk; ++j)
        {
            total_sum += array[j];
        }
    }
    return total_sum;
}

// --- Float Functions ---

/**
 * @brief Parallel sum of a float array using POSIX threads.
 */
double vsize_parallel_sum_float(const float *array, size_t total_elements)
{
    if (!array || total_elements == 0)
        return 0.0;

    int num_threads = total_elements / MIN_ELEMENTS_PER_THREAD;
    if (num_threads > VSIZE_MAX_THREADS)
        num_threads = VSIZE_MAX_THREADS;
    if (num_threads < 1)
        num_threads = 1;

    if (num_threads == 1)
    {
        double sum = 0.0;
        for (size_t i = 0; i < total_elements; ++i)
            sum += array[i];
        return sum;
    }

    pthread_t threads[VSIZE_MAX_THREADS];
    VsizeThreadDataFloat thread_data[VSIZE_MAX_THREADS];
    size_t elements_per_thread = total_elements / num_threads;
    size_t remaining_elements = total_elements % num_threads;
    size_t current_offset = 0;
    int active_threads = 0;

    for (int i = 0; i < num_threads; ++i)
    {
        size_t chunk_size = elements_per_thread + ((size_t)i < remaining_elements ? 1 : 0);
        if (chunk_size == 0)
            continue;

        thread_data[i].array_start = array + current_offset;
        thread_data[i].num_elements = chunk_size;
        thread_data[i].partial_sum = 0.0;
        thread_data[i].thread_id = i;

        int rc = pthread_create(&threads[active_threads], NULL, process_sum_chunk_float, &thread_data[i]);
        if (rc)
        {
            fprintf(stderr, "Error [vsize]: Failed to create float thread %d, code: %d\n", i, rc);
            thread_data[i].num_elements = 0;
        }
        else
        {
            active_threads++;
        }
        current_offset += chunk_size;
    }

    if (active_threads == 0)
    {
        fprintf(stderr, "Warning [vsize]: No float worker threads created. Falling back to sequential.\n");
        double sum = 0.0;
        for (size_t i = 0; i < total_elements; ++i)
            sum += array[i];
        return sum;
    }

    for (int i = 0; i < active_threads; ++i)
    {
        pthread_join(threads[i], NULL);
    }

    double total_sum = 0.0;
    for (int i = 0; i < num_threads; ++i)
    {
        if (thread_data[i].num_elements > 0)
        {
            total_sum += thread_data[i].partial_sum;
        }
    }
    return total_sum;
}

/**
 * @brief Sum of a float array using AVX SIMD instructions if available.
 */
double vsize_simd_sum_float(const float *array, size_t total_elements)
{
    if (!array || total_elements == 0)
        return 0.0;

    vsize_internal_ensure_features_checked();

    double total_sum = 0.0; // Use double for final sum precision

#if ARCH_X86
    // AVX is sufficient for _mm256_add_ps
    if (avx_supported && total_elements >= 8)
    {
        size_t num_vectors = total_elements / 8;
        size_t start_index = num_vectors * 8;
        bool is_aligned = ((uintptr_t)array % VSIZE_AVX_ALIGNMENT) == 0;

        __m256 sum_vec = _mm256_setzero_ps(); // 8x float vector

        for (size_t i = 0; i < num_vectors; ++i)
        {
            __m256 data_vec;
            if (is_aligned)
            {
                data_vec = _mm256_load_ps(array + i * 8);
            }
            else
            {
                data_vec = _mm256_loadu_ps(array + i * 8);
            }
            sum_vec = _mm256_add_ps(sum_vec, data_vec);
        }

        // Horizontal sum for float vector
        float sums_float[8];
        _mm256_storeu_ps(sums_float, sum_vec); // Store unaligned is fine here
        for (int i = 0; i < 8; ++i)
        {
            total_sum += sums_float[i];
        }

        // Process remaining elements
        for (size_t i = start_index; i < total_elements; ++i)
        {
            total_sum += array[i];
        }
        return total_sum;
    }
    // Placeholder: Add SSE implementation for float here
    // else if (sse_supported && total_elements >= 4) { ... }

#elif ARCH_ARM_NEON
    // Placeholder: Add NEON implementation for float here
    // if (neon_supported && total_elements >= 4) { ... }
#endif

    // Fallback scalar implementation
    // printf("Note [vsize]: Using scalar loop for vsize_simd_sum_float.\n");
    for (size_t i = 0; i < total_elements; ++i)
    {
        total_sum += array[i];
    }
    return total_sum;
}

/**
 * @brief Sum of a float array using cache-friendly sequential access.
 */
double vsize_cache_friendly_sum_float(const float *array, size_t total_elements)
{
    if (!array || total_elements == 0)
        return 0.0;

    double total_sum = 0.0;
    const size_t element_size = sizeof(float);
    size_t elements_per_cacheline = (VSIZE_CACHE_LINE_SIZE / element_size);
    if (elements_per_cacheline == 0)
        elements_per_cacheline = 1;

    for (size_t i = 0; i < total_elements; i += elements_per_cacheline)
    {
        size_t end_chunk = (i + elements_per_cacheline > total_elements) ? total_elements : i + elements_per_cacheline;
        for (size_t j = i; j < end_chunk; ++j)
        {
            total_sum += array[j];
        }
    }
    return total_sum;
}

// --- Double Functions ---

/**
 * @brief Parallel sum of a double array using POSIX threads.
 */
double vsize_parallel_sum_double(const double *array, size_t total_elements)
{
    if (!array || total_elements == 0)
        return 0.0;

    int num_threads = total_elements / MIN_ELEMENTS_PER_THREAD;
    if (num_threads > VSIZE_MAX_THREADS)
        num_threads = VSIZE_MAX_THREADS;
    if (num_threads < 1)
        num_threads = 1;

    if (num_threads == 1)
    {
        double sum = 0.0;
        for (size_t i = 0; i < total_elements; ++i)
            sum += array[i];
        return sum;
    }

    pthread_t threads[VSIZE_MAX_THREADS];
    VsizeThreadDataDouble thread_data[VSIZE_MAX_THREADS];
    size_t elements_per_thread = total_elements / num_threads;
    size_t remaining_elements = total_elements % num_threads;
    size_t current_offset = 0;
    int active_threads = 0;

    for (int i = 0; i < num_threads; ++i)
    {
        size_t chunk_size = elements_per_thread + ((size_t)i < remaining_elements ? 1 : 0);
        if (chunk_size == 0)
            continue;

        thread_data[i].array_start = array + current_offset;
        thread_data[i].num_elements = chunk_size;
        thread_data[i].partial_sum = 0.0;
        thread_data[i].thread_id = i;

        int rc = pthread_create(&threads[active_threads], NULL, process_sum_chunk_double, &thread_data[i]);
        if (rc)
        {
            fprintf(stderr, "Error [vsize]: Failed to create double thread %d, code: %d\n", i, rc);
            thread_data[i].num_elements = 0;
        }
        else
        {
            active_threads++;
        }
        current_offset += chunk_size;
    }

    if (active_threads == 0)
    {
        fprintf(stderr, "Warning [vsize]: No double worker threads created. Falling back to sequential.\n");
        double sum = 0.0;
        for (size_t i = 0; i < total_elements; ++i)
            sum += array[i];
        return sum;
    }

    for (int i = 0; i < active_threads; ++i)
    {
        pthread_join(threads[i], NULL);
    }

    double total_sum = 0.0;
    for (int i = 0; i < num_threads; ++i)
    {
        if (thread_data[i].num_elements > 0)
        {
            total_sum += thread_data[i].partial_sum;
        }
    }
    return total_sum;
}

/**
 * @brief Sum of a double array using AVX SIMD instructions if available.
 */
double vsize_simd_sum_double(const double *array, size_t total_elements)
{
    if (!array || total_elements == 0)
        return 0.0;

    vsize_internal_ensure_features_checked();

    double total_sum = 0.0;

#if ARCH_X86
    // AVX is sufficient for _mm256_add_pd
    if (avx_supported && total_elements >= 4) // AVX processes 4 doubles
    {
        size_t num_vectors = total_elements / 4;
        size_t start_index = num_vectors * 4;
        bool is_aligned = ((uintptr_t)array % VSIZE_AVX_ALIGNMENT) == 0;

        __m256d sum_vec = _mm256_setzero_pd(); // 4x double vector

        for (size_t i = 0; i < num_vectors; ++i)
        {
            __m256d data_vec;
            if (is_aligned)
            {
                data_vec = _mm256_load_pd(array + i * 4);
            }
            else
            {
                data_vec = _mm256_loadu_pd(array + i * 4);
            }
            sum_vec = _mm256_add_pd(sum_vec, data_vec);
        }

        // Horizontal sum for double vector
        double sums_double[4];
        _mm256_storeu_pd(sums_double, sum_vec);
        total_sum += sums_double[0] + sums_double[1] + sums_double[2] + sums_double[3];

        // Process remaining elements
        for (size_t i = start_index; i < total_elements; ++i)
        {
            total_sum += array[i];
        }
        return total_sum;
    }
    // Placeholder: Add SSE2 implementation for double here (_mm_add_pd)
    // else if (sse2_supported && total_elements >= 2) { ... }

#elif ARCH_ARM_NEON
    // Placeholder: Add NEON implementation for double here (if double precision supported)
    // if (neon_fp64_supported && total_elements >= 2) { ... }
#endif

    // Fallback scalar implementation
    // printf("Note [vsize]: Using scalar loop for vsize_simd_sum_double.\n");
    for (size_t i = 0; i < total_elements; ++i)
    {
        total_sum += array[i];
    }
    return total_sum;
}

/**
 * @brief Sum of a double array using cache-friendly sequential access.
 */
double vsize_cache_friendly_sum_double(const double *array, size_t total_elements)
{
    if (!array || total_elements == 0)
        return 0.0;

    double total_sum = 0.0;
    const size_t element_size = sizeof(double);
    size_t elements_per_cacheline = (VSIZE_CACHE_LINE_SIZE / element_size);
    if (elements_per_cacheline == 0)
        elements_per_cacheline = 1;

    for (size_t i = 0; i < total_elements; i += elements_per_cacheline)
    {
        size_t end_chunk = (i + elements_per_cacheline > total_elements) ? total_elements : i + elements_per_cacheline;
        for (size_t j = i; j < end_chunk; ++j)
        {
            total_sum += array[j];
        }
    }
    return total_sum;
}
