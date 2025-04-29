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
 * @file vsum.c
 * @brief Implementation of the vsum library for efficient array processing.
 */

#include "vsum.h" // Library header
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h> // POSIX threads
#include <string.h>
#include <stdbool.h> // For bool
#include <stdint.h>  // For uintptr_t
#include <unistd.h>  // For sysconf

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

// Maximum number of threads for parallel computation (used as a cap)
#define VSUM_MAX_ALLOWED_THREADS 16
// Common cache line size in bytes
#define VSUM_CACHE_LINE_SIZE 64
// Minimum elements per thread to justify parallel overhead
#define MIN_ELEMENTS_PER_THREAD 1024
// Alignment requirement for AVX/AVX2 (256-bit = 32 bytes)
#define VSUM_AVX_ALIGNMENT 32

// --- Internal Runtime CPU Feature Detection ---

// Static variables to cache CPU feature checks
#if ARCH_X86
static bool sse_supported = false;
static bool sse2_supported = false;
static bool sse3_supported = false; // Needed for hadd
static bool avx_supported = false;
static bool avx2_supported = false;
// static bool avx512f_supported = false; // Placeholder for AVX512
#endif
// static bool neon_supported = false;    // Placeholder for NEON (usually compile-time)
static bool features_checked = false;

#if ARCH_X86
// Helper function to check CPU features using CPUID (GCC/Clang specific)
static void vsum_internal_check_cpu_features_x86(void)
{
    if (features_checked)
        return;

    unsigned int eax, ebx, ecx, edx;
    unsigned int max_level = __get_cpuid_max(0, NULL);

    // Check for SSE, SSE2, SSE3, AVX (Function 1)
    if (max_level >= 1)
    {
        __cpuid(1, eax, ebx, ecx, edx);
        if (edx & (1 << 25))
            sse_supported = true; // SSE check (EDX bit 25)
        if (edx & (1 << 26))
            sse2_supported = true; // SSE2 check (EDX bit 26)
        if (ecx & (1 << 0))
            sse3_supported = true; // SSE3 check (ECX bit 0) - for hadd
        if (ecx & (1 << 28))
            avx_supported = true; // AVX check (ECX bit 28)
    }

    // Check for AVX2 (Function 7, Sub-leaf 0)
    if (max_level >= 7)
    {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        if (ebx & (1 << 5)) // AVX2 check (EBX bit 5)
        {
            avx2_supported = true;
        }
        // Placeholder: Check for AVX512F (Function 7, Sub-leaf 0, EBX bit 16)
        // if (ebx & (1 << 16)) {
        //     avx512f_supported = true;
        // }
    }

    features_checked = true;
#ifdef VSUM_DEBUG
    fprintf(stderr, "[vsum debug] CPU Features: SSE=%d, SSE2=%d, SSE3=%d, AVX=%d, AVX2=%d\n",
            sse_supported, sse2_supported, sse3_supported, avx_supported, avx2_supported);
#endif
}
#elif ARCH_ARM_NEON
// On ARM, NEON support is typically checked at compile time via __ARM_NEON
static void vsum_internal_check_cpu_features_arm(void)
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
static void vsum_internal_check_cpu_features_other(void)
{
    features_checked = true;
}
#endif

// Unified function to trigger the check
static void vsum_internal_ensure_features_checked(void)
{
#if ARCH_X86
    vsum_internal_check_cpu_features_x86();
#elif ARCH_ARM_NEON
    vsum_internal_check_cpu_features_arm();
#else
    vsum_internal_check_cpu_features_other();
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
} VsumThreadDataInt;

// Structure for float
typedef struct
{
    const float *array_start;
    size_t num_elements;
    double partial_sum; // Use double for intermediate sum precision
    int thread_id;
} VsumThreadDataFloat;

// Structure for double
typedef struct
{
    const double *array_start;
    size_t num_elements;
    double partial_sum;
    int thread_id;
} VsumThreadDataDouble;

// --- Internal Worker Functions (static) ---

// Worker for int
static void *process_sum_chunk_int(void *arg)
{
    VsumThreadDataInt *data = (VsumThreadDataInt *)arg;
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
    VsumThreadDataFloat *data = (VsumThreadDataFloat *)arg;
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
    VsumThreadDataDouble *data = (VsumThreadDataDouble *)arg;
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
int64_t vsum_parallel_sum_int(const int *restrict array, size_t total_elements)
{
    if (!array || total_elements == 0)
        return 0;

    // Determine number of threads based on cores, capped
    long core_count = sysconf(_SC_NPROCESSORS_ONLN);
    int max_threads = (core_count > 0) ? (int)core_count : 1;
    if (max_threads > VSUM_MAX_ALLOWED_THREADS)
    {
        max_threads = VSUM_MAX_ALLOWED_THREADS;
    }

    int num_threads = total_elements / MIN_ELEMENTS_PER_THREAD;
    if (num_threads > max_threads)
        num_threads = max_threads;
    if (num_threads < 1)
        num_threads = 1;

    if (num_threads == 1)
    {
        // Fallback to simple sequential sum for single thread case
        return vsum_cache_friendly_sum_int(array, total_elements);
    }

    pthread_t threads[VSUM_MAX_ALLOWED_THREADS]; // Use cap for array size
    VsumThreadDataInt thread_data[VSUM_MAX_ALLOWED_THREADS];
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
            fprintf(stderr, "Error [vsum]: Failed to create thread %d, code: %d\n", i, rc);
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
        fprintf(stderr, "Warning [vsum]: No worker threads created. Falling back to sequential.\n");
        return vsum_cache_friendly_sum_int(array, total_elements); // Or a direct loop
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
 * @brief Sum of an integer array using AVX2 SIMD instructions if available, falling back to SSE2.
 */
int64_t vsum_simd_sum_int(const int *restrict array, size_t total_elements)
{
    if (!array || total_elements == 0)
        return 0;

    vsum_internal_ensure_features_checked(); // Ensure CPU features are checked

    int64_t total_sum = 0;
    size_t i = 0; // Index for loops

#if ARCH_X86
    if (avx2_supported && total_elements >= 8) // Use AVX2 if available (8 ints)
    {
        size_t num_vectors = total_elements / 8;
        size_t end_index = num_vectors * 8;
        bool use_aligned_loads = vsum_internal_is_aligned(array, VSUM_AVX_ALIGNMENT); // Use the helper function

        __m256i sum_vec_low64 = _mm256_setzero_si256();
        __m256i sum_vec_high64 = _mm256_setzero_si256();

        if (use_aligned_loads)
        {
            for (i = 0; i < end_index; i += 8)
            {
                __m256i data_vec = _mm256_load_si256((__m256i const *)(array + i));
                // Convert 32-bit ints to 64-bit ints and add
                __m128i data_low128 = _mm256_castsi256_si128(data_vec);
                __m128i data_high128 = _mm256_extracti128_si256(data_vec, 1);
                __m256i data_low64 = _mm256_cvtepi32_epi64(data_low128);
                __m256i data_high64 = _mm256_cvtepi32_epi64(data_high128);
                sum_vec_low64 = _mm256_add_epi64(sum_vec_low64, data_low64);
                sum_vec_high64 = _mm256_add_epi64(sum_vec_high64, data_high64);
            }
        }
        else
        {
            for (i = 0; i < end_index; i += 8)
            {

                __m256i data_vec = _mm256_loadu_si256((__m256i const *)(array + i));
                // Convert 32-bit ints to 64-bit ints and add
                __m128i data_low128 = _mm256_castsi256_si128(data_vec);
                __m128i data_high128 = _mm256_extracti128_si256(data_vec, 1);
                __m256i data_low64 = _mm256_cvtepi32_epi64(data_low128);
                __m256i data_high64 = _mm256_cvtepi32_epi64(data_high128);
                sum_vec_low64 = _mm256_add_epi64(sum_vec_low64, data_low64);
                sum_vec_high64 = _mm256_add_epi64(sum_vec_high64, data_high64);
            }
        }

        // Horizontal sum (optimized - store and sum scalar)
        __m256i total_vec = _mm256_add_epi64(sum_vec_low64, sum_vec_high64);
        int64_t sums64[4]; // Aligned allocation not strictly necessary for storeu
        _mm256_storeu_si256((__m256i *)sums64, total_vec);
        total_sum += sums64[0] + sums64[1] + sums64[2] + sums64[3];

        // i is already at end_index for the remainder loop
    }
    else if (sse2_supported && total_elements >= 4) // Fallback to SSE2 (4 ints)
    {
        // SSE2 needs 16-byte alignment, but loadu is safer/simpler
        size_t num_vectors = total_elements / 4;
        size_t end_index = num_vectors * 4;

        __m128i sum_vec_low64 = _mm_setzero_si128();
        __m128i sum_vec_high64 = _mm_setzero_si128();

        for (i = 0; i < end_index; i += 4)
        {
            __m128i data_vec = _mm_loadu_si128((__m128i const *)(array + i));

            // Convert lower 2 ints to 64-bit
            __m128i data_low_half = data_vec; // Lower 64 bits contain the first 2 ints
            __m128i data_low64 = _mm_cvtepi32_epi64(data_low_half);
            sum_vec_low64 = _mm_add_epi64(sum_vec_low64, data_low64);

            // Convert upper 2 ints to 64-bit
            __m128i data_high_half = _mm_shuffle_epi32(data_vec, _MM_SHUFFLE(3, 2, 3, 2)); // Move upper 2 ints to lower half
            __m128i data_high64 = _mm_cvtepi32_epi64(data_high_half);
            sum_vec_high64 = _mm_add_epi64(sum_vec_high64, data_high64);
        }

        // Horizontal sum
        int64_t sums64[2];
        _mm_storeu_si128((__m128i *)sums64, _mm_add_epi64(sum_vec_low64, sum_vec_high64));
        total_sum += sums64[0] + sums64[1];

        // i is already at end_index for the remainder loop
    }
#elif ARCH_ARM_NEON
    // Placeholder: Add NEON implementation here
    // if (neon_supported && total_elements >= 4) { ... i = end_index; }
#endif

    // Process remaining elements sequentially
    for (; i < total_elements; ++i)
    {
        total_sum += array[i];
    }
    return total_sum;
}

/**
 * @brief Sum of an integer array using cache-friendly sequential access.
 */
int64_t vsum_cache_friendly_sum_int(const int *restrict array, size_t total_elements)
{
    if (!array || total_elements == 0)
        return 0;

    int64_t total_sum = 0;
    const size_t element_size = sizeof(int);
    size_t elements_per_cacheline = (VSUM_CACHE_LINE_SIZE / element_size);
    if (elements_per_cacheline == 0)
        elements_per_cacheline = 1;

    for (size_t i = 0; i < total_elements; i += elements_per_cacheline)
    {
        size_t end_chunk = (i + elements_per_cacheline > total_elements) ? total_elements : i + elements_per_cacheline;
        // Optimized inner loop (potential for compiler unrolling)
        int64_t chunk_sum = 0;
        for (size_t j = i; j < end_chunk; ++j)
        {
            chunk_sum += array[j];
        }
        total_sum += chunk_sum;
    }
    return total_sum;
}

// --- Float Functions ---

/**
 * @brief Parallel sum of a float array using POSIX threads.
 */
double vsum_parallel_sum_float(const float *restrict array, size_t total_elements)
{
    if (!array || total_elements == 0)
        return 0.0;

    // Determine number of threads based on cores, capped
    long core_count = sysconf(_SC_NPROCESSORS_ONLN);
    int max_threads = (core_count > 0) ? (int)core_count : 1;
    if (max_threads > VSUM_MAX_ALLOWED_THREADS)
    {
        max_threads = VSUM_MAX_ALLOWED_THREADS;
    }

    int num_threads = total_elements / MIN_ELEMENTS_PER_THREAD;
    if (num_threads > max_threads)
        num_threads = max_threads;
    if (num_threads < 1)
        num_threads = 1;

    if (num_threads == 1)
    {
        return vsum_cache_friendly_sum_float(array, total_elements); // Fallback
    }

    pthread_t threads[VSUM_MAX_ALLOWED_THREADS];
    VsumThreadDataFloat thread_data[VSUM_MAX_ALLOWED_THREADS];
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
            fprintf(stderr, "Error [vsum]: Failed to create float thread %d, code: %d\n", i, rc);
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
        fprintf(stderr, "Warning [vsum]: No float worker threads created. Falling back to sequential.\n");
        return vsum_cache_friendly_sum_float(array, total_elements); // Fallback
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
 * @brief Sum of a float array using AVX SIMD instructions if available, falling back to SSE.
 */
double vsum_simd_sum_float(const float *restrict array, size_t total_elements)
{
    if (!array || total_elements == 0)
        return 0.0;

    vsum_internal_ensure_features_checked();

    double total_sum = 0.0; // Use double for final sum precision
    size_t i = 0;

#if ARCH_X86
    if (avx_supported && total_elements >= 8) // Use AVX (8 floats)
    {
        size_t num_vectors = total_elements / 8;
        size_t end_index = num_vectors * 8;
        bool use_aligned_loads = vsum_internal_is_aligned(array, VSUM_AVX_ALIGNMENT); // Use the helper function

        __m256 sum_vec = _mm256_setzero_ps(); // 8x float vector

        if (use_aligned_loads)
        {
            for (i = 0; i < end_index; i += 8)
            {
                // No manual check needed here
                __m256 data_vec = _mm256_load_ps(array + i);
                sum_vec = _mm256_add_ps(sum_vec, data_vec);
            }
        }
        else
        {
            for (i = 0; i < end_index; i += 8)
            {
                // No manual check needed here
                __m256 data_vec = _mm256_loadu_ps(array + i);
                sum_vec = _mm256_add_ps(sum_vec, data_vec);
            }
        }

        // Horizontal sum for float vector (optimized AVX + SSE3)
        __m128 sum_low128 = _mm256_castps256_ps128(sum_vec);
        __m128 sum_high128 = _mm256_extractf128_ps(sum_vec, 1); // AVX instruction
        __m128 final_sum128 = _mm_add_ps(sum_low128, sum_high128);

#if defined(__SSE3__)
        if (sse3_supported)
        { // Runtime check might be redundant if compiler defines __SSE3__
            final_sum128 = _mm_hadd_ps(final_sum128, final_sum128);
            final_sum128 = _mm_hadd_ps(final_sum128, final_sum128);
            total_sum += _mm_cvtss_f32(final_sum128);
        }
        else
#endif
        { // Fallback if no SSE3 hadd_ps available (store and sum)
            float sums_float[4];
            _mm_storeu_ps(sums_float, final_sum128);
            total_sum += (double)sums_float[0] + (double)sums_float[1] + (double)sums_float[2] + (double)sums_float[3];
        }
        // i is already at end_index
    }
    else if (sse_supported && total_elements >= 4) // Fallback to SSE (4 floats)
    {
        // SSE path remains unchanged (already uses loadu)
        size_t num_vectors = total_elements / 4;
        size_t end_index = num_vectors * 4;
        __m128 sum_vec = _mm_setzero_ps(); // 4x float vector

        for (i = 0; i < end_index; i += 4)
        {
            __m128 data_vec = _mm_loadu_ps(array + i); // Use unaligned load
            sum_vec = _mm_add_ps(sum_vec, data_vec);
        }

        // Horizontal sum using SSE3 hadd_ps if available
#if defined(__SSE3__)
        if (sse3_supported)
        {
            sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
            sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
            total_sum += _mm_cvtss_f32(sum_vec);
        }
        else
#endif
        { // Fallback horizontal sum if no SSE3 intrinsics (store and sum)
            float sums_float[4];
            _mm_storeu_ps(sums_float, sum_vec);
            total_sum += (double)sums_float[0] + (double)sums_float[1] + (double)sums_float[2] + (double)sums_float[3];
        }
        // i is already at end_index
    }

#elif ARCH_ARM_NEON
    // Placeholder: Add NEON implementation for float here
    // if (neon_supported && total_elements >= 4) { ... i = end_index; }
#endif

    // Fallback scalar implementation for remainder or if no SIMD
    for (; i < total_elements; ++i)
    {
        total_sum += array[i];
    }
    return total_sum;
}

/**
 * @brief Sum of a float array using cache-friendly sequential access.
 */
double vsum_cache_friendly_sum_float(const float *restrict array, size_t total_elements)
{
    if (!array || total_elements == 0)
        return 0.0;

    double total_sum = 0.0;
    const size_t element_size = sizeof(float);
    size_t elements_per_cacheline = (VSUM_CACHE_LINE_SIZE / element_size);
    if (elements_per_cacheline == 0)
        elements_per_cacheline = 1;

    for (size_t i = 0; i < total_elements; i += elements_per_cacheline)
    {
        size_t end_chunk = (i + elements_per_cacheline > total_elements) ? total_elements : i + elements_per_cacheline;
        double chunk_sum = 0.0; // Use double for chunk sum precision
        for (size_t j = i; j < end_chunk; ++j)
        {
            chunk_sum += array[j];
        }
        total_sum += chunk_sum;
    }
    return total_sum;
}

// --- Double Functions ---

/**
 * @brief Parallel sum of a double array using POSIX threads.
 */
double vsum_parallel_sum_double(const double *restrict array, size_t total_elements)
{
    if (!array || total_elements == 0)
        return 0.0;

    // Determine number of threads based on cores, capped
    long core_count = sysconf(_SC_NPROCESSORS_ONLN);
    int max_threads = (core_count > 0) ? (int)core_count : 1;
    if (max_threads > VSUM_MAX_ALLOWED_THREADS)
    {
        max_threads = VSUM_MAX_ALLOWED_THREADS;
    }

    int num_threads = total_elements / MIN_ELEMENTS_PER_THREAD;
    if (num_threads > max_threads)
        num_threads = max_threads;
    if (num_threads < 1)
        num_threads = 1;

    if (num_threads == 1)
    {
        return vsum_cache_friendly_sum_double(array, total_elements); // Fallback
    }

    pthread_t threads[VSUM_MAX_ALLOWED_THREADS];
    VsumThreadDataDouble thread_data[VSUM_MAX_ALLOWED_THREADS];
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
            fprintf(stderr, "Error [vsum]: Failed to create double thread %d, code: %d\n", i, rc);
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
        fprintf(stderr, "Warning [vsum]: No double worker threads created. Falling back to sequential.\n");
        return vsum_cache_friendly_sum_double(array, total_elements); // Fallback
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
 * @brief Sum of a double array using AVX SIMD instructions if available, falling back to SSE2.
 */
double vsum_simd_sum_double(const double *restrict array, size_t total_elements)
{
    if (!array || total_elements == 0)
        return 0.0;

    vsum_internal_ensure_features_checked();

    double total_sum = 0.0;
    size_t i = 0;

#if ARCH_X86
    if (avx_supported && total_elements >= 4) // Use AVX (4 doubles)
    {
        size_t num_vectors = total_elements / 4;
        size_t end_index = num_vectors * 4;
        bool use_aligned_loads = vsum_internal_is_aligned(array, VSUM_AVX_ALIGNMENT); // Use the helper function

        __m256d sum_vec = _mm256_setzero_pd(); // 4x double vector

        if (use_aligned_loads)
        {
            for (i = 0; i < end_index; i += 4)
            {
                // No manual check needed here
                __m256d data_vec = _mm256_load_pd(array + i);
                sum_vec = _mm256_add_pd(sum_vec, data_vec);
            }
        }
        else
        {
            for (i = 0; i < end_index; i += 4)
            {
                // No manual check needed here
                __m256d data_vec = _mm256_loadu_pd(array + i);
                sum_vec = _mm256_add_pd(sum_vec, data_vec);
            }
        }

        // Horizontal sum for double vector (optimized AVX + SSE3)
        __m128d sum_low128 = _mm256_castpd256_pd128(sum_vec);
        __m128d sum_high128 = _mm256_extractf128_pd(sum_vec, 1);    // AVX instruction
        __m128d final_sum128 = _mm_add_pd(sum_low128, sum_high128); // [d1+d3, d0+d2]

#if defined(__SSE3__)
        if (sse3_supported)
        {
            final_sum128 = _mm_hadd_pd(final_sum128, final_sum128); // Requires SSE3
            total_sum += _mm_cvtsd_f64(final_sum128);               // Extract the first double
        }
        else
#endif
        { // Fallback if no SSE3 hadd_pd
            double sums_double[2];
            _mm_storeu_pd(sums_double, final_sum128);
            total_sum += sums_double[0] + sums_double[1];
        }
        // i is already at end_index
    }
    else if (sse2_supported && total_elements >= 2) // Fallback to SSE2 (2 doubles)
    {
        // SSE2 path remains unchanged (already uses loadu)
        size_t num_vectors = total_elements / 2;
        size_t end_index = num_vectors * 2;
        __m128d sum_vec = _mm_setzero_pd(); // 2x double vector

        for (i = 0; i < end_index; i += 2)
        {
            __m128d data_vec = _mm_loadu_pd(array + i); // Use unaligned load
            sum_vec = _mm_add_pd(sum_vec, data_vec);
        }

        // Horizontal sum using SSE3 hadd_pd if available
#if defined(__SSE3__)
        if (sse3_supported)
        {
            sum_vec = _mm_hadd_pd(sum_vec, sum_vec);
            total_sum += _mm_cvtsd_f64(sum_vec);
        }
        else
#endif
        { // Fallback horizontal sum if no SSE3 (store and sum)
            double sums_double[2];
            _mm_storeu_pd(sums_double, sum_vec);
            total_sum += sums_double[0] + sums_double[1];
        }
        // i is already at end_index
    }

#elif ARCH_ARM_NEON
    // Placeholder: Add NEON implementation for double here (if double precision supported)
    // if (neon_fp64_supported && total_elements >= 2) { ... i = end_index; }
#endif

    // Fallback scalar implementation for remainder or if no SIMD
    for (; i < total_elements; ++i)
    {
        total_sum += array[i];
    }
    return total_sum;
}

/**
 * @brief Sum of a double array using cache-friendly sequential access.
 */
double vsum_cache_friendly_sum_double(const double *restrict array, size_t total_elements)
{
    if (!array || total_elements == 0)
        return 0.0;

    double total_sum = 0.0;
    const size_t element_size = sizeof(double);
    size_t elements_per_cacheline = (VSUM_CACHE_LINE_SIZE / element_size);
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
