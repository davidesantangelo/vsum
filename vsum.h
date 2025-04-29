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

#ifndef VSUM_H
#define VSUM_H

#include <stddef.h>  // For size_t
#include <stdint.h>  // For int64_t
#include <stdbool.h> // For bool type used internally

/**
 * @file vsum.h
 * @brief Public interface for the vsum library.
 *
 * Provides optimized array summation functions using multithreading,
 * SIMD instructions (with runtime checks and alignment handling),
 * and cache-friendly access patterns for int, float, and double types.
 *
 * IMPORTANT: All functions require the total size of the array
 * (`total_elements`) to be provided correctly.
 */

// --- Configuration Constants ---

/**
 * @brief Alignment requirement (in bytes) for optimal AVX/AVX2 performance.
 * Use this value when allocating memory with functions like aligned_alloc
 * if you intend to use the SIMD functions with potentially aligned data.
 */
#define VSUM_AVX_ALIGNMENT 32

// --- Integer Functions ---

/**
 * @brief Calculates the sum of elements in an integer array using multithreading.
 * Dynamically determines the number of threads based on available cores.
 * @param array Pointer to the integer array (const int* restrict).
 * @param total_elements Total number of elements in the array.
 * @return int64_t The total sum. Returns 0 on error or empty/null array.
 */
int64_t vsum_parallel_sum_int(const int *restrict array, size_t total_elements);

/**
 * @brief Calculates the sum of elements in an integer array using SIMD (AVX2 if available).
 * Performs runtime check for AVX2 support. Uses aligned loads if possible.
 * Uses 64-bit accumulation to prevent overflow. Falls back to SSE2/scalar if AVX2 unavailable.
 * @param array Pointer to the integer array (const int* restrict).
 * @param total_elements Total number of elements in the array.
 * @return int64_t The total sum.
 */
int64_t vsum_simd_sum_int(const int *restrict array, size_t total_elements);

/**
 * @brief Calculates the sum of elements in an integer array with cache-friendly sequential access.
 * @param array Pointer to the integer array (const int* restrict).
 * @param total_elements Total number of elements in the array.
 * @return int64_t The total sum. Returns 0 if array is null/empty.
 */
int64_t vsum_cache_friendly_sum_int(const int *restrict array, size_t total_elements);

// --- Float Functions ---

/**
 * @brief Calculates the sum of elements in a float array using multithreading.
 * Dynamically determines the number of threads based on available cores.
 * @param array Pointer to the float array (const float* restrict).
 * @param total_elements Total number of elements in the array.
 * @return double The total sum (using double for better precision). Returns 0.0 on error or empty/null array.
 */
double vsum_parallel_sum_float(const float *restrict array, size_t total_elements);

/**
 * @brief Calculates the sum of elements in a float array using SIMD (AVX if available).
 * Performs runtime check for AVX support. Uses aligned loads if possible.
 * Uses double precision for intermediate sums in some cases. Falls back to SSE/scalar if AVX unavailable.
 * Note: AVX (not just AVX2) is sufficient for basic float operations.
 * @param array Pointer to the float array (const float* restrict).
 * @param total_elements Total number of elements in the array.
 * @return double The total sum.
 */
double vsum_simd_sum_float(const float *restrict array, size_t total_elements);

/**
 * @brief Calculates the sum of elements in a float array with cache-friendly sequential access.
 * @param array Pointer to the float array (const float* restrict).
 * @param total_elements Total number of elements in the array.
 * @return double The total sum. Returns 0.0 if array is null/empty.
 */
double vsum_cache_friendly_sum_float(const float *restrict array, size_t total_elements);

// --- Double Functions ---

/**
 * @brief Calculates the sum of elements in a double array using multithreading.
 * Dynamically determines the number of threads based on available cores.
 * @param array Pointer to the double array (const double* restrict).
 * @param total_elements Total number of elements in the array.
 * @return double The total sum. Returns 0.0 on error or empty/null array.
 */
double vsum_parallel_sum_double(const double *restrict array, size_t total_elements);

/**
 * @brief Calculates the sum of elements in a double array using SIMD (AVX if available).
 * Performs runtime check for AVX support. Uses aligned loads if possible.
 * Falls back to SSE2/scalar if AVX unavailable.
 * Note: AVX (not just AVX2) is sufficient for basic double operations.
 * @param array Pointer to the double array (const double* restrict).
 * @param total_elements Total number of elements in the array.
 * @return double The total sum.
 */
double vsum_simd_sum_double(const double *restrict array, size_t total_elements);

/**
 * @brief Calculates the sum of elements in a double array with cache-friendly sequential access.
 * @param array Pointer to the double array (const double* restrict).
 * @param total_elements Total number of elements in the array.
 * @return double The total sum. Returns 0.0 if array is null/empty.
 */
double vsum_cache_friendly_sum_double(const double *restrict array, size_t total_elements);

// --- Internal Helper Function Declarations (Optional, for testing/visibility) ---
// bool vsum_internal_check_avx2(void); // Example if needed publicly

#endif // VSUM_H
