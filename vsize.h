#ifndef VSIZE_H
#define VSIZE_H

#include <stddef.h>  // For size_t
#include <stdint.h>  // For int64_t
#include <stdbool.h> // For bool type used internally

/**
 * @file vsize.h
 * @brief Public interface for the vsize library.
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
#define VSIZE_AVX_ALIGNMENT 32

// --- Integer Functions ---

/**
 * @brief Calculates the sum of elements in an integer array using multithreading.
 * Dynamically determines the number of threads.
 * @param array Pointer to the integer array (const int*).
 * @param total_elements Total number of elements in the array.
 * @return int64_t The total sum. Returns 0 on error or empty/null array.
 */
int64_t vsize_parallel_sum_int(const int *array, size_t total_elements);

/**
 * @brief Calculates the sum of elements in an integer array using SIMD (AVX2 if available).
 * Performs runtime check for AVX2 support. Uses aligned loads if possible.
 * Uses 64-bit accumulation to prevent overflow. Falls back to scalar if AVX2 unavailable.
 * @param array Pointer to the integer array (const int*).
 * @param total_elements Total number of elements in the array.
 * @return int64_t The total sum.
 */
int64_t vsize_simd_sum_int(const int *array, size_t total_elements);

/**
 * @brief Calculates the sum of elements in an integer array with cache-friendly sequential access.
 * @param array Pointer to the integer array (const int*).
 * @param total_elements Total number of elements in the array.
 * @return int64_t The total sum. Returns 0 if array is null/empty.
 */
int64_t vsize_cache_friendly_sum_int(const int *array, size_t total_elements);

// --- Float Functions ---

/**
 * @brief Calculates the sum of elements in a float array using multithreading.
 * @param array Pointer to the float array (const float*).
 * @param total_elements Total number of elements in the array.
 * @return double The total sum (using double for better precision). Returns 0.0 on error or empty/null array.
 */
double vsize_parallel_sum_float(const float *array, size_t total_elements);

/**
 * @brief Calculates the sum of elements in a float array using SIMD (AVX if available).
 * Performs runtime check for AVX support. Uses aligned loads if possible.
 * Uses double precision for intermediate sums in some cases. Falls back to scalar if AVX unavailable.
 * Note: AVX (not just AVX2) is sufficient for basic float operations.
 * @param array Pointer to the float array (const float*).
 * @param total_elements Total number of elements in the array.
 * @return double The total sum.
 */
double vsize_simd_sum_float(const float *array, size_t total_elements);

/**
 * @brief Calculates the sum of elements in a float array with cache-friendly sequential access.
 * @param array Pointer to the float array (const float*).
 * @param total_elements Total number of elements in the array.
 * @return double The total sum. Returns 0.0 if array is null/empty.
 */
double vsize_cache_friendly_sum_float(const float *array, size_t total_elements);

// --- Double Functions ---

/**
 * @brief Calculates the sum of elements in a double array using multithreading.
 * @param array Pointer to the double array (const double*).
 * @param total_elements Total number of elements in the array.
 * @return double The total sum. Returns 0.0 on error or empty/null array.
 */
double vsize_parallel_sum_double(const double *array, size_t total_elements);

/**
 * @brief Calculates the sum of elements in a double array using SIMD (AVX if available).
 * Performs runtime check for AVX support. Uses aligned loads if possible.
 * Falls back to scalar if AVX unavailable.
 * Note: AVX (not just AVX2) is sufficient for basic double operations.
 * @param array Pointer to the double array (const double*).
 * @param total_elements Total number of elements in the array.
 * @return double The total sum.
 */
double vsize_simd_sum_double(const double *array, size_t total_elements);

/**
 * @brief Calculates the sum of elements in a double array with cache-friendly sequential access.
 * @param array Pointer to the double array (const double*).
 * @param total_elements Total number of elements in the array.
 * @return double The total sum. Returns 0.0 if array is null/empty.
 */
double vsize_cache_friendly_sum_double(const double *array, size_t total_elements);

// --- Internal Helper Function Declarations (Optional, for testing/visibility) ---
// bool vsize_internal_check_avx2(void); // Example if needed publicly

#endif // VSIZE_H
