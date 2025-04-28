# VSIZE Library

**vsize** is a C library designed for efficient processing of array data, focusing initially on optimized summation techniques for `int`, `float`, and `double` types. It demonstrates the use of multithreading, SIMD instructions (AVX/AVX2 with runtime detection and alignment handling), and cache-aware access patterns to potentially accelerate computations on large datasets.

**Current Focus:** Integer, Float, and Double Array Summation

**Important Note:** This library operates under the standard C assumption that the size of the array being processed is **known beforehand** and passed correctly to the library functions. It does **not** attempt unsafe or non-portable methods to guess array sizes from raw pointers.

## Features

*   **Data Types:** Supports summation for `int`, `float`, and `double` arrays.
*   **Parallel Summation:**
    *   `vsize_parallel_sum_int`, `vsize_parallel_sum_float`, `vsize_parallel_sum_double`
    *   Utilizes POSIX threads (pthreads) to divide the summation task across multiple CPU cores.
    *   Dynamically determines the number of threads based on array size and available cores (up to a defined maximum).
    *   Includes a fallback to sequential processing for small arrays or if thread creation fails.
*   **SIMD Accelerated Summation:**
    *   `vsize_simd_sum_int`, `vsize_simd_sum_float`, `vsize_simd_sum_double`
    *   Leverages AVX/AVX2 intrinsics on x86-64 architectures if available **at runtime**.
    *   Performs runtime CPU feature detection (using CPUID) to safely use AVX/AVX2 instructions only when supported.
    *   Checks array pointer alignment and uses faster aligned load instructions (`_mm256_load_si256`, `_mm256_load_ps`, `_mm256_load_pd`) when possible, falling back to unaligned loads otherwise.
    *   Uses 64-bit integer accumulation (`vsize_simd_sum_int`) or double-precision floating-point accumulation where appropriate to prevent overflow/precision loss.
    *   Includes a standard scalar fallback if suitable SIMD instructions are unavailable or the array is too small.
*   **Cache-Friendly Summation:**
    *   `vsize_cache_friendly_sum_int`, `vsize_cache_friendly_sum_float`, `vsize_cache_friendly_sum_double`
    *   Processes the array sequentially in chunks roughly matching the CPU cache line size. (Primarily for demonstration; modern prefetchers are often effective).
*   **Modular Design:** Provided as a simple header (`vsize.h`) and implementation (`vsize.c`) for easy integration.

## Dependencies

*   **POSIX Threads (pthreads):** Required for the parallel summation features. Most Unix-like systems (Linux, macOS) provide this. Link with `-pthread`.
*   **(Optional) AVX/AVX2 Support:** For the SIMD features to provide acceleration, the target CPU must support AVX/AVX2 instructions. The library performs runtime checks, so it will safely fall back to scalar code if AVX/AVX2 is not present, even if compiled with AVX flags.
*   **Compiler Intrinsics Support:** Requires a compiler (like GCC or Clang) that supports Intel intrinsics (`immintrin.h`) and CPUID (`cpuid.h`) on x86-64.

## Compilation

Here's an example of how to compile the library and an example program using GCC or Clang on Linux/macOS:

**Compile the library source into an object file:**

*   **On x86-64 (Intel/AMD) - Recommended:** Enable AVX/AVX2 at compile time if your toolchain supports it. The runtime check will ensure safety.
    ```bash
    # Enable AVX2 (implies AVX)
    gcc -c vsize.c -o vsize.o -O2 -Wall -Wextra -pthread -mavx2
    # Or just enable AVX
    # gcc -c vsize.c -o vsize.o -O2 -Wall -Wextra -pthread -mavx
    ```
*   **On x86-64 *without* specific AVX flags, or on other architectures (e.g., ARM):** The SIMD parts for x86 will likely be disabled by preprocessor or fail the runtime check.
    ```bash
    gcc -c vsize.c -o vsize.o -O2 -Wall -Wextra -pthread
    ```
    *   `-O2`: Optimization level (or `-O3`).
    *   `-Wall -Wextra`: Enable useful compiler warnings.
    *   `-pthread`: Enable and link pthreads support.
    *   `-mavx2` / `-mavx`: (Optional, **x86-64 only**) Allows the compiler to generate AVX/AVX2 code, which the library then guards with runtime checks.

## Usage

1.  **Include the header:**
    ```c
    #include "vsize.h"
    ```
2.  **Call the desired function:**
    ```c
    #include "vsize.h"
    #include <stdio.h>
    #include <stdlib.h>
    #include <time.h> // For timing
    #include <math.h> // For fabs in comparison

    // Example sequential sums for comparison
    int64_t sequential_sum_int(const int *array, size_t n) { int64_t s=0; for(size_t i=0; i<n; ++i) s+=array[i]; return s; }
    double sequential_sum_float(const float *array, size_t n) { double s=0.0; for(size_t i=0; i<n; ++i) s+=array[i]; return s; }
    double sequential_sum_double(const double *array, size_t n) { double s=0.0; for(size_t i=0; i<n; ++i) s+=array[i]; return s; }

    // Helper to check float/double results approximately
    bool check_double_result(double res, double expected) {
        return fabs(res - expected) < 1e-9 * fabs(expected) + 1e-12;
    }

    int main() {
        size_t array_size = 20000000;
        printf("Allocating arrays of %zu elements...\n", array_size);

        // Allocate aligned memory if possible for SIMD demo
        int*    int_array    = (int*)aligned_alloc(VSIZE_AVX_ALIGNMENT, array_size * sizeof(int));
        float*  float_array  = (float*)aligned_alloc(VSIZE_AVX_ALIGNMENT, array_size * sizeof(float));
        double* double_array = (double*)aligned_alloc(VSIZE_AVX_ALIGNMENT, array_size * sizeof(double));

        if (!int_array || !float_array || !double_array) {
            perror("Failed to allocate memory");
            // free any successfully allocated arrays before exiting
            free(int_array); free(float_array); free(double_array);
            return 1;
        }
        printf("Allocated arrays (approx %.2f MB each)\n", (double)array_size * sizeof(double) / (1024*1024));

        printf("Initializing arrays...\n");
        for (size_t i = 0; i < array_size; ++i) {
            int_array[i]    = (int)(i % 101);
            float_array[i]  = (float)(i % 101) / 101.0f;
            double_array[i] = (double)(i % 101) / 101.0;
        }
        printf("Arrays initialized.\n\n");

        clock_t start, end;
        double cpu_time_used;

        // --- Integer Benchmarks ---
        printf("--- Integer Benchmarks ---\n");
        int64_t expected_int = sequential_sum_int(int_array, array_size);
        printf("Expected Sum: %lld\n", expected_int);

        start = clock(); int64_t res_int_par = vsize_parallel_sum_int(int_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Parallel: %f s, Result: %lld (%s)\n", cpu_time_used, res_int_par, (res_int_par == expected_int ? "OK" : "FAIL"));

        start = clock(); int64_t res_int_simd = vsize_simd_sum_int(int_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("SIMD:     %f s, Result: %lld (%s)\n", cpu_time_used, res_int_simd, (res_int_simd == expected_int ? "OK" : "FAIL"));

        start = clock(); int64_t res_int_cache = vsize_cache_friendly_sum_int(int_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("CacheFr:  %f s, Result: %lld (%s)\n\n", cpu_time_used, res_int_cache, (res_int_cache == expected_int ? "OK" : "FAIL"));

        // --- Float Benchmarks ---
        printf("--- Float Benchmarks ---\n");
        double expected_float = sequential_sum_float(float_array, array_size);
        printf("Expected Sum: %f\n", expected_float);

        start = clock(); double res_float_par = vsize_parallel_sum_float(float_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Parallel: %f s, Result: %f (%s)\n", cpu_time_used, res_float_par, (check_double_result(res_float_par, expected_float) ? "OK" : "FAIL"));

        start = clock(); double res_float_simd = vsize_simd_sum_float(float_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("SIMD:     %f s, Result: %f (%s)\n", cpu_time_used, res_float_simd, (check_double_result(res_float_simd, expected_float) ? "OK" : "FAIL"));

        start = clock(); double res_float_cache = vsize_cache_friendly_sum_float(float_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("CacheFr:  %f s, Result: %f (%s)\n\n", cpu_time_used, res_float_cache, (check_double_result(res_float_cache, expected_float) ? "OK" : "FAIL"));

        // --- Double Benchmarks ---
        printf("--- Double Benchmarks ---\n");
        double expected_double = sequential_sum_double(double_array, array_size);
        printf("Expected Sum: %f\n", expected_double);

        start = clock(); double res_double_par = vsize_parallel_sum_double(double_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Parallel: %f s, Result: %f (%s)\n", cpu_time_used, res_double_par, (check_double_result(res_double_par, expected_double) ? "OK" : "FAIL"));

        start = clock(); double res_double_simd = vsize_simd_sum_double(double_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("SIMD:     %f s, Result: %f (%s)\n", cpu_time_used, res_double_simd, (check_double_result(res_double_simd, expected_double) ? "OK" : "FAIL"));

        start = clock(); double res_double_cache = vsize_cache_friendly_sum_double(double_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("CacheFr:  %f s, Result: %f (%s)\n\n", cpu_time_used, res_double_cache, (check_double_result(res_double_cache, expected_double) ? "OK" : "FAIL"));


        printf("Cleaning up memory...\n");
        free(int_array);
        free(float_array);
        free(double_array);
        printf("Done.\n");
        return 0;
    }
    ```

## Benchmark Example (Illustrative)

Performance varies significantly based on hardware, compiler, flags, and data size. Parallelism helps on large datasets with multiple cores. SIMD (AVX/AVX2) provides significant speedups if applicable and enabled. Runtime checks ensure safety across different CPUs. Alignment handling can provide a small additional boost for SIMD.

*(The following results are from a specific run and may vary on different systems/compilations.)*

```text
Allocating arrays of 20000000 elements...
Allocated arrays (approx 152.59 MB each)
Initializing arrays...
Arrays initialized.

--- Integer Benchmarks ---
Expected Sum: 999999190
Parallel: 0.007566 s, Result: 999999190 (OK)
SIMD:     0.001788 s, Result: 999999190 (OK)
CacheFr:  0.008076 s, Result: 999999190 (OK)

--- Float Benchmarks ---
Expected Sum: 9900982.125526
Parallel: 0.013540 s, Result: 9900982.125497 (OK)
SIMD:     0.010913 s, Result: 9900982.125526 (OK)
CacheFr:  0.010398 s, Result: 9900982.125526 (OK)

--- Double Benchmarks ---
Expected Sum: 9900982.079208
Parallel: 0.013204 s, Result: 9900982.079208 (OK)
SIMD:     0.009891 s, Result: 9900982.079208 (OK)
CacheFr:  0.009962 s, Result: 9900982.079208 (OK)

Cleaning up memory...
Done.
```

## Future Work

*   **Implement other SIMD Instruction Sets:** Add optimized code paths for:
    *   SSE/SSE2 (128-bit vectors, widely available on x86).
    *   AVX-512 (512-bit vectors, available on some newer Intel CPUs).
    *   NEON (for ARM architectures, common on mobile/embedded and Apple Silicon).
*   **Implement Other Operations:** Add functions for min/max, average, vector scaling, dot products, etc.
*   **Dynamic Library:** Consider creating static (`.a`) or dynamic (`.so`/`.dylib`) library versions for easier distribution.
*   **Error Handling:** Improve error reporting beyond `fprintf` to `stderr`.
*   **Configuration:** Allow runtime configuration of max threads or SIMD usage.