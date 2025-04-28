# VSIZE Library

**vsize** is a C library designed for efficient processing of array data, focusing initially on optimized summation techniques for `int`, `float`, and `double` types. It demonstrates the use of multithreading, SIMD instructions (AVX/AVX2 with fallbacks to SSE/SSE2, runtime detection, and alignment handling), and cache-aware access patterns to potentially accelerate computations on large datasets.

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
    *   **Includes fallbacks to SSE/SSE2** if AVX/AVX2 are not supported, providing acceleration on a wider range of x86-64 CPUs.
    *   Performs runtime CPU feature detection (using CPUID) to safely use the best available SIMD instructions (AVX2, AVX, SSE2, SSE).
    *   Checks array pointer alignment for AVX/AVX2 paths and uses faster aligned load instructions when possible, falling back to unaligned loads otherwise. SSE/SSE2 paths typically use unaligned loads for simplicity.
    *   Uses 64-bit integer accumulation (`vsize_simd_sum_int`) or double-precision floating-point accumulation where appropriate to prevent overflow/precision loss.
    *   Includes a standard scalar fallback if no suitable SIMD instructions are available or the array is too small.
*   **Cache-Friendly Summation:**
    *   `vsize_cache_friendly_sum_int`, `vsize_cache_friendly_sum_float`, `vsize_cache_friendly_sum_double`
    *   Processes the array sequentially. (Primarily for demonstration/comparison; modern prefetchers are often effective for simple sequential access).
*   **Modular Design:** Provided as a simple header (`vsize.h`) and implementation (`vsize.c`) for easy integration.

## Dependencies

*   **POSIX Threads (pthreads):** Required for the parallel summation features. Most Unix-like systems (Linux, macOS) provide this. Link with `-pthread`.
*   **(Optional) AVX/AVX2/SSE/SSE2 Support:** For the SIMD features to provide acceleration, the target CPU must support the corresponding instructions. The library performs runtime checks, so it will safely fall back to scalar code if SIMD is not present, even if compiled with SIMD flags (like `-mavx2` or `-msse2`). Most x86-64 CPUs support at least SSE2.
*   **Compiler Intrinsics Support:** Requires a compiler (like GCC or Clang) that supports Intel intrinsics (`immintrin.h`, `emmintrin.h`, `xmmintrin.h`) and CPUID (`cpuid.h`) on x86-64. SSE3 intrinsics (`pmmintrin.h`) are used for horizontal sums if available (detected by compiler via `__SSE3__`), otherwise a store-and-sum fallback is used.

## Compilation

Here's an example of how to compile the library and an example program using GCC or Clang on Linux/macOS:

**Compile the library source into an object file:**

*   **On x86-64 (Intel/AMD) - Recommended:** Enable AVX2 at compile time if your toolchain supports it. This allows the compiler to generate code for all levels (AVX2, AVX, SSE2, SSE), and the runtime check will select the best available path.
    ```bash
    # Enable AVX2 (implies AVX, SSE4, SSE3, SSE2, SSE)
    gcc -c vsize.c -o vsize.o -O2 -Wall -Wextra -pthread -mavx2
    # Or enable only up to SSE2 if AVX is not desired/available for compilation
    # gcc -c vsize.c -o vsize.o -O2 -Wall -Wextra -pthread -msse2
    # Using -march=native might yield slightly better code for the specific machine compiling it
    # gcc -c vsize.c -o vsize.o -O2 -Wall -Wextra -pthread -march=native
    ```
*   **On other architectures (e.g., ARM):** The x86 SIMD parts will be disabled by preprocessor directives or fail the runtime check. NEON placeholders exist but are not yet implemented.
    ```bash
    gcc -c vsize.c -o vsize.o -O2 -Wall -Wextra -pthread
    ```
    *   `-O2`: Optimization level (or `-O3`).
    *   `-Wall -Wextra`: Enable useful compiler warnings.
    *   `-pthread`: Enable and link pthreads support.
    *   `-mavx2` / `-msse2` / `-march=native`: (Optional, **x86-64 only**) Allows the compiler to generate specific SIMD code, which the library then guards with runtime checks.

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
    #include <stdbool.h> // For bool

    // Define VSIZE_AVX_ALIGNMENT if using aligned_alloc, otherwise it might not be defined
    // (It's defined in vsize.h, so including that should be sufficient)
    #ifndef VSIZE_AVX_ALIGNMENT
    #define VSIZE_AVX_ALIGNMENT 32 // Define fallback if header not included directly here
    #endif

    // Example sequential sums for comparison
    int64_t sequential_sum_int(const int *array, size_t n) { int64_t s=0; for(size_t i=0; i<n; ++i) s+=array[i]; return s; }
    double sequential_sum_float(const float *array, size_t n) { double s=0.0; for(size_t i=0; i<n; ++i) s+=array[i]; return s; }
    double sequential_sum_double(const double *array, size_t n) { double s=0.0; for(size_t i=0; i<n; ++i) s+=array[i]; return s; }

    // Helper to check float/double results approximately
    bool check_double_result(double res, double expected) {
        // Handle cases where expected is zero or very small
        double abs_expected = fabs(expected);
        double tolerance = (abs_expected > 1e-9) ? (1e-9 * abs_expected) : 1e-12;
        return fabs(res - expected) < tolerance;
    }

    int main() {
        size_t array_size = 20000000;
        printf("Allocating arrays of %zu elements...\n", array_size);

        // Allocate aligned memory if possible for SIMD demo (requires C11 or POSIX)
        // Using malloc and relying on unaligned loads is also an option.
        #if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_THREADS__) && !defined(__STDC_NO_ATOMICS__)
            // C11 aligned_alloc
            int*    int_array    = (int*)aligned_alloc(VSIZE_AVX_ALIGNMENT, array_size * sizeof(int));
            float*  float_array  = (float*)aligned_alloc(VSIZE_AVX_ALIGNMENT, array_size * sizeof(float));
            double* double_array = (double*)aligned_alloc(VSIZE_AVX_ALIGNMENT, array_size * sizeof(double));
        #else
            // Fallback using malloc (SIMD will use unaligned loads if necessary)
            printf("Warning: C11 aligned_alloc not detected, using malloc (SIMD will use unaligned loads).\n");
            int*    int_array    = (int*)malloc(array_size * sizeof(int));
            float*  float_array  = (float*)malloc(array_size * sizeof(float));
            double* double_array = (double*)malloc(array_size * sizeof(double));
        #endif


        if (!int_array || !float_array || !double_array) {
            perror("Failed to allocate memory");
            // free any successfully allocated arrays before exiting
            free(int_array); free(float_array); free(double_array);
            return 1;
        }
        printf("Allocated arrays (approx %.2f MB each)\n", (double)array_size * sizeof(double) / (1024*1024));

        printf("Initializing arrays...\n");
        for (size_t i = 0; i < array_size; ++i) {
            int_array[i]    = (int)(i % 101); // Example data
            float_array[i]  = (float)(i % 101) / 101.0f;
            double_array[i] = (double)(i % 101) / 101.0;
        }
        printf("Arrays initialized.\n\n");

        clock_t start, end;
        double cpu_time_used;

        // --- Integer Benchmarks ---
        printf("--- Integer Benchmarks ---\n");
        start = clock(); int64_t expected_int = sequential_sum_int(int_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Sequential: %f s, Result: %lld\n", cpu_time_used, expected_int);


        start = clock(); int64_t res_int_par = vsize_parallel_sum_int(int_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Parallel:   %f s, Result: %lld (%s)\n", cpu_time_used, res_int_par, (res_int_par == expected_int ? "OK" : "FAIL"));

        start = clock(); int64_t res_int_simd = vsize_simd_sum_int(int_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("SIMD:       %f s, Result: %lld (%s)\n", cpu_time_used, res_int_simd, (res_int_simd == expected_int ? "OK" : "FAIL"));

        start = clock(); int64_t res_int_cache = vsize_cache_friendly_sum_int(int_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("CacheFr:    %f s, Result: %lld (%s)\n\n", cpu_time_used, res_int_cache, (res_int_cache == expected_int ? "OK" : "FAIL"));

        // --- Float Benchmarks ---
        printf("--- Float Benchmarks ---\n");
        start = clock(); double expected_float = sequential_sum_float(float_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Sequential: %f s, Result: %f\n", cpu_time_used, expected_float);

        start = clock(); double res_float_par = vsize_parallel_sum_float(float_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Parallel:   %f s, Result: %f (%s)\n", cpu_time_used, res_float_par, (check_double_result(res_float_par, expected_float) ? "OK" : "FAIL"));

        start = clock(); double res_float_simd = vsize_simd_sum_float(float_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("SIMD:       %f s, Result: %f (%s)\n", cpu_time_used, res_float_simd, (check_double_result(res_float_simd, expected_float) ? "OK" : "FAIL"));

        start = clock(); double res_float_cache = vsize_cache_friendly_sum_float(float_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("CacheFr:    %f s, Result: %f (%s)\n\n", cpu_time_used, res_float_cache, (check_double_result(res_float_cache, expected_float) ? "OK" : "FAIL"));

        // --- Double Benchmarks ---
        printf("--- Double Benchmarks ---\n");
        start = clock(); double expected_double = sequential_sum_double(double_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Sequential: %f s, Result: %f\n", cpu_time_used, expected_double);

        start = clock(); double res_double_par = vsize_parallel_sum_double(double_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Parallel:   %f s, Result: %f (%s)\n", cpu_time_used, res_double_par, (check_double_result(res_double_par, expected_double) ? "OK" : "FAIL"));

        start = clock(); double res_double_simd = vsize_simd_sum_double(double_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("SIMD:       %f s, Result: %f (%s)\n", cpu_time_used, res_double_simd, (check_double_result(res_double_simd, expected_double) ? "OK" : "FAIL"));

        start = clock(); double res_double_cache = vsize_cache_friendly_sum_double(double_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("CacheFr:    %f s, Result: %f (%s)\n\n", cpu_time_used, res_double_cache, (check_double_result(res_double_cache, expected_double) ? "OK" : "FAIL"));


        printf("Cleaning up memory...\n");
        free(int_array);
        free(float_array);
        free(double_array);
        printf("Done.\n");
        return 0;
    }
    ```

## Benchmark Example (Illustrative)

Performance varies significantly based on hardware, compiler, flags, and data size. Parallelism helps on large datasets with multiple cores. SIMD (AVX/AVX2/SSE2/SSE) provides significant speedups if applicable and enabled. Runtime checks ensure safety across different CPUs. Alignment handling can provide a small additional boost for AVX/AVX2 SIMD paths.

*(The following results are from a specific run on a system with AVX2 and may vary significantly on different systems/compilations. SSE/SSE2 fallbacks will be slower than AVX/AVX2 but faster than scalar.)*

```text
Allocating arrays of 20000000 elements...
Allocated arrays (approx 152.59 MB each)
Initializing arrays...
Arrays initialized.

--- Integer Benchmarks ---
Sequential: 0.016152 s, Result: 999999190
Parallel:   0.007566 s, Result: 999999190 (OK)
SIMD:       0.001788 s, Result: 999999190 (OK)
CacheFr:    0.008076 s, Result: 999999190 (OK)

--- Float Benchmarks ---
Sequential: 0.020796 s, Result: 9900982.125526
Parallel:   0.013540 s, Result: 9900982.125497 (OK)
SIMD:       0.010913 s, Result: 9900982.125526 (OK)
CacheFr:    0.010398 s, Result: 9900982.125526 (OK)

--- Double Benchmarks ---
Sequential: 0.019924 s, Result: 9900982.079208
Parallel:   0.013204 s, Result: 9900982.079208 (OK)
SIMD:       0.009891 s, Result: 9900982.079208 (OK)
CacheFr:    0.009962 s, Result: 9900982.079208 (OK)

Cleaning up memory...
Done.

```

## Future Work

*   **Implement other SIMD Instruction Sets:** Add optimized code paths for:
    *   AVX-512 (512-bit vectors, available on some newer Intel CPUs).
    *   NEON (for ARM architectures, common on mobile/embedded and Apple Silicon).
*   **Implement Other Operations:** Add functions for min/max, average, vector scaling, dot products, etc.
*   **Dynamic Library:** Consider creating static (`.a`) or dynamic (`.so`/`.dylib`) library versions for easier distribution.
*   **Error Handling:** Improve error reporting beyond `fprintf` to `stderr` (e.g., return codes, error callbacks).
*   **Configuration:** Allow runtime configuration of max threads or disabling SIMD.
*   **Precision:** Investigate higher-precision summation algorithms (e.g., Kahan, pairwise) for floating-point types, especially in SIMD implementations, as an optional alternative.