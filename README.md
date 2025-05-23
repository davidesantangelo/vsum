# VSUM - Optimized Array Summation

**vsum** is a C library specifically designed for **highly efficient summation** of array data. It focuses on optimized summation techniques for `int`, `float`, and `double` types. It demonstrates the use of multithreading (with dynamic thread count based on core availability), SIMD instructions (AVX/AVX2 with fallbacks to SSE/SSE2, runtime detection, SSE3 horizontal adds, and alignment handling), the `restrict` keyword for compiler optimization, and cache-aware access patterns to significantly accelerate the summation of large datasets.

**Current Focus:** Optimized Integer, Float, and Double Array Summation

**Important Note:** This library operates under the standard C assumption that the size of the array being summed is **known beforehand** and passed correctly to the library functions. It does **not** attempt unsafe or non-portable methods to guess array sizes from raw pointers.

## Features

*   **Data Types:** Supports optimized summation for `int`, `float`, and `double` arrays.
*   **Parallel Summation:**
    *   `vsum_parallel_sum_int`, `vsum_parallel_sum_float`, `vsum_parallel_sum_double`
    *   Utilizes POSIX threads (pthreads) to divide the summation task across multiple CPU cores for maximum throughput.
    *   Dynamically determines the number of threads based on array size and available cores (via `sysconf`), capped at a reasonable internal maximum.
    *   Includes a fallback to sequential processing for small arrays or if thread creation fails.
*   **SIMD Accelerated Summation:**
    *   `vsum_simd_sum_int`, `vsum_simd_sum_float`, `vsum_simd_sum_double`
    *   Leverages AVX/AVX2 intrinsics on x86-64 architectures if available **at runtime** for substantial speedups.
    *   **Includes fallbacks to SSE/SSE2** if AVX/AVX2 are not supported, providing acceleration on a wider range of x86-64 CPUs.
    *   Performs runtime CPU feature detection (using CPUID) to safely use the best available SIMD instructions (AVX2, AVX, SSE3, SSE2, SSE) for summation. Uses SSE3 horizontal adds (`hadd`) for float/double sums when available.
    *   Checks array pointer alignment once per call for AVX/AVX2 paths and uses faster aligned load instructions when possible, falling back to unaligned loads otherwise. SSE/SSE2 paths typically use unaligned loads.
    *   Uses 64-bit integer accumulation (`vsum_simd_sum_int`) or double-precision floating-point accumulation where appropriate to prevent overflow/precision loss during summation.
    *   Includes a standard scalar fallback if no suitable SIMD instructions are available or the array is too small.
*   **Cache-Friendly Summation:**
    *   `vsum_cache_friendly_sum_int`, `vsum_cache_friendly_sum_float`, `vsum_cache_friendly_sum_double`
    *   Processes the array sequentially. (Primarily for demonstration/comparison).
*   **Compiler Optimization Hints:** Uses the `restrict` keyword on array parameters to potentially allow better optimization by the compiler.
*   **Modular Design:** Provided as a simple header (`vsum.h`) and implementation (`vsum.c`) for easy integration.

## Dependencies

*   **POSIX Threads (pthreads):** Required for the parallel summation features. Most Unix-like systems (Linux, macOS) provide this. Link with `-pthread`.
*   **POSIX `sysconf`:** Used to determine the number of online processors for dynamic thread count adjustment. Available on most Unix-like systems. Requires `<unistd.h>`.
*   **(Optional) AVX/AVX2/SSE/SSE2/SSE3 Support:** For the SIMD summation features to provide acceleration, the target CPU must support the corresponding instructions. The library performs runtime checks, so it will safely fall back to scalar code if SIMD is not present, even if compiled with SIMD flags (like `-mavx2` or `-msse3`). Most x86-64 CPUs support at least SSE2; SSE3 is common and improves float/double horizontal sums.
*   **Compiler Intrinsics Support:** Requires a compiler (like GCC or Clang) that supports Intel intrinsics (`immintrin.h`, `pmmintrin.h` for SSE3, `emmintrin.h`, `xmmintrin.h`) and CPUID (`cpuid.h`) on x86-64.

## Compilation

Here's an example of how to compile the library and an example program using GCC or Clang on Linux/macOS:

**Compile the library source into an object file:**

*   **On x86-64 (Intel/AMD) - Recommended:** Enable AVX2 and SSE3 at compile time if your toolchain supports it. This allows the compiler to generate code for all levels, and the runtime check will select the best available path.
    ```bash
    # Enable AVX2 (implies AVX, SSE4, SSE3, SSE2, SSE)
    gcc -c vsum.c -o vsum.o -O2 -Wall -Wextra -pthread -mavx2
    # Or explicitly enable SSE3 if only targeting up to SSE/SSE2 but wanting hadd optimization
    # gcc -c vsum.c -o vsum.o -O2 -Wall -Wextra -pthread -msse3
    # Using -march=native might yield slightly better code for the specific machine compiling it
    # gcc -c vsum.c -o vsum.o -O2 -Wall -Wextra -pthread -march=native
    ```
*   **On other architectures (e.g., ARM):** The x86 SIMD parts will be disabled by preprocessor directives or fail the runtime check.
    ```bash
    gcc -c vsum.c -o vsum.o -O2 -Wall -Wextra -pthread
    ```
    *   `-O2`: Optimization level (or `-O3`).
    *   `-Wall -Wextra`: Enable useful compiler warnings.
    *   `-pthread`: Enable and link pthreads support.
    *   `-mavx2` / `-msse3` / `-march=native`: (Optional, **x86-64 only**) Allows the compiler to generate specific SIMD code, which the library then guards with runtime checks. The `restrict` keyword is standard C99 and doesn't require special flags.

## Usage

1.  **Include the header:**
    ```c
    #include "vsum.h"
    ```
2.  **Call the desired summation function:**
    ```c
    #include "vsum.h"
    #include <stdio.h>
    #include <stdlib.h>
    #include <time.h> // For timing
    #include <math.h> // For fabs in comparison
    #include <stdbool.h> // For bool

    // Define VSUM_AVX_ALIGNMENT if using aligned_alloc, otherwise it might not be defined
    // (It's defined in vsum.h, so including that should be sufficient)
    #ifndef VSUM_AVX_ALIGNMENT
    #define VSUM_AVX_ALIGNMENT 32 // Define fallback if header not included directly here
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
            int*    int_array    = (int*)aligned_alloc(VSUM_AVX_ALIGNMENT, array_size * sizeof(int));
            float*  float_array  = (float*)aligned_alloc(VSUM_AVX_ALIGNMENT, array_size * sizeof(float));
            double* double_array = (double*)aligned_alloc(VSUM_AVX_ALIGNMENT, array_size * sizeof(double));
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

        // --- Integer Summation Benchmarks ---
        printf("--- Integer Summation Benchmarks ---\n");
        start = clock(); int64_t expected_int = sequential_sum_int(int_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Sequential: %f s, Result: %lld\n", cpu_time_used, expected_int);


        start = clock(); int64_t res_int_par = vsum_parallel_sum_int(int_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Parallel:   %f s, Result: %lld (%s)\n", cpu_time_used, res_int_par, (res_int_par == expected_int ? "OK" : "FAIL"));

        start = clock(); int64_t res_int_simd = vsum_simd_sum_int(int_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("SIMD:       %f s, Result: %lld (%s)\n", cpu_time_used, res_int_simd, (res_int_simd == expected_int ? "OK" : "FAIL"));

        start = clock(); int64_t res_int_cache = vsum_cache_friendly_sum_int(int_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("CacheFr:    %f s, Result: %lld (%s)\n\n", cpu_time_used, res_int_cache, (res_int_cache == expected_int ? "OK" : "FAIL"));

        // --- Float Summation Benchmarks ---
        printf("--- Float Summation Benchmarks ---\n");
        start = clock(); double expected_float = sequential_sum_float(float_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Sequential: %f s, Result: %f\n", cpu_time_used, expected_float);

        start = clock(); double res_float_par = vsum_parallel_sum_float(float_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Parallel:   %f s, Result: %f (%s)\n", cpu_time_used, res_float_par, (check_double_result(res_float_par, expected_float) ? "OK" : "FAIL"));

        start = clock(); double res_float_simd = vsum_simd_sum_float(float_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("SIMD:       %f s, Result: %f (%s)\n", cpu_time_used, res_float_simd, (check_double_result(res_float_simd, expected_float) ? "OK" : "FAIL"));

        start = clock(); double res_float_cache = vsum_cache_friendly_sum_float(float_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("CacheFr:    %f s, Result: %f (%s)\n\n", cpu_time_used, res_float_cache, (check_double_result(res_float_cache, expected_float) ? "OK" : "FAIL"));

        // --- Double Summation Benchmarks ---
        printf("--- Double Summation Benchmarks ---\n");
        start = clock(); double expected_double = sequential_sum_double(double_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Sequential: %f s, Result: %f\n", cpu_time_used, expected_double);

        start = clock(); double res_double_par = vsum_parallel_sum_double(double_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Parallel:   %f s, Result: %f (%s)\n", cpu_time_used, res_double_par, (check_double_result(res_double_par, expected_double) ? "OK" : "FAIL"));

        start = clock(); double res_double_simd = vsum_simd_sum_double(double_array, array_size); end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("SIMD:       %f s, Result: %f (%s)\n", cpu_time_used, res_double_simd, (check_double_result(res_double_simd, expected_double) ? "OK" : "FAIL"));

        start = clock(); double res_double_cache = vsum_cache_friendly_sum_double(double_array, array_size); end = clock();
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

Performance varies significantly based on hardware, compiler, flags, and data size. Parallelism helps on large datasets with multiple cores. SIMD (AVX/AVX2/SSE2/SSE) provides significant speedups for summation if applicable and enabled. Runtime checks ensure safety across different CPUs. Alignment handling can provide a small additional boost for AVX/AVX2 SIMD paths.

*(The following results are from a specific run on a system with AVX2 and may vary significantly on different systems/compilations. SSE/SSE2 fallbacks will be slower than AVX/AVX2 but faster than scalar summation.)*

```text
Allocating arrays of 20000000 elements...
Allocated arrays (approx 152.59 MB each)
Initializing arrays...
Arrays initialized.

--- Integer Summation Benchmarks ---
Sequential: 0.016152 s, Result: 999999190
Parallel:   0.007566 s, Result: 999999190 (OK)
SIMD:       0.001788 s, Result: 999999190 (OK)
CacheFr:    0.008076 s, Result: 999999190 (OK)

--- Float Summation Benchmarks ---
Sequential: 0.020796 s, Result: 9900982.125526
Parallel:   0.013540 s, Result: 9900982.125497 (OK)
SIMD:       0.010913 s, Result: 9900982.125526 (OK)
CacheFr:    0.010398 s, Result: 9900982.125526 (OK)

--- Double Summation Benchmarks ---
Sequential: 0.019924 s, Result: 9900982.079208
Parallel:   0.013204 s, Result: 9900982.079208 (OK)
SIMD:       0.009891 s, Result: 9900982.079208 (OK)
CacheFr:    0.009962 s, Result: 9900982.079208 (OK)

Cleaning up memory...
Done.

```
