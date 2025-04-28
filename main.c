#include "vsize.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h> // For timing
#include <math.h> // For fabs in comparison

// Example sequential sums for comparison
int64_t sequential_sum_int(const int *array, size_t n)
{
    int64_t s = 0;
    for (size_t i = 0; i < n; ++i)
        s += array[i];
    return s;
}
double sequential_sum_float(const float *array, size_t n)
{
    double s = 0.0;
    for (size_t i = 0; i < n; ++i)
        s += array[i];
    return s;
}
double sequential_sum_double(const double *array, size_t n)
{
    double s = 0.0;
    for (size_t i = 0; i < n; ++i)
        s += array[i];
    return s;
}

// Helper to check float/double results approximately
bool check_double_result(double res, double expected)
{
    return fabs(res - expected) < 1e-9 * fabs(expected) + 1e-12;
}

int main()
{
    size_t array_size = 20000000;
    printf("Allocating arrays of %zu elements...\n", array_size);

    // Allocate aligned memory if possible for SIMD demo
    int *int_array = (int *)aligned_alloc(VSIZE_AVX_ALIGNMENT, array_size * sizeof(int));
    float *float_array = (float *)aligned_alloc(VSIZE_AVX_ALIGNMENT, array_size * sizeof(float));
    double *double_array = (double *)aligned_alloc(VSIZE_AVX_ALIGNMENT, array_size * sizeof(double));

    if (!int_array || !float_array || !double_array)
    {
        perror("Failed to allocate memory");
        // free any successfully allocated arrays before exiting
        free(int_array);
        free(float_array);
        free(double_array);
        return 1;
    }
    printf("Allocated arrays (approx %.2f MB each)\n", (double)array_size * sizeof(double) / (1024 * 1024));

    printf("Initializing arrays...\n");
    for (size_t i = 0; i < array_size; ++i)
    {
        int_array[i] = (int)(i % 101);
        float_array[i] = (float)(i % 101) / 101.0f;
        double_array[i] = (double)(i % 101) / 101.0;
    }
    printf("Arrays initialized.\n\n");

    clock_t start, end;
    double cpu_time_used;

    // --- Integer Benchmarks ---
    printf("--- Integer Benchmarks ---\n");
    int64_t expected_int = sequential_sum_int(int_array, array_size);
    printf("Expected Sum: %lld\n", expected_int);

    start = clock();
    int64_t res_int_par = vsize_parallel_sum_int(int_array, array_size);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Parallel: %f s, Result: %lld (%s)\n", cpu_time_used, res_int_par, (res_int_par == expected_int ? "OK" : "FAIL"));

    start = clock();
    int64_t res_int_simd = vsize_simd_sum_int(int_array, array_size);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("SIMD:     %f s, Result: %lld (%s)\n", cpu_time_used, res_int_simd, (res_int_simd == expected_int ? "OK" : "FAIL"));

    start = clock();
    int64_t res_int_cache = vsize_cache_friendly_sum_int(int_array, array_size);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("CacheFr:  %f s, Result: %lld (%s)\n\n", cpu_time_used, res_int_cache, (res_int_cache == expected_int ? "OK" : "FAIL"));

    // --- Float Benchmarks ---
    printf("--- Float Benchmarks ---\n");
    double expected_float = sequential_sum_float(float_array, array_size);
    printf("Expected Sum: %f\n", expected_float);

    start = clock();
    double res_float_par = vsize_parallel_sum_float(float_array, array_size);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Parallel: %f s, Result: %f (%s)\n", cpu_time_used, res_float_par, (check_double_result(res_float_par, expected_float) ? "OK" : "FAIL"));

    start = clock();
    double res_float_simd = vsize_simd_sum_float(float_array, array_size);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("SIMD:     %f s, Result: %f (%s)\n", cpu_time_used, res_float_simd, (check_double_result(res_float_simd, expected_float) ? "OK" : "FAIL"));

    start = clock();
    double res_float_cache = vsize_cache_friendly_sum_float(float_array, array_size);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("CacheFr:  %f s, Result: %f (%s)\n\n", cpu_time_used, res_float_cache, (check_double_result(res_float_cache, expected_float) ? "OK" : "FAIL"));

    // --- Double Benchmarks ---
    printf("--- Double Benchmarks ---\n");
    double expected_double = sequential_sum_double(double_array, array_size);
    printf("Expected Sum: %f\n", expected_double);

    start = clock();
    double res_double_par = vsize_parallel_sum_double(double_array, array_size);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Parallel: %f s, Result: %f (%s)\n", cpu_time_used, res_double_par, (check_double_result(res_double_par, expected_double) ? "OK" : "FAIL"));

    start = clock();
    double res_double_simd = vsize_simd_sum_double(double_array, array_size);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("SIMD:     %f s, Result: %f (%s)\n", cpu_time_used, res_double_simd, (check_double_result(res_double_simd, expected_double) ? "OK" : "FAIL"));

    start = clock();
    double res_double_cache = vsize_cache_friendly_sum_double(double_array, array_size);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("CacheFr:  %f s, Result: %f (%s)\n\n", cpu_time_used, res_double_cache, (check_double_result(res_double_cache, expected_double) ? "OK" : "FAIL"));

    printf("Cleaning up memory...\n");
    free(int_array);
    free(float_array);
    free(double_array);
    printf("Done.\n");
    return 0;
}