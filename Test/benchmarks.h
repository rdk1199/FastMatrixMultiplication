#ifndef __BENCHMARKS_H__
#define __BENCHMARKS_H__

#include <iostream>
#include <chrono>
#include <functional>

#include <matrix.h>
#include <thread_pool.h>

void test_add()
{
    // Prime numbers, to try and find any weird edge cases regarding boundaries
    size_t rows = 2243;
    size_t columns = 3209;

    Matrix a(rows, columns, [](size_t r, size_t c) {return static_cast<double>(r) - static_cast<double>(c);});
    Matrix b(rows, columns, [](size_t r, size_t c) {return static_cast<double>(r) * static_cast<double>(c);});
    
    // basic add
    Matrix basic_add_sum(rows, columns);

    auto start = std::chrono::high_resolution_clock::now();
    Matrix::basic_add(a, b, basic_add_sum);
    auto basic_add = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(basic_add -start);
    std::cout << "basic_add : Time elapsed (ns): " << duration.count() << std::endl;

   // simd_add
    Matrix simd_add_sum(rows, columns);

    auto simd_add_start = std::chrono::high_resolution_clock::now();
    Matrix::simd_add(a,b, simd_add_sum);
    auto simd_add_end = std::chrono::high_resolution_clock::now();

    auto simd_add_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(simd_add_end - simd_add_start);
    std::cout << "simd_add : Time elapsed (ns) : " << simd_add_duration.count() << std::endl;

    assert(simd_add_sum == basic_add_sum);
}

void test_multiply()
{
    size_t a_rows = 449;
    size_t a_cols = 563;
    size_t b_cols = 311;

    Matrix a(a_rows, a_cols, [](size_t r, size_t c) {return 2 * r -  static_cast<double>(c);});
    Matrix b(a_cols, b_cols, [](size_t r, size_t c){return r + c;});
    Matrix product(a_rows, b_cols);


    auto start = std::chrono::high_resolution_clock::now();

    Matrix::basic_multiply(a, b, product);

    auto basic_multiply_end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(basic_multiply_end -start);
    std::cout << "basic_multiply: Time elapsed (ns): " << duration.count() << std::endl;


    Matrix basic_loop_switched_product(a_rows,  b_cols);

    auto basic_loop_switched_start = std::chrono::high_resolution_clock::now();
    Matrix::basic_loop_switched_multiply(a, b, basic_loop_switched_product);
     auto basic_loop_switched_end = std::chrono::high_resolution_clock::now();

    auto basic_loop_switched_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(basic_loop_switched_end - basic_loop_switched_start);
    std::cout << "basic_loop_switched_multiply: Time elapsed (ns): " << basic_loop_switched_duration.count() << std::endl;


    Matrix cached_col_product(a_rows,  b_cols);

    auto cached_col_start = std::chrono::high_resolution_clock::now();
    Matrix::cached_column_multiply(a, b, cached_col_product);
     auto cached_col_end = std::chrono::high_resolution_clock::now();

    auto cached_col_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(cached_col_end - cached_col_start);
    std::cout << "cached_col_multiply: Time elapsed (ns): " << cached_col_duration.count() << std::endl;


    Matrix simd_col_product(a_rows, b_cols);

    auto simd_col_start = std::chrono::high_resolution_clock::now();
    Matrix::simd_column_multiply(a, b, simd_col_product);
    auto simd_col_end = std::chrono::high_resolution_clock::now();

    auto simd_col_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(simd_col_end - simd_col_start);
    std::cout << "simd_col_multiply: Time elapsed (ns): " << simd_col_duration.count() << std::endl;


    Matrix cache_block_product(a_rows, b_cols);

    auto cache_block_start = std::chrono::high_resolution_clock::now();
    Matrix::cache_block_multiply(a, b, cache_block_product);
    auto cache_block_end = std::chrono::high_resolution_clock::now();

    auto cache_block_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(cache_block_end - cache_block_start);
    std::cout << "cache_block_multiply: Time elapsed (ns): " << cache_block_duration.count() << std::endl;


    //Multithreaded tests
    ThreadPool thread_pool(std::thread::hardware_concurrency());

    Matrix mt_basic_product(a_rows, b_cols);

    auto mt_basic_start = std::chrono::high_resolution_clock::now();
    Matrix::mt_basic_multiply(a, b, mt_basic_product, thread_pool);
    auto mt_basic_end = std::chrono::high_resolution_clock::now();

    auto mt_basic_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(mt_basic_end - mt_basic_start);
    std::cout << "mt_basic_multiply: Time elapsed (ns): " << mt_basic_duration.count() << std::endl;

    Matrix mt_simd_product(a_rows, b_cols);

    auto mt_simd_start = std::chrono::high_resolution_clock::now();
    Matrix::mt_simd_column_multiply(a, b, mt_simd_product, thread_pool);
    auto mt_simd_end = std::chrono::high_resolution_clock::now();

    auto mt_simd_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(mt_simd_end - mt_simd_start);
    std::cout << "mt_simd_multiply: Time elapsed (ns): " << mt_simd_duration.count() << std::endl;

    assert(product == cached_col_product);
    assert(product == cache_block_product);
    assert(product == simd_col_product);
    assert(product == mt_basic_product);
    assert(product == mt_simd_product);
}



#endif
