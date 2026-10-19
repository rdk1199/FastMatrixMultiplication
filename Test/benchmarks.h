#ifndef __BENCHMARKS_H__
#define __BENCHMARKS_H__

#include <iostream>
#include <chrono>

#include <matrix.h>

void test_add()
{
    // Prime numbers, to try and find any weird edge cases regarding boundaries
    size_t rows = 2243;
    size_t columns = 3209;

    Matrix a(rows, columns, [](size_t r, size_t c) {return static_cast<double>(r) - static_cast<double>(c);});
    Matrix b(rows, columns, [](size_t r, size_t c) {return static_cast<double>(r) * static_cast<double>(c);});
    Matrix c(rows, columns);

    auto start = std::chrono::high_resolution_clock::now();

    Matrix::basic_add(a, b, c);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start);
    std::cout << "basic_add : Time elapsed (ns): " << duration.count() << std::endl;
}

void test_multiply()
{
    size_t a_rows = 251;
    size_t a_cols = 193;
    size_t b_cols = 311;

    Matrix a(a_rows, a_cols, [](size_t r, size_t c) {return 2 * r -  static_cast<double>(c);});
    Matrix b(a_cols, b_cols, [](size_t r, size_t c){return r + c;});
    Matrix product(a_rows, b_cols);


    auto start = std::chrono::high_resolution_clock::now();

    Matrix::basic_multiply(a, b, product);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end -start);
    std::cout << "basic_multiply: Time elapsed (ns): " << duration.count() << std::endl;
}



#endif
