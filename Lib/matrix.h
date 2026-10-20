#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "thread_pool.h"

// Class to represent a matrix of doubles
class Matrix
{

using size_t = std::size_t;

public:
    // Initialize matrix of all zeroes
    Matrix(size_t n_rows, size_t n_columns) :
        n_rows_(n_rows),
        n_columns_(n_columns)
    {
        // Resize the vector to the proper size - will auto-fill with zeros
        entries_.resize(n_rows_ * n_columns_);
    }

    // Initialize matrix given entries - destroys original vector
    Matrix(size_t n_rows, size_t n_columns, std::vector<double>& entries):
        n_rows_(n_rows),
        n_columns_(n_columns),
        entries_(std::move(entries))
    {
        // Truncate overly long vectors, pad short ones with zeroes
        entries.resize(n_rows * n_columns);
    }

    // Initialize matrix given a function to compute the entries from the row + column index
    Matrix(size_t n_rows, size_t n_columns, const std::function<double(size_t row, size_t col)> func):
        n_rows_(n_rows),
        n_columns_(n_columns)
    {
        entries_.resize(n_rows_ * n_columns_);

        for(size_t i = 0; i < n_rows_; i++)
        {
            for(size_t j = 0; j < n_columns_; j++)
            {
                entries_[index(i, j)] = func(i, j);
            }
        }
    }

    inline constexpr size_t n_rows() const {return n_rows_;}
    inline constexpr size_t n_columns() const {return n_columns_;}

    // Entry-wise getter
    inline constexpr double at(size_t row, size_t col) const {
        return entries_[index(row, col)];
    }

    // Entry-wise setter
    inline constexpr void set(size_t row, size_t col, double val) {
        entries_[index(row,col)] = val;
    }

    // Get row as raw array of doubles
    inline constexpr double* get_row(size_t row) {
        return &entries_[index(row, 0)];
    }

    // Check value equality
    bool operator==(const Matrix& other) const
    {
       return n_rows_ == other.n_rows() && 
              n_columns_ == other.n_columns() &&
              (std::memcmp(entries_.data(), other.entries_.data(), n_rows_*n_columns_) == 0 );

       return true;
    }

    // Debug: write matrix as string - each row separated by newline
    std::string to_string() const;

    // Operations:

    // All of these assume that the input matrices are compatible and the result matrices are properly sized

    // Basic entry-by-entry matrix addition
    static void basic_add(const Matrix& a, const Matrix& b, Matrix& sum);

    // Unaligned SIMD add
    static void simd_add(const Matrix& a, const Matrix& b, Matrix& sum);

    // Basic entry-by-entry matrix multiplication using the standard method
    static void basic_multiply(const Matrix& a, const Matrix& b, Matrix& product);

    // Basic entry-by-entry matrix multiplication using the standard method, but columns of b are copied into contiguous memory
    // to improve cache friendliness
    static void cached_column_multiply(const Matrix& a, const Matrix& b, Matrix& product);

    // SIMD multiply
    static void simd_column_multiply(const Matrix& a, const Matrix& b, Matrix& product);


    // Compute columns of product from min to max inclusive using basic entry-by-entry multiplication
    static void basic_multiply_compute_columns(const Matrix& a, const Matrix& b, Matrix& product, size_t min, size_t max);

    // Basic entry-by-entry matrix multiplication, but multithreaded
    template<size_t num_threads>
    static void mt_basic_multiply(const Matrix&a, const Matrix& b, Matrix& product, ThreadPool<num_threads>& thread_pool)
    {
        size_t num_columns = b.n_columns();

        size_t columns_per_thread = num_columns / num_threads;
        size_t column_remainder = num_columns % num_threads;

        auto compute_columns_task = [&](size_t thread_index, size_t n_threads) {
            
            // Helper lambda to compute the number of columns a given thread will compute
            // The first remainder columns will each take one thread from those remaining after the division -
            // the rest of the columns will take only columns_per_thread
            auto num_columns_of_thread_index = [&](size_t p_thread_index) {
                return (p_thread_index < column_remainder) ? columns_per_thread + 1 : columns_per_thread;
            };
            
            // Number of columns this thread will compute
            size_t num_columns = num_columns_of_thread_index(thread_index);

            // Number of columns before this one already computed in other threads
            // TODO: There's definitely a slicker way to do this
            size_t num_columns_already_computed = 0;
            for(size_t i = 0; i < thread_index; i++)
            {
                num_columns_already_computed += num_columns_of_thread_index(i);
            }

            // The first non-computed column
            size_t start = num_columns_already_computed;

            // The last column we will compute
            size_t end = start + num_columns - 1;

            basic_multiply_compute_columns(a, b, product, start, end);
        };

        thread_pool.set_function(compute_columns_task);
        thread_pool.execute();
    }

private:
    size_t n_rows_;
    size_t n_columns_;

    // Compute the flat index of an entry given the row and column
    inline constexpr size_t index(size_t row, size_t col) const
    {
        return row * n_columns_ + col;
    }

    // row major flat pack
    std::vector<double> entries_;
};


#endif
