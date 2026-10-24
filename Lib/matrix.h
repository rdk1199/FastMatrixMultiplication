#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <cblas.h>

#include "thread_pool.h"
#include "util.h"

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

    inline void set_zero()
    {
        std::fill(entries_.begin(), entries_.end(), 0.0);
    }

    // All of these assume that the input matrices are compatible and the result matrices are properly sized

    // Basic entry-by-entry matrix addition
    static void basic_add(const Matrix& a, const Matrix& b, Matrix& sum);

    // Unaligned SIMD add
    static void simd_add(const Matrix& a, const Matrix& b, Matrix& sum);

    // Basic entry-by-entry matrix multiplication using the standard method
    static void basic_multiply(const Matrix& a, const Matrix& b, Matrix& product);

    // Entry-by-entry matrix multiplication with loop order switched
    static void basic_loop_switched_multiply(const Matrix& a, const Matrix& b, Matrix& product);

    // Basic entry-by-entry matrix multiplication using the standard method, but columns of b are copied into contiguous memory
    // to improve cache friendliness
    static void cached_column_multiply(const Matrix& a, const Matrix& b, Matrix& product);

    // SIMD multiply
    static void simd_column_multiply(const Matrix& a, const Matrix& b, Matrix& product);


    

    // Cache block multiply
    static void cache_block_multiply(const Matrix& a, const Matrix& b, Matrix& product);

    // SIMD Cache block multiply
    static void simd_cache_block_multiply(const Matrix& a, const Matrix& b, Matrix& product);

    // Basic entry-by-entry matrix multiplication, but multithreaded
    static void mt_basic_multiply(const Matrix& a, const Matrix& b, Matrix& product, ThreadPool& thread_pool);

    // Multithreaded matrix multiplication using SIMD to traverse multiple columns at once
    static void mt_simd_column_multiply(const Matrix& a, const Matrix& b, Matrix& product, ThreadPool& thread_pool);

    // Multithreaded SIMD cache blocked multiplication
    static void mt_simd_cache_block_multiply(const Matrix& a, const Matrix& b, Matrix& product, ThreadPool& thread_pool);

    // For comparison: OpenBLAS multiply:
    static void open_blas_multiply(const Matrix& a, const Matrix& b, Matrix& product);
    

private:

    // Compute columns of product from min to max inclusive using basic entry-by-entry multiplication
    static void basic_multiply_compute_columns(const Matrix& a, const Matrix& b, Matrix& product, size_t min, size_t max); 

    // Compute columns of product from min to max inclusive using SIMD
    static void simd_multiply_compute_columns(const Matrix& a, const Matrix& b, Matrix& product, size_t min, size_t max);

    // compute the product of sub-blocks of a and b and place them in the appropriate entries in product
    static inline void multiply_block(const Matrix& a, 
                               const Matrix& b, 
                               size_t a_start_row, 
                               size_t a_end_row,
                               size_t a_start_col,
                               size_t a_end_col,
                               size_t b_start_col,
                               size_t b_end_col,
                               Matrix& product);

    static inline void simd_multiply_block(const Matrix& a, 
                               const Matrix& b, 
                               size_t a_start_row, 
                               size_t a_end_row,
                               size_t a_start_col,
                               size_t a_end_col,
                               size_t b_start_col,
                               size_t b_end_col,
                               Matrix& product);

    static inline void simd_multiply_blocks(const Matrix& a, 
                                            const Matrix& b,
                                            Matrix& product, 
                                            size_t num_a_block_rows,
                                            size_t num_a_block_cols,
                                            size_t num_b_block_cols,
                                            size_t block_size,
                                            const std::vector<MatrixCoord>& blocks);



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
