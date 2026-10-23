#include "matrix.h"
#include "util.h"

#include <immintrin.h>
#include <iostream>
#include <mutex>
#include <sstream>

using size_t = std::size_t;

std::string Matrix::to_string() const
{
    std::stringstream out;
    for(size_t i = 0; i < n_rows_; i++)
    {
        for(size_t j = 0; j < n_columns_; j++)
        {
            out << " " << at(i, j);
        }
        out << '\n';
    }
    return out.str();
}

void Matrix::basic_add(const Matrix& a, const Matrix& b, Matrix& sum)
{
    for(size_t i = 0; i < a.entries_.size(); i++)
    {
        sum.entries_[i] = a.entries_[i] + b.entries_[i];
    }
}

void Matrix::simd_add(const Matrix& a, const Matrix& b, Matrix& sum)
{
    // Each SIMD instruction can add exactly 4 doubles at a time, so
    size_t remainder = (a.entries_.size()) % 4;

    for(size_t i = 0; i < remainder && i < a.entries_.size(); i++)
    {
        sum.entries_[i] = a.entries_[i] + b.entries_[i];
    }

    __m256d block_sum = {0.0, 0.0, 0.0, 0.0};

    // Add remaining entries in blocks of size 4
    for(size_t index = remainder; index < a.entries_.size(); index += 4)
    {
        __m256d a_block = _mm256_loadu_pd(&a.entries_[index]);
        __m256d b_block = _mm256_loadu_pd(&b.entries_[index]);
        block_sum = _mm256_add_pd(a_block, b_block);
        _mm256_storeu_pd(&sum.entries_[index], block_sum);
    }

}

void Matrix::basic_multiply(const Matrix& a, const Matrix& b, Matrix& product)
{
    std::fill(product.entries_.begin(), product.entries_.end(), 0.0);
    for(size_t i = 0; i < a.n_rows(); i++) {
        for(size_t j = 0; j < b.n_columns(); j++) {
            for(size_t k = 0; k < a.n_columns(); k++) {
                product.entries_[product.index(i, j)] += a.at(i, k) * b.at(k, j);
            }
        }
    }
}

// Entry-by-entry matrix multiplication with loop order switched
void Matrix::basic_loop_switched_multiply(const Matrix& a, const Matrix& b, Matrix& product)
{
    product.set_zero();
    for(size_t i = 0; i < a.n_rows(); i++) {
        for(size_t k = 0; k < a.n_columns(); k++) {
            for(size_t j = 0; j < b.n_columns(); j++) {
                product.entries_[product.index(i, j)] += a.at(i, k) * b.at(k, j);
            }
        }
    }
}


void Matrix::cached_column_multiply(const Matrix& a, const Matrix& b, Matrix& product)
{
    // Vector to store each column of b
    std::vector<double> b_col(b.n_rows());

    // Vector to store column values as they are computed, contiguous for cache friendliness
    std::vector<double> prod_col(a.n_rows());

    // Compute product column by column
    for(size_t col = 0; col < b.n_columns(); col++) {
        //copy column into b_col - column is non-contiguous so this loop is about as good as it gets?
        for(size_t i = 0; i < b.n_rows(); i++) {
            b_col[i] = b.at(i, col);
        }

        //set product column to zero, so we can safely accumulate values
        std::fill(prod_col.begin(), prod_col.end(), 0.0);

        // compute this column of the product, row by row
        for(size_t i = 0; i < a.n_rows(); i++)
        {
            for(size_t k = 0; k < a.n_columns(); k++)
            {
                prod_col[i] += a.at(i, k) * b_col[k];
            }
        }

        //copy the computed column to its destination
        for(size_t i = 0; i < a.n_rows(); i++)
        {
            product.set(i, col, prod_col[i]);
        }
    }
}


void Matrix::simd_column_multiply(const Matrix& a, const Matrix& b, Matrix& product)
{
    // compute product row by row, 4 entries at a time
    const int col_remainder = b.n_columns() % 4;

    //do the first entries manually
    for(size_t i = 0; i < a.n_rows(); i++)
    {
        for(size_t j = 0; j < col_remainder && j < b.n_columns(); j++)
        {
            product.set(i, j, 0.0);
            for(size_t k = 0; k < a.n_columns(); k++)
            {
                product.entries_[product.index(i, j)] += a.at(i, k) * b.at(k, j);
            }
        }
    }

    // Vector to hold the 4 columns we are working on
    std::vector<__m256d> cols(b.n_rows());
    for(size_t j = col_remainder; j < b.n_columns(); j += 4)
    {
        // first read the 4 columns into contiguous memory
        //store column sub-matrix in row-major order
        for(size_t k = 0; k < b.n_rows(); k++)
        {
            // load entries j, j+1, j+2, j+3 of row k
            cols[k] = _mm256_loadu_pd(&b.entries_.data()[b.index(k,j)]);
        }

        // compute row by row
        for(size_t i = 0; i < a.n_rows(); i++)
        {
            __m256d product_entries = {0.0, 0.0, 0.0, 0.0};
            for(size_t k = 0; k < a.n_columns(); k++)
            {
                __m256d a_ik = _mm256_set1_pd(a.at(i,k));
                product_entries = _mm256_fmadd_pd(a_ik, cols[k], product_entries);
            }

            _mm256_storeu_pd(&(product.entries_[product.index(i, j)]), product_entries);
        }
    }
}

void Matrix::multiply_block(const Matrix& a, 
                           const Matrix& b, 
                           size_t a_start_row, 
                           size_t a_end_row,
                           size_t a_start_col,
                           size_t a_end_col,
                           size_t b_start_col,
                           size_t b_end_col,
                           Matrix& product)
{
    for(size_t i = a_start_row; i <= a_end_row; i++)
    {
        for(size_t k = a_start_col; k <= a_end_col; k++)
        {
            for(size_t j = b_start_col; j <= b_end_col; j++)
            {
                product.entries_[product.index(i,j)] += a.at(i, k) * b.at(k,j);
            }
        }
    }
}

/*
void Matrix::simd_multiply_block(const Matrix& a, 
                                const Matrix& b, 
                                size_t a_start_row, 
                                size_t a_end_row,
                                size_t a_start_col,
                                size_t a_end_col,
                                size_t b_start_col,
                                size_t b_end_col,
                                Matrix& product)
{

}
*/

void Matrix::cache_block_multiply(const Matrix& a, const Matrix& b, Matrix& product)
{
    constexpr size_t BLOCK_SIZE = 32;

    product.set_zero();

    size_t block_a_row_remainder = a.n_rows() % BLOCK_SIZE;
    size_t block_a_row_quotient = a.n_rows() / BLOCK_SIZE;

    size_t block_a_col_remainder = a.n_columns() % BLOCK_SIZE;
    size_t block_a_col_quotient = a.n_columns() / BLOCK_SIZE;

    size_t block_b_col_remainder = b.n_columns() % BLOCK_SIZE;
    size_t block_b_col_quotient = b.n_columns() / BLOCK_SIZE;

    size_t num_a_block_rows = block_a_row_quotient + (block_a_row_remainder > 0 ?  1 : 0);
    size_t num_a_block_cols = block_a_col_quotient + (block_a_col_remainder > 0 ?  1 : 0);
    size_t num_b_block_cols = block_b_col_quotient + (block_b_col_remainder > 0 ?  1 : 0);

    for(size_t i = 0; i < num_a_block_rows; i++)
    {
        for(size_t k = 0; k < num_a_block_cols; k++)
        {
            for(size_t j = 0; j < num_b_block_cols; j++)
            {     
                size_t a_start_row = i * BLOCK_SIZE;
                size_t a_end_row = std::min(a_start_row + BLOCK_SIZE - 1, a.n_rows() - 1);

                size_t a_start_col = k * BLOCK_SIZE;
                size_t a_end_col = std::min(a_start_col + BLOCK_SIZE - 1, a.n_columns() - 1);
                
                size_t b_start_col = j * BLOCK_SIZE;
                size_t b_end_col = std::min(b_start_col + BLOCK_SIZE - 1, b.n_columns() - 1);

                multiply_block(a, b, a_start_row, a_end_row, a_start_col, a_end_col, b_start_col, b_end_col, product);
            }
            
        }
    }
}

void Matrix::basic_multiply_compute_columns(const Matrix& a, const Matrix& b, Matrix& product, size_t min, size_t max)
{
    for(size_t i = 0; i < a.n_rows(); i++) {
        for(size_t j = min; j <= max; j++) {
            product.entries_[product.index(i, j)] = 0.0;
            for(size_t k = 0; k < a.n_columns(); k++) {
                product.entries_[product.index(i, j)] += a.at(i, k) * b.at(k, j);
            }
        }
    }
}

// Basic entry-by-entry matrix multiplication, but multithreaded
void Matrix::mt_basic_multiply(const Matrix&a, const Matrix& b, Matrix& product, ThreadPool& thread_pool)
{
    auto compute_columns_task = [&](size_t thread_index, size_t n_threads) {
        IndexPair column_interval = get_interval(b.n_columns(), thread_pool.num_threads(), thread_index);
        basic_multiply_compute_columns(a, b, product, column_interval.start, column_interval.end);
    };

    thread_pool.set_function(compute_columns_task);
    thread_pool.execute();
}

void Matrix::simd_multiply_compute_columns(const Matrix& a, const Matrix& b, Matrix& product, size_t min, size_t max)
{
    // compute product row by row, 4 entries at a time
    const int col_remainder = (max - min + 1) % 4;

    //do the first entries manually
    for(size_t i = 0; i < a.n_rows(); i++)
    {
        for(size_t j = min; j < min + col_remainder && j <= max; j++)
        {
            product.set(i, j, 0.0);
            for(size_t k = 0; k < a.n_columns(); k++)
            {
                product.entries_[product.index(i, j)] += a.at(i, k) * b.at(k, j);
            }
        }
    }

    // Vector to hold columns that are being worked on
    std::vector<__m256d> cols(b.n_rows());
    for(size_t j = min + col_remainder; j <= max; j += 4)
    {
        // first read the 4 columns into contiguous memory
        //store column sub-matrix in row-major order
        for(size_t k = 0; k < b.n_rows(); k++)
        {
            // load entries j, j+1, j+2, j+3 of row k
            cols[k] = _mm256_loadu_pd(&b.entries_.data()[b.index(k,j)]);
        }

        // compute row by row
        for(size_t i = 0; i < a.n_rows(); i++)
        {
            __m256d product_entries = {0.0, 0.0, 0.0, 0.0};
            for(size_t k = 0; k < a.n_columns(); k++)
            {
                __m256d a_ik = _mm256_set1_pd(a.at(i,k));
                product_entries = _mm256_fmadd_pd(a_ik, cols[k], product_entries);
            }

            _mm256_storeu_pd(&(product.entries_[product.index(i, j)]), product_entries);
        }
    }
}

void Matrix::mt_simd_column_multiply(const Matrix&a, const Matrix& b, Matrix& product, ThreadPool& thread_pool)
{
    auto compute_columns_task = [&](size_t thread_index, size_t n_threads) {
        IndexPair column_interval = get_interval(b.n_columns(), thread_pool.num_threads(), thread_index);
        simd_multiply_compute_columns(a, b, product, column_interval.start, column_interval.end);
    };

    thread_pool.set_function(compute_columns_task);
    thread_pool.execute();
}

void Matrix::open_blas_multiply(const Matrix& a, const Matrix& b, Matrix& product)
{
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        a.n_rows(),
        b.n_columns(),
        a.n_columns(),
        1.0,
        a.entries_.data(),
        a.n_columns(),
        b.entries_.data(),
        b.n_columns(),
        0.0,
        product.entries_.data(),
        product.n_columns()
    );
}