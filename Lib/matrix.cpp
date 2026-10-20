#include "matrix.h"

#include <immintrin.h>
#include <iostream>
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

    // compute remainder column by column
    for(size_t j = col_remainder; j < b.n_columns(); j += 4)
    {
        // first read the 4 columns into contiguous memory
        std::vector<__m256d> cols(b.n_rows());

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