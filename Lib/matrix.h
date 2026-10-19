#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

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

    // Debug: write matrix as string - each row separated by newline
    std::string to_string() const;

    // Operations:

    // All of these assume that the input matrices are compatible and the result matrices are properly sized

    // Basic entry-by-entry matrix addition
    static void basic_add(const Matrix& a, const Matrix& b, Matrix& sum);

    // Basic entry-by-entry matrix multiplication using the standard method
    static void basic_multiply(const Matrix& a, const Matrix& b, Matrix& product);

    bool operator==(const Matrix& other)
    {
        return n_rows_ == other.n_rows() && 
               n_columns_ == other.n_columns() &&
               (std::memcmp(entries_.data(), other.entries_.data(), n_rows_*n_columns_) == 0 );
    }

private:
    size_t n_rows_;
    size_t n_columns_;

    // Compute the flat index of an entry given the row and column
    inline size_t index(size_t row, size_t col) const;

    // row major flat pack
    std::vector<double> entries_;
};


#endif
