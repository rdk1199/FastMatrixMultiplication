#include "matrix.h"

#include <sstream>

using size_t = std::size_t;

size_t Matrix::index(size_t row, size_t col) const
{
    return row * n_columns_ + col;
}

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

void Matrix::basic_multiply(const Matrix& a, const Matrix& b, Matrix& product)
{
    for(size_t i = 0; i < a.n_rows(); i++) {
        for(size_t j = 0; j < b.n_columns(); j++) {
            product.set(i, j, 0.0);
            for(size_t k = 0; k < a.n_columns(); k++) {
                product.entries_[product.index(i, j)] += a.at(i, k) * b.at(k, j);
            }
        }
    }
}