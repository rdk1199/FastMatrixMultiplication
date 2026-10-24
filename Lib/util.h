#ifndef __UTIL_H__
#define __UTIL_H__

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdio.h>

struct MatrixCoord
{
    size_t i;
    size_t j;
};

struct IndexPair
{
    size_t start;
    size_t end;
};

// get the start and end of the i'th interval, where we divide total into num_intervals as evenly as possible
constexpr IndexPair get_interval(size_t total, size_t num_intervals, size_t i)
{
    size_t quotient = total / num_intervals;
    size_t remainder = total % num_intervals;
    size_t interval_size = (i < remainder) ? quotient + 1 : quotient;

    size_t start = std::min(remainder, i) * (quotient + 1) + 
    std::max((static_cast<int64_t>(i) - static_cast<int64_t>(remainder)), int64_t{0}) * quotient;

    return {start, start + interval_size - 1};
}

#endif