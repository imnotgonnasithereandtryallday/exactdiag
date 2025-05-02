#pragma once
#include <vector>
#include <cmath>
#include "./types_and_constants.h"

template <typename T>
T dot_product_without_conj(const T* vec1, const T* vec2, size_t length) {
    T summ {};
    for (size_t i=0; i < length; ++i) {
        summ += vec1[i] * vec2[i];
    }
    return summ;
}

template <typename T>
pos_int find_first(const std::vector<T>& array, const T& value) { 
    // TODO: this definitely already exists in std.
    // TODO: just return size_t and let the caller check that -1 properly
    //       or return a struct with bool and index.
    // FIXME: return option type
    // assumes the vector's length fits pos_int. 
    // returns -1 if not found
    for (pos_int i=0; i < array.size(); ++i) {
        if (value == array[i]) {
            return i;
        }
    }
    return -1;
}



inline int python_mod(int dividend, int divisor) { // FIXME: repeated in symmetry.h
    // c modulo treats negative numbers differently than python.
    // branchless shift to a positive number
    int remainder = dividend % divisor;
    remainder += divisor * (remainder < 0);
    int sign = 1 - 2*std::signbit((double) divisor);
    return remainder * sign;
}