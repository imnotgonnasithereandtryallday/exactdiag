#pragma once
// defined in .h because cython generates .cpp files
#include <vector>
#include <functional>
#include <iostream>
#include <numeric>
using std::vector;
using std::unique_ptr;
using std::make_unique;
using std::malloc; 

template <typename T>
struct ndarray {
    // last dimension is contiguous
    size_t ndim;
    size_t size;
    vector<size_t> shape;
    vector<T> data;

    ndarray(const size_t size) {
        // refactor -- call the other constructors
        this->ndim = 1;
        this->shape = vector<size_t>(1, size);
        this->size = size;
        this->data = vector<T>(size);
    }

    ndarray(const vector<size_t>& shape) {
        this->ndim = shape.size();
        this->shape = shape;
        this->size = std::reduce(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
        this->data = vector<T>(this->size);
    }

    ndarray(const vector<size_t>& shape, const T* data) 
        : ndarray (shape)
    {    
        std::memcpy(this->data.data(), data, this->size * sizeof(T));
    }
    ndarray(const vector<size_t>& shape, const vector<T>& data) : ndarray(shape, data.data()) {}
    ndarray(const vector<size_t>& shape, const ndarray<T>& that) : ndarray(shape, that.data.data()) {}
    ndarray(const ndarray<T>& that) : ndarray(that.shape, that.data.data()) {}

    ndarray(ndarray<T>&& that) {
        this->ndim = that.ndim;
        this->shape = std::move(that.shape);
        this->size = that.size;
        this->data = std::move(that.data);
    }


    static ndarray<T> meshindices(const vector<vector<T>>& vectors) {
    // Constructs ndarray of shape [product([v.size() for vector in vectors])]*vectors.size()]
    // could be variadic after compilers support pack indexing
        size_t length = 1;
        size_t ndim = vectors.size();
        for (const vector<T>& vec : vectors) {
            length *= vec.size();
        }
        size_t size = std::pow(length, ndim);
        vector<size_t> shape (ndim, length);

        vector<size_t> data_strides (ndim);
        for (size_t i=0; i < ndim; ++i) {
            data_strides[i] = std::pow(length, ndim-i-1);
        }
        // TODO: merge the loops
        vector<size_t> indices_strides (ndim, 1);
        for (size_t i=1; i < ndim; ++i) {
            indices_strides[ndim-1-i] = indices_strides[ndim-i] * vectors[ndim-i].size();
        }
        vector<T> data (size, 0);
        for (size_t i=0; i < size; ++i) {
            size_t remaining_index = i;
            for (size_t vector_index=0; vector_index < ndim; ++vector_index) {
                size_t element_index = (remaining_index / data_strides[vector_index]) % vectors[vector_index].size();
                remaining_index -= remaining_index / data_strides[vector_index] * data_strides[vector_index];
                data[i] += indices_strides[vector_index] * vectors[vector_index][element_index];
            }
        }
        return ndarray<T>(std::move(shape), std::move(data));
    }

    ndarray<T>& operator=(const ndarray<T>& that) {
        if (this == &that){
            return *this;
        }
        this->ndim = that.ndim;
        this->shape = vector<size_t>(that.shape);
        this->size = that.size;
        this->data = vector<T>(that.data);
        return *this;
    }

    ndarray<T>& operator=(ndarray<T>&& that) {
        if (this == &that){
            return *this;
        }
        this->ndim = that.ndim;
        this->shape = std::move(that.shape);
        this->size = that.size;
        this->data = std::move(that.data);
        return *this;
    }

    T& operator[] (size_t index) { 
        return this->data[index];
    }
    const T& operator[] (size_t index) const {  
        return this->data[index];
    }

    // In this ancient version of C++, operator[] expects one argument and just ignores the rest.
    T& operator[] (vector<size_t> indices) {
        size_t i = this->get_flattened_index(indices);    
        return this->data[i];
    }
    const T& operator[] (vector<size_t> indices) const { 
        size_t i = this->get_flattened_index(indices);    
        return this->data[i];
    }


    size_t get_flattened_index(const vector<size_t>& indices) const {
        size_t index = 0;
        size_t remaining_size = this->size;
        for (size_t i=0; i < indices.size(); ++i) {
            size_t base = remaining_size / this->shape[i];
            index += base * indices[i];
            remaining_size = base;
        }
        return index;
    }


    template <std::convertible_to<size_t>... T2>
    size_t get_flattened_index(const T2&... indices) const {
        vector<size_t> remaining_shape_reversed = vector<size_t>(this->shape.rend(), this->shape.rbegin()); // FIXME: this should be rbegin, rend?
        return _flattened_index_iteration(this->size, remaining_shape_reversed, indices...);
    }
    template <std::convertible_to<size_t>... T2>
    size_t _flattened_index_iteration(size_t remaining_size, const vector<size_t>& remaining_shape_reversed, size_t index, const T2&... indices) const {
        size_t least_significant_length = remaining_shape_reversed.back();
        remaining_shape_reversed.pop_back();
        size_t base = remaining_size / least_significant_length;
        return base * index + _flattened_index_iteration(base, remaining_shape_reversed, indices...);
    }
    size_t _flattened_index_iteration(size_t base, const vector<size_t>& remaining_shape_reversed, size_t index) const {
        return base * index;
    }


    const vector<T>& flattened() const {
        return this->data;
    }

    vector<size_t> argwhere_first(const T& val) const {
        size_t flat_index = std::find(this->data.begin(), this->data.end(), val) - this->data.begin();
        vector<size_t> indices = vector<size_t>(this->ndim); 
        size_t base = this->size;
        std::lldiv_t quot_rem {0LL, static_cast<long long>(flat_index)}; // TODO: chceck if flat_index (unsigned) fits in std::intmax_t (signed)!
        for (int j=0; j<this->ndim; ++j) {
            base /= this->shape[j];
            quot_rem = std::div(quot_rem.rem, base);
            indices[j] = quot_rem.quot;
        }
        return indices;
    }
};
    