#pragma once
#include <algorithm>
#include <vector>
#include "./ndarray.h"
#include "./commutation_counter.h"
#include "./permutation.h"
#include "../types_and_constants.h"
#include <iostream>

template<typename T>
void shift(T& element, size_t length) {
    element = (element+1) % length;
}

Permutations get_rectangular_cluster(pos_int x_length, pos_int y_length, 
                                     const vector<float>& quantum_numbers,
                                     const State_Hierarchy& hierarchy,
                                     const Commutation_Counter_Factory& commutation_counter_factory) {
    // with sides parallel to elementary translations
    // can we do num_spin_options with None signifying that spins are treated with a value and not a repetition of the spatial nodes?
    pos_int basis_length = x_length*y_length;
    vector<pos_int> xs (x_length);
    vector<pos_int> ys (y_length);
    std::iota(xs.begin(), xs.end(), 0);
    std::iota(ys.begin(), ys.end(), 0);
    ndarray<pos_int> shifts_to_index = ndarray<pos_int>::meshindices(vector{xs, ys});

    ndarray<pos_int> index_maps (vector<size_t>{2, static_cast<size_t>(basis_length)});
    for (size_t i=0; i < basis_length; ++i) {
        index_maps[vector<size_t>{0, i}] = (i+y_length) % basis_length;
        if (i%2 == 0) {
            index_maps[vector<size_t>{1, i}] = i+1;
        } else {
            index_maps[vector<size_t>{1, i}] = i-1;
        }
    } 

    size_t num_shifts = 2;
    size_t internal_stride = 1;
    return Permutations(num_shifts, basis_length, internal_stride, 
                                    shifts_to_index, index_maps, 
                                    quantum_numbers,
                                    hierarchy, commutation_counter_factory);
}
