#pragma once
// defined in .h because cython generates .cpp files
#include <cstring>
#include <vector>
#include <memory>
#include <numeric>
#include "./types_and_constants.h"
#include "./group/symmetry_generator.h"
#include "./group/commutation_counter.h"

#include <iostream>
struct Index_Amplitude {
    MATRIX_INDEX_TYPE ind;
    VALUE_TYPE amplitude;
};

struct Indices_Counts {
    vector<MATRIX_INDEX_TYPE> indices;
    vector<int> counts;
};

struct Interval {
    MATRIX_INDEX_TYPE start;
    MATRIX_INDEX_TYPE end;
};

/**
 * Abstract class for handling the relation between 
 * sparse indices, sparse states, lowest states.
 */
class I_State_Index_Amplitude_Translator {
    public:
    virtual ~I_State_Index_Amplitude_Translator() = default;

    virtual size_t get_num_minor_sparse_states() const = 0;
    virtual std::shared_ptr<I_Symmetry_Generator> get_symmetries() const = 0;

    virtual vector<state_int> sparse_index_to_state(MATRIX_INDEX_TYPE sparse_index) const = 0;
        // TODO: could be done with the state_value_hierarchy?

    virtual MATRIX_INDEX_TYPE state_to_sparse_index(const state_int* state) const = 0;
        
    /** 
     * Return major_index if it can support, -1 otherwise.
     * Major index can support lowest states if it can not be lowered by applying symmetries.
     * In some implementations, the compatibility of the anticommutation sign can also be checked here.
     */ // TODO: this could be generalized to more than just the most-major index
    virtual MATRIX_INDEX_TYPE check_major_index_supports_lowest_states(MATRIX_INDEX_TYPE major_index) const = 0;

    /** 
     * Returns a struct containig 
     * 1) a vector of sparse indices of the lowest states that share the major index  
     * and 2) a vector containing the number of sparse states in the linear combination forming the 
     * dense state represented by the sparse index.
     */ // TODO: this could also be generalized to more than just the most-major index
    Indices_Counts get_lowest_sparse_indices_from_major_index(MATRIX_INDEX_TYPE major_index) const {
        std::shared_ptr<I_Symmetry_Generator> symmetries = this->get_symmetries();
        auto shift_periodicities = symmetries->get_shift_periodicities();
        size_t max_num_states = std::reduce(shift_periodicities.begin(), shift_periodicities.end(), size_t{1}, std::multiplies<size_t>()); // TODO: is this true in general?
        vector<State_Amplitude> states_amplitudes (max_num_states, State_Amplitude());

        Interval sparse_index_range = this->get_sparse_index_range_from_major_index(major_index);
        Indices_Counts sparse_indices_counts {std::vector<MATRIX_INDEX_TYPE>(sparse_index_range.end - sparse_index_range.start), 
                                              std::vector<int>(sparse_index_range.end - sparse_index_range.start)};

        MATRIX_INDEX_TYPE allocated = 0;
        for (MATRIX_INDEX_TYPE i=sparse_index_range.start; i < sparse_index_range.end; ++i) {
            vector<state_int> state = this->sparse_index_to_state(i);
            auto num_states = symmetries->get_symmetry_states_from_state(state.data(), true, states_amplitudes.data());
            if (num_states != 0) {
                sparse_indices_counts.indices[allocated] = i;
                sparse_indices_counts.counts[allocated] = num_states;
                allocated += 1;
            }
        }
        sparse_indices_counts.indices.resize(allocated);
        sparse_indices_counts.counts.resize(allocated);
        sparse_indices_counts.indices.shrink_to_fit();
        sparse_indices_counts.counts.shrink_to_fit();
        return sparse_indices_counts;
    }

    /** 
     * The first returned element is included in the range, the second is not.
     * Does not check for validity of the input.
     */// TODO: this could also be generalized to more than just the most-major index
    Interval get_sparse_index_range_from_major_index(MATRIX_INDEX_TYPE major_index) const {
        MATRIX_INDEX_TYPE num_minor_sparse_states = this->get_num_minor_sparse_states();
        return Interval {major_index*num_minor_sparse_states, (major_index+1)*num_minor_sparse_states};
    }

    /**
     * Finds the sparse index and the amplitude of the lowest sparse state in the linear combination of the dense state
     * uniquely determined by the state argument. The linear combination is constructed such that the coefficient of the
     * input state is 1.
     */
    void get_lowest_sparse_amplitude_from_state(const state_int* state, Index_Amplitude& index_amplitude) const {

        std::shared_ptr<I_Symmetry_Generator> symmetries = this->get_symmetries();
        auto shift_periodicities = symmetries->get_shift_periodicities();
        size_t max_num_states = std::accumulate(shift_periodicities.begin(), shift_periodicities.end(), 1, std::multiplies()); // TODO: is this true in general?
        vector<State_Amplitude> states (max_num_states, State_Amplitude());

        size_t num_states = symmetries->get_symmetry_states_from_state(state, false, states.data());

        if (num_states == 0) {
            index_amplitude.ind = -1;
            index_amplitude.amplitude = 0;
            return;
        }

        const state_int* lowest_state = state;
        VALUE_TYPE lowest_c = states[0].amplitude;
        pos_int num_nodes = states[0].length;

        State_Hierarchy state_hierarchy (vector<state_int>{0,1,2});  // FIXME: more general vector. 
        for (size_t i=1; i < num_states; ++i) {
            if (state_hierarchy.check_lower_state(lowest_state, states[i].state, num_nodes) == 2) {
                lowest_state = states[i].state;
                lowest_c = states[i].amplitude;
            }
        }
        index_amplitude.ind = this->state_to_sparse_index(lowest_state);
        index_amplitude.amplitude = lowest_c;
        return;
    }
};
