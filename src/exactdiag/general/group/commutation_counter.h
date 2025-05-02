#pragma once
// defined in .h because cython generates .cpp files
#include <vector>
#include <initializer_list>
#include "../types_and_constants.h"
#include "./ndarray.h"
using std::vector;
using std::find;

// state_int can be templated. should it?


inline int lower_single_check(state_int state1, state_int state2, state_int preferred_value) {
    // return 1 if state1 is lower or 2 if state2 is prefered, 0 otherwise.
    if (state1 == preferred_value and state2 != preferred_value) {
        return 1;
    }
    if (state1 != preferred_value and state2 == preferred_value) {
        return 2;
    }
    return 0;
}

class State_Hierarchy {
    public:
    vector<state_int> state_values_hierarchy;
    size_t num_values;

    State_Hierarchy(const vector<state_int>& state_values_hierarchy) 
        : state_values_hierarchy(state_values_hierarchy)
        , num_values(this->state_values_hierarchy.size())
    {}
    State_Hierarchy(const State_Hierarchy& state_values_hierarchy) 
    : State_Hierarchy(state_values_hierarchy.state_values_hierarchy)
    {}
    State_Hierarchy(std::initializer_list<state_int> state_values_hierarchy) 
        : state_values_hierarchy(state_values_hierarchy)
        , num_values(this->state_values_hierarchy.size())
    {}

    state_int operator[] (size_t index) const {
        return this->state_values_hierarchy[index];
    }

    int check_lower_state(const state_int* state1, const state_int* state2, int state_length) const { // TODO: could be a simple memcmp if values in the hierarchy were ordered?
        // Compares the sparse indices of two states
        // (and by construction also dense indices if the two states are both the lowest sparse state in the respective dense index).
        // Returns 0 if the indices are equal, 1 if state1 is lower, 2 otherwise.
        // We assume that there is a hierarchy of values -- each value in the hierarchy has its positions in the state determined by the
        // corresponding partial index (e.g. the most-major index for the hierarchy[0] value and the most-minor index for the last value in the hierarchy),
        // such that the positions at the end of the state are preferred (if not already occupied by more-major value).
        // We also assume, that the count of each value is the same between the states.
        // TODO: enum of return values
        for (size_t k=0; k<this->num_values-1; ++k) {
            // the last value has to fill in the remaining gaps
            for (pos_int j=0; j<state_length; ++j) {
                pos_int i = state_length - 1 - j;
                int retval = lower_single_check(state2[i], state1[i], this->state_values_hierarchy[k]);
                // The states are swapped on purpose: we go back to front,
                // if the less significant position has lower value,
                // that low value must be missing at a more-significant place, 
                // so the node having lower value means the state is higher.
                if (retval != 0) {
                    return retval;
                }
            }
        }
        return 0;
    }
};


class Commutation_Counter {
    // TODO: should it not only count commutations but also construct the state at the same time?
    // TOOD: should it distinguish between count equal and count unequal?
    // Counts (anti-)commutations by building a state from (pos_int, state_int) pairs.
    // The pair specifies a creation operator.
    // The added operators act on an empty ket in the order they were added in,
    // i. e., the left-most operator is added last.
    // Does not distinguish the commutations, e. g. between identical / different operators.
    // The number of operators must not be higher than max_indices.
    public:
    vector<state_int> counted_values;
    bool commutation_parity;
    size_t num_indices;
    vector<pos_int> ac_indices;
    pos_int stride;

    Commutation_Counter(const vector<state_int>& counted_values, size_t max_indices, pos_int stride) {
        this->counted_values = counted_values;
        this->commutation_parity = false;
        this->num_indices = 0;
        this->ac_indices = vector<pos_int>(max_indices);
        this->stride = stride;
    }

    bool is_counted_value(const state_int value) const {
        return find(this->counted_values.begin(), this->counted_values.end(), value) != this->counted_values.end();
    }
    
    void add(const pos_int position_index, const state_int* values) {
        // Add values[:stride] to positions position_index:position_index+stride.
        // len(values) >= self.stride is assumed.
        bool is_num_counted_even = true;
        pos_int initial_num_indices = this->num_indices;

        for (pos_int i=0; i<this->stride; ++i) {
            bool is_counted = this->is_counted_value(values[i]);
            is_num_counted_even ^= is_counted;
            if (is_counted) {
                this->ac_indices[this->num_indices] = position_index+i;
                this->num_indices += 1; 
            }
        }

        if (is_num_counted_even) {
            return;
        }
        
        for (pos_int pos=0; pos < initial_num_indices; ++pos) {
            this->commutation_parity ^= this->ac_indices[pos] > position_index;
        }
    }
};


class Commutation_Counter_Factory { 
    public:
    vector<state_int> counted_values;
    Commutation_Counter_Factory(const vector<state_int>& counted_values) 
        : counted_values(counted_values) 
    {}
    Commutation_Counter_Factory(const Commutation_Counter_Factory& factory) 
        : Commutation_Counter_Factory(factory.counted_values) 
    {}
    Commutation_Counter_Factory(std::initializer_list<state_int> counted_values) 
        : counted_values(counted_values.begin(), counted_values.end()) 
    {}

    Commutation_Counter get(const size_t max_indices, const pos_int stride) const {
        return Commutation_Counter(this->counted_values, max_indices, stride);
    }
};
