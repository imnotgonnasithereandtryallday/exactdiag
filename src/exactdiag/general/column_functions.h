#pragma once
#include <vector>
#include <memory>
#include "./types_and_constants.h"
#include "./basis_indexing.h"
#include "./symmetry.h"

using std::vector;

/**
 * This module contains functions used during calculation of matrix elements.
 * Elements are calculated column by column:
 *     A set of operators is applied to an initial basis state and valid results are collected.
 *     The collected final basis state indices and amplitudes are collected, potentially merged, and sorted.
 * The Lambdas were introduced because Cython does not support c++ closures.
 */

typedef bool is_valid_indices_type(MATRIX_INDEX_TYPE initial_index, MATRIX_INDEX_TYPE final_index);
typedef bool is_valid_state_type(const state_int* state, const pos_int* indices, int length); // FIXME: why int here?
typedef void calculate_weights_type(const state_int* state, const vector<pos_int>& operator_indices, VALUE_TYPE& explicit_diagonal_weight, VALUE_TYPE& off_diagonal_weight);
typedef void get_new_state_type(const state_int* state, state_int* new_state, pos_int state_length, const vector<pos_int>& operator_indices);
typedef int count_commutations_type(const state_int* state, pos_int state_length, const vector<pos_int>& operator_indices);


/**
 * A collection of operations that transform a state.
 * Populates the pointers when called. 
 */
class I_Lambda_Column {
    public:
    virtual ~I_Lambda_Column() = default;
    virtual void operator() (MATRIX_INDEX_TYPE initial_index, VALUE_TYPE* out_values, MATRIX_INDEX_TYPE* out_indices, int* num_out_elements) const = 0;
};

/**
 * An operation that transforms a state an initial state and, if the result is valid,
 * adds the final index and amplitude to a collection using add_matrix_element_to_results.
 */
class I_Lambda_Matrix_Elements {
    public:
    virtual ~I_Lambda_Matrix_Elements() = default;
    virtual void operator() (MATRIX_INDEX_TYPE initial_index, const state_int* state,  
                    VALUE_TYPE explicit_diagonal_weight, VALUE_TYPE off_diagonal_weight,
                    VALUE_TYPE* vals, MATRIX_INDEX_TYPE* inds, int* num_out_elements) const = 0;
};


void add_matrix_element_to_results(VALUE_TYPE value, MATRIX_INDEX_TYPE index,
                                   VALUE_TYPE* vals, MATRIX_INDEX_TYPE* inds, int* num_out_elements) {
    // FIXME: cython 0.29 has a bug that prevents num_out_elements from being a reference. Since we are no longer using cython for this code, make it a reference.
    for (size_t i=0; i < *num_out_elements; ++i) {
        if (inds[i] == index) {
            vals[i] += value;
            return;
        }
    }
    vals[*num_out_elements] = value;
    inds[*num_out_elements] = index;
    *num_out_elements += 1;
}

void swap(const state_int* state_in, state_int* state_out, pos_int state_length, const vector<pos_int>& indices) {
    memcpy(state_out, state_in, state_length*sizeof(state_int));
    state_out[indices[0]] = state_in[indices[1]];
    state_out[indices[1]] = state_in[indices[0]];
}
void dont_change(const state_int* state_in, state_int* state_out, pos_int state_length, const vector<pos_int>& indices) {
    memcpy(state_out, state_in, state_length*sizeof(state_int));
}

inline bool is_valid_indices_triangular(MATRIX_INDEX_TYPE initial_index, MATRIX_INDEX_TYPE final_index) { 
    return initial_index <= final_index;
}

inline bool check_order_index_amplitude(const Index_Amplitude& first_index_amplitude, const Index_Amplitude& second_index_amplitude) {
    return first_index_amplitude.ind < second_index_amplitude.ind;
}

void order_nonzero_value_index_pairs(VALUE_TYPE* vals, MATRIX_INDEX_TYPE* inds, int* num_out_elements) {
    int length = *num_out_elements;
    int num_zero_elements = 0;
    vector<Index_Amplitude> zipped (length);
    for (size_t i=0; i < length; ++i) {
        if (abs(vals[i]) < 1e-8) {
            num_zero_elements += 1;
            continue;
        }
        zipped[i-num_zero_elements].ind = inds[i];
        zipped[i-num_zero_elements].amplitude = vals[i];
    }
    sort(zipped.begin(), zipped.end()-num_zero_elements, check_order_index_amplitude);
    *num_out_elements -= num_zero_elements;
    for (size_t i=0; i < *num_out_elements; ++i) {
        vals[i] = zipped[i].amplitude;
        inds[i] = zipped[i].ind;
    }
}

void add_generic_matrix_element(MATRIX_INDEX_TYPE initial_index, const state_int* state, pos_int state_length,
                        const vector<pos_int>& operator_indices, VALUE_TYPE explicit_diagonal_weight, VALUE_TYPE off_diagonal_weight,
                        count_commutations_type* get_commutation_sign, // FIXME: use commutator counter factory
                        const std::shared_ptr<I_State_Index_Amplitude_Translator>& final_state_creator, const std::shared_ptr<Basis_Index_Map>& basis_map,
                        is_valid_state_type* is_valid_state,
                        is_valid_indices_type* is_valid_indices,
                        calculate_weights_type* calculate_weights,
                        get_new_state_type* generate_new_state,
                        VALUE_TYPE* vals, MATRIX_INDEX_TYPE* inds, int* num_out_elements) {
    /*Applies a single rule to the initial state. 
    
    The rule can add to the diagonal value and create up to one off-diagonal state.
    The rule consists of, in order, checking the initial state for validity, calculating the matrix element values from the state,
    adding the diagonal weight, generating a new final sate, finding its dense index, checking the index for validity, 
    and adding the off-diagonal value to results. 
    Steps can be skipped by setting the corresponding function call to nullptr or the weight to zero.
    The sparse indices of the final states and the corresponding values are saved to inds[n:], vals[n:],
    if the final index is not already found in inds, where n is the initial value of *num_out_elements. 
    *num_out_elements is incremented by the number of appended values. 
    */
    if (is_valid_state != nullptr and !is_valid_state(state,operator_indices.data(), operator_indices.size())) {
        return;
    }
    if (calculate_weights != nullptr) {
        calculate_weights(state, operator_indices, explicit_diagonal_weight, off_diagonal_weight);
    }
    if (abs(explicit_diagonal_weight) > 1e-8) {
        add_matrix_element_to_results(explicit_diagonal_weight, initial_index, vals, inds, num_out_elements);
    }
    if (generate_new_state == nullptr or abs(off_diagonal_weight) < 1e-8) {
        return;
    }
    state_int* new_state = (state_int*) malloc(state_length * sizeof(state_int));
    generate_new_state(state, new_state, state_length, operator_indices);
    Index_Amplitude new_index_amplitude;
    final_state_creator->get_lowest_sparse_amplitude_from_state(new_state, new_index_amplitude); // TODO: calculation of the final state and the final value should be done in another function?
    new_index_amplitude.ind = basis_map->get_dense(new_index_amplitude.ind); // FIXME: return an option
    if (new_index_amplitude.ind < 0 or (is_valid_indices != nullptr and !is_valid_indices(initial_index,new_index_amplitude.ind))) { 
        // FIXME: ind should be unsigned and its max value should signal incomlatible state.
        // new_index_amplitude.ind is -1 when new_state is incompatible with the symmetry block
        free(new_state);
        return;
    }
    VALUE_TYPE val = off_diagonal_weight * new_index_amplitude.amplitude;
    if (get_commutation_sign != nullptr) {
        val *= get_commutation_sign(state, state_length, operator_indices);
    }
    add_matrix_element_to_results(val, new_index_amplitude.ind, vals, inds, num_out_elements);    
    free(new_state);
}

void calc_column_matrix_elements(
            MATRIX_INDEX_TYPE initial_dense_index, 
            const std::shared_ptr<Basis_Index_Map>& basis_map,
            const std::shared_ptr<I_State_Index_Amplitude_Translator>& initial_state_creator,
            bool commutes_with_symmetries,
            const I_Lambda_Matrix_Elements& calculate_matrix_elements,
            VALUE_TYPE* out_values, 
            MATRIX_INDEX_TYPE* out_indices, 
            int* num_out_elements) { 
    /* FIXME: update text
    # The matrix elements between the initial_dense_index and all the final states reachable by the calculate_matrix_elements.call method
    # are calculated and saved into out_values and out_indices. The number of (nonzero) saved values is saved into num_out_elements[0]
    # (Cython 0.29 has a bug that prevents num_out_elements from being a reference)                           
    # Prior to calling this function, the pointers are expected to be alocated length at least the number of final states reachable 
    # in the calculate_matrix_elements.call method.
    # Only the first num_out_elements elements of out_values, out_indices are defined on exit.
    # They are ordered ascending in index.
    # initial_state_creator: model-specific implementation of the that inherits from the abstract class
    # commutes_with_symmetries: should be set to True only if <f|[S,O]|i>=0 for any i in the initial basis and f in the final basis,
    #                           where S is the symmetry translation and O is the operator we want to calculate
    #                           (i.e. [S,O] does not have to be strictly zero).
    #                           Setting to False when the above condition holds leads to longer (but still correct) calculation.
    # calculate_matrix_elements: implementation of the Abstract_Lambda_Matrix_Elements class that mimics a lambda with set (uncaptured) parameters.
    #                            (cdef classes can only be passed by reference) 
    */
    vector<state_int> state_vector = initial_state_creator->sparse_index_to_state(basis_map->get_sparse(initial_dense_index));
    int num_states = basis_map->get_num_sparse_in_dense(initial_dense_index);
    num_out_elements[0] = 0;

    if (commutes_with_symmetries) {
        /* if <f|[S,O]|i>=0 for any i in the initial basis and f in the final basis,
          where S is the symmetry translation and O is the operator we want to calculate
          (i.e. [S,O] does not have to be strictly zero)
          simplification of  <kf|O|states> = sum_{n,n'} c_n c'_n <kf[n']|O|states[n]>
          = sum_{n,n'} c_0 c'_n <kf[n'-n]|(S^\dagger)^n O S^n|states[0]> 
          (if states is generated from state using a single symmetry translation S-- generalization to more symmetries is obvious)
          = sum_{n,n'} c_0 c'_{n'-n} <kf[n'-n]|O (S^\dagger)^n S^n|states[0]>
          = sum_{m} N c_0 c'_{m} <kf[m]|O|states[0]>
          and N c_0 = sqrt(N), with n = num_states 
        */  
        VALUE_TYPE explicit_diagonal_weight = 1;
        VALUE_TYPE off_diagonal_weight = sqrt(num_states);
        calculate_matrix_elements(initial_dense_index, state_vector.data(),
                                  explicit_diagonal_weight, off_diagonal_weight,
                                  out_values, out_indices, num_out_elements);
    } else {
        vector<State_Amplitude> states_amplitudes (initial_state_creator->get_symmetries()->get_basis_length(), State_Amplitude());
        initial_state_creator->get_symmetries()->get_symmetry_states_from_state(state_vector.data(), false, states_amplitudes.data());
        for (size_t i=0; i < num_states; ++i) { 
            VALUE_TYPE explicit_diagonal_weight = std::conj(states_amplitudes[i].amplitude) * states_amplitudes[i].amplitude;
            VALUE_TYPE off_diagonal_weight = states_amplitudes[i].amplitude;
            calculate_matrix_elements(initial_dense_index, states_amplitudes[i].state,\
                                      explicit_diagonal_weight, off_diagonal_weight,\
                                      out_values, out_indices, num_out_elements);
        }
    }
    order_nonzero_value_index_pairs(out_values, out_indices, num_out_elements);
}


/* FIXME: update text
# The following abstract classes aim to provide a bridge between the information known to the 
# matrix-constructing function and information required by the function calculating the matrix elements,
# as well as to provide some flexibility to the matrix element creation rules.
# Since some calculations need to be done only once per column, we provide two abstract classes 
# compatible with the calc_column_matrix_elements and add_generic_matrix_element functions.
# (The implementations of) the classes mimic a lambda with set (uncaptured) arguments.
# Classes that inherit from them can add their own captured arguments.
# (Cython 0.29 does not support nogil lambdas)
    # This implementation additionally captures information necessary to recreate the initial state
    # and the lambda class that does the actual calculation.
*/
class Lambda_Matrix_Elements: public I_Lambda_Matrix_Elements { // FIXME: the names are not great!
/* FIXME: update text
    # An implementation that uses the add_generic_matrix_element function.
    # Weights based on operator_index_combinations are stored here
    # to simplify weight calculations in the function.
*/
    private:
    std::shared_ptr<I_State_Index_Amplitude_Translator> final_state_creator;
    std::shared_ptr<Basis_Index_Map> basis_map;
    is_valid_state_type* is_valid_state;
    is_valid_indices_type* is_valid_indices;
    count_commutations_type* get_commutation_sign;
    pos_int state_length;
    get_new_state_type* generate_new_state;
    calculate_weights_type* calculate_weights;
    vector<vector<pos_int>> operator_index_combinations; 
    // operator_index_combinations is 2D: operator_index_combinations[i,j] gives the j-th index in i-th combination.
    // TODO: change to ndarray to have continuous data?
    vector<std::vector<VALUE_TYPE>> combination_weights;
    // combination_weights could be vector<std::array<VALUE_TYPE, 2>> but cython does not know non-type template arguments
    // so working with std::array is a pain
    int num_combinations;

    public:
    Lambda_Matrix_Elements (
                    const std::shared_ptr<I_State_Index_Amplitude_Translator>& final_state_creator, 
                    const std::shared_ptr<Basis_Index_Map>& basis_map,
                    pos_int state_length, 
                    get_new_state_type* generate_new_state,
                    is_valid_state_type* is_valid_state, 
                    is_valid_indices_type* is_valid_indices, 
                    calculate_weights_type* calculate_weights,
                    count_commutations_type* get_commutation_sign,
                    const vector<vector<pos_int>>& operator_index_combinations, 
                    const vector<vector<VALUE_TYPE>>& combination_weights
        ) 
        : final_state_creator(final_state_creator)
        , basis_map(basis_map)
        , is_valid_state(is_valid_state)
        , is_valid_indices(is_valid_indices)
        , state_length(state_length)
        , generate_new_state(generate_new_state)
        , num_combinations(operator_index_combinations.size())
        , operator_index_combinations(operator_index_combinations)
        , combination_weights(combination_weights)
        , get_commutation_sign(get_commutation_sign)
        , calculate_weights(calculate_weights)
    {}

    void operator() (MATRIX_INDEX_TYPE initial_index, const state_int* state,  
                    VALUE_TYPE explicit_diagonal_weight, VALUE_TYPE off_diagonal_weight,
                    VALUE_TYPE* vals, MATRIX_INDEX_TYPE* inds, int* num_out_elements) const {
        for (size_t i=0; i < this->num_combinations; ++i) {
            add_generic_matrix_element(initial_index, state, this->state_length, 
                        this->operator_index_combinations[i], 
                        explicit_diagonal_weight*this->combination_weights[i][0], off_diagonal_weight*this->combination_weights[i][1],
                        this->get_commutation_sign, 
                        this->final_state_creator, 
                        this->basis_map,
                        this->is_valid_state, this->is_valid_indices,
                        this->calculate_weights, this->generate_new_state,
                        vals, inds, num_out_elements);    
        }    
    }
};


class Lambda_Column: public I_Lambda_Column {
    private:
    std::shared_ptr<Basis_Index_Map> basis_map;
    std::shared_ptr<I_State_Index_Amplitude_Translator> initial_state_creator;
    std::shared_ptr<I_Lambda_Matrix_Elements> calculate_matrix_element;
    bool commutes_with_symmetries;

    public:
    Lambda_Column (const std::shared_ptr<Basis_Index_Map>& basis_map,
                   const std::shared_ptr<I_State_Index_Amplitude_Translator>& initial_state_creator, 
                   const std::shared_ptr<I_Lambda_Matrix_Elements>& calculate_matrix_element, 
                   bool commutes_with_symmetries) 
        : basis_map(basis_map)
        , initial_state_creator(initial_state_creator)
        , calculate_matrix_element(calculate_matrix_element)
        , commutes_with_symmetries(commutes_with_symmetries)
    {}

    void operator() (MATRIX_INDEX_TYPE initial_index, VALUE_TYPE* out_values, MATRIX_INDEX_TYPE* out_indices, int* num_out_elements) const {
        calc_column_matrix_elements(initial_index, this->basis_map, this->initial_state_creator,
                    this->commutes_with_symmetries, *(this->calculate_matrix_element),
                    out_values, out_indices, num_out_elements);
    }
};



// FIXME: not needed with the commutation counter factory?
int count_commutations(const state_int* state, pos_int state_length, const vector<pos_int>& indices, 
                       state_int counted_value, bool compare_equals) {
    /* Counts the number of (anti)commutations for operators acting on indices put in front of the state in order 
      specified by indices (indices[-1] acts first).
      Checks state elements for equality with the counted_value when compare_equals is True, for inequality otherwise.
      Does not check for validity of the final state.
      To count multiple values for which the mixed commutator is zero, call the function multiple times
      with different counted_value.
      The mutually non-commuting case is not implemented.
    */
    int *op_number = (int *) calloc(state_length, sizeof(int)); // could be one shorter - the last element is always 0 
    for (pos_int i=0; i < state_length; ++i) {
        if (compare_equals == (state[i] == counted_value)) {
            for (pos_int j=0; j < i; ++j) {
                op_number[j] += 1;  // sum of number of operators on previous nodes changes as we add the operators from indices
            }
        }
    }
    int num_indices = indices.size();
    int count = 0;
    for (int i=num_indices-1; i >= 0; --i) {
        int index = indices[i];
        count += op_number[index];
        for (int j=0; j < index; ++j) {
                op_number[j] += 1;
        }
    }
    free(op_number);
    return count;
}