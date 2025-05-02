
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

from exactdiag.general.basis_indexing cimport Basis_Index_Map
from exactdiag.general.types_and_constants cimport state_int, pos_int, MATRIX_INDEX_TYPE_t, VALUE_TYPE_t
from exactdiag.general.symmetry cimport Index_Amplitude, I_State_Index_Amplitude_Translator

cdef extern from "./column_functions.h":
    ctypedef bool is_valid_indices_type(MATRIX_INDEX_TYPE_t initial_index, MATRIX_INDEX_TYPE_t final_index) noexcept nogil
    ctypedef bool is_valid_state_type(const state_int* state, const pos_int* indices, int length) noexcept nogil
    ctypedef void calculate_weights_type(const state_int* state, const vector[pos_int]& operator_indices, VALUE_TYPE_t& explicit_diagonal_weight, VALUE_TYPE_t& off_diagonal_weight) noexcept nogil
    # ctypedef void calculate_column_def(MATRIX_INDEX_TYPE_t column_index, VALUE_TYPE* vals, MATRIX_INDEX_TYPE_t* inds, int* num_out_elements) noexcept nogil
    ctypedef void get_new_state_type(const state_int* state, state_int* new_state, pos_int state_length, const vector[pos_int]& operator_indices) noexcept nogil
    ctypedef int count_commutations_type(const state_int* state, pos_int state_length, const vector[pos_int]& operator_indices) noexcept nogil
    
    cdef cppclass I_Lambda_Column nogil:
        void operator() (MATRIX_INDEX_TYPE_t initial_index, VALUE_TYPE_t* out_values, MATRIX_INDEX_TYPE_t* out_indices, int* num_out_elements) noexcept nogil
        
    cdef cppclass I_Lambda_Matrix_Elements nogil:
        void operator() (MATRIX_INDEX_TYPE_t initial_index, const state_int* state,  
                    VALUE_TYPE_t explicit_diagonal_weight, VALUE_TYPE_t off_diagonal_weight,
                    VALUE_TYPE_t* vals, MATRIX_INDEX_TYPE_t* inds, int* num_out_elements) noexcept nogil



    void add_matrix_element_to_results(VALUE_TYPE_t value, MATRIX_INDEX_TYPE_t index,
                                    VALUE_TYPE_t* vals, MATRIX_INDEX_TYPE_t* inds, int* num_out_elements) noexcept nogil

    void swap(const state_int* state_in, state_int* state_out, pos_int state_length, const vector[pos_int]& indices) noexcept nogil
    void dont_change(const state_int* state_in, state_int* state_out, pos_int state_length, const vector[pos_int]& indices) noexcept nogil
    inline bool is_valid_indices_triangular(MATRIX_INDEX_TYPE_t initial_index, MATRIX_INDEX_TYPE_t final_index) noexcept nogil
    inline bool check_order_index_amplitude(const Index_Amplitude& first_index_amplitude, const Index_Amplitude& second_index_amplitude) noexcept nogil
    void order_nonzero_value_index_pairs(VALUE_TYPE_t* vals, MATRIX_INDEX_TYPE_t* inds, int* num_out_elements) noexcept nogil
    void add_generic_matrix_element(MATRIX_INDEX_TYPE_t initial_index, const state_int* state, pos_int state_length,
                            const vector[pos_int]& operator_indices, VALUE_TYPE_t explicit_diagonal_weight, VALUE_TYPE_t off_diagonal_weight,
                            count_commutations_type* get_commutation_sign, 
                            const shared_ptr[I_State_Index_Amplitude_Translator]& final_state_creator, const shared_ptr[Basis_Index_Map]& basis_map,
                            is_valid_state_type* is_valid_state,
                            is_valid_indices_type* is_valid_indices,
                            calculate_weights_type* calculate_weights,
                            get_new_state_type* generate_new_state,
                            VALUE_TYPE_t* vals, MATRIX_INDEX_TYPE_t* inds, int* num_out_elements) noexcept nogil

    void calc_column_matrix_elements(
                MATRIX_INDEX_TYPE_t initial_dense_index, 
                const shared_ptr[Basis_Index_Map]& basis_map, 
                const shared_ptr[I_State_Index_Amplitude_Translator]& initial_state_creator,
                bool commutes_with_symmetries,
                const I_Lambda_Matrix_Elements& calculate_matrix_elements,
                VALUE_TYPE_t* out_values, 
                MATRIX_INDEX_TYPE_t* out_indices, 
                int* num_out_elements) noexcept nogil

    cdef cppclass Lambda_Matrix_Elements(I_Lambda_Matrix_Elements) nogil:
        shared_ptr[I_State_Index_Amplitude_Translator] final_state_creator;
        shared_ptr[Basis_Index_Map] basis_map;
        is_valid_state_type* is_valid_state;
        is_valid_indices_type* is_valid_indices;
        count_commutations_type* get_commutation_sign;
        pos_int state_length;
        get_new_state_type* generate_new_state;
        calculate_weights_type* calculate_weights;
        vector[vector[pos_int]] operator_index_combinations;
        vector[vector[VALUE_TYPE_t]] combination_weights;
        int num_combinations;
        Lambda_Matrix_Elements () noexcept nogil: pass# cython requires nullary constructor
        Lambda_Matrix_Elements (
                    const shared_ptr[I_State_Index_Amplitude_Translator]& final_state_creator, 
                    const shared_ptr[Basis_Index_Map]& basis_map,
                    pos_int state_length, 
                    get_new_state_type* generate_new_state,
                    is_valid_state_type* is_valid_state, 
                    is_valid_indices_type* is_valid_indices, 
                    calculate_weights_type* calculate_weights,
                    count_commutations_type* get_commutation_sign,
                    const vector[vector[pos_int]]& operator_index_combinations, 
                    const vector[vector[VALUE_TYPE_t]]& combination_weights
        ) noexcept nogil
        void operator() (MATRIX_INDEX_TYPE_t initial_index, const state_int* state,  
                    VALUE_TYPE_t explicit_diagonal_weight, VALUE_TYPE_t off_diagonal_weight,
                    VALUE_TYPE_t* vals, MATRIX_INDEX_TYPE_t* inds, int* num_out_elements) noexcept nogil

    cdef cppclass Lambda_Column(I_Lambda_Column) nogil:
        shared_ptr[Basis_Index_Map] basis_map;
        shared_ptr[I_State_Index_Amplitude_Translator] initial_state_creator;
        shared_ptr[I_Lambda_Matrix_Elements] calculate_matrix_element;
        bool commutes_with_symmetries;
        Lambda_Column () noexcept nogil: pass # cython requires nullary constructor
        Lambda_Column (const shared_ptr[Basis_Index_Map]& basis_map,
                    const shared_ptr[I_State_Index_Amplitude_Translator]& initial_state_creator, 
                    const shared_ptr[I_Lambda_Matrix_Elements]& calculate_matrix_element, 
                    bool commutes_with_symmetries) noexcept nogil

        void operator() (MATRIX_INDEX_TYPE_t initial_index, VALUE_TYPE_t* out_values, MATRIX_INDEX_TYPE_t* out_indices, int* num_out_elements) noexcept nogil

    int count_commutations(const state_int* state, pos_int state_length, const vector[pos_int]& indices, 
                       state_int counted_value, bool compare_equals) noexcept nogil
