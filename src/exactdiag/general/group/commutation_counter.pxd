# http://docs.cython.org/en/latest/src/userguide/wrapping_CPlusPlus.html#add-public-attributes

from libcpp.vector cimport vector
from libcpp cimport bool
from exactdiag.general.types_and_constants cimport state_int, pos_int


cdef extern from './commutation_counter.h':
    cdef cppclass State_Hierarchy nogil:
        vector[state_int] state_values_hierarchy
        size_t num_values

        #State_Hierarchy[T](T state_values_hierarchy) except + # cython does not recognize this as valid syntax
        State_Hierarchy(vector[state_int] state_values_hierarchy) except +
        #State_Hierarchy(const T& state_values_hierarchy, const size_t num_values) except +

        int check_lower_state(const state_int* const state1, const state_int* const state2, const int state_length) except +


    cdef cppclass Commutation_Counter nogil:
        const vector[state_int]* counted_values;
        bool commutation_parity;
        size_t num_indices;
        vector[pos_int] ac_indices;
        pos_int stride;

        Commutation_Counter(const vector[state_int]* const counted_values, const size_t max_indices, const pos_int stride) except +
        bool is_counted_value(const state_int value) except + # cython does not support const method?
        void add(const pos_int position_index, const state_int* values) except +

    
    cdef cppclass Commutation_Counter_Factory nogil: 
        vector[state_int] counted_values;

        Commutation_Counter_Factory(vector[state_int] counted_values)

        Commutation_Counter get(const size_t max_indices, const pos_int stride) const
