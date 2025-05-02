from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr

from exactdiag.general.types_and_constants cimport state_int, pos_int, VALUE_TYPE_t
from exactdiag.general.cpp_classes cimport State_Amplitude
from .symmetry_generator cimport I_Symmetry_Generator
from .commutation_counter cimport State_Hierarchy, Commutation_Counter_Factory

cdef extern from './ndarray.h':
    cdef cppclass ndarray[T]:
        pass

cdef extern from './permutation.h':
    cdef cppclass Permutations(I_Symmetry_Generator) nogil:
        vector[int] shift_periodicities
        size_t num_shifts
        int basis_length
        pos_int internal_stride
        vector[pos_int] flattened_shifts_to_index
        ndarray[pos_int] index_to_shifts
        ndarray[pos_int] index_maps
        vector[float] quantum_numbers
        State_Hierarchy state_hierarchy

        #Permutations() except + # cython requires this
        Permutations(size_t num_shifts, pos_int basis_length, pos_int internal_stride, 
                const ndarray[pos_int]& shifts_to_index, const ndarray[pos_int]& index_maps, 
                const vector[float] quantum_numbers,
                State_Hierarchy state_hierarchy, Commutation_Counter_Factory commutation_counter_factory) except +
        Permutations(const Permutations& that) except +
        #Permutations(const Permutations* const that) except +
        unique_ptr[I_Symmetry_Generator] clone() except +
        # FIXME: except + or noexcept?
        pos_int get_index_by_relative_index(int initial_index, int relative_index) except +
        pos_int get_index_by_relative_shift(int initial_index, const int* relative_shift) except +
        void get_unit_shifts_from_index(int index, int* shifts) except +
        pos_int get_index_from_unit_shifts(const int* shifts) except +
        void translate_by_symmetry(const state_int* state, const int* symmetry_indices, State_Amplitude& state_amplitude) except +
        int get_symmetry_states_from_state(const state_int* state, bool check_lowest, int external_num_repeat, 
                                                int external_stride, State_Amplitude* states_amplitudes) except +
        void translate_by_symmetry(const state_int* state, const int* symmetry_indices, int external_num_repeat, 
                                        int external_stride, State_Amplitude& state_amplitude) except +
        VALUE_TYPE_t get_q_dot_r(pos_int index_of_node_at_r, const vector[int]& symmetry_qs) except +
