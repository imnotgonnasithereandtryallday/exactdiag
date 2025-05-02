from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr, shared_ptr

from exactdiag.general.group.symmetry_generator cimport I_Symmetry_Generator
from exactdiag.general.types_and_constants cimport state_int, pos_int, VALUE_TYPE_t, MATRIX_INDEX_TYPE_t
from exactdiag.general.symmetry cimport I_State_Index_Amplitude_Translator, Interval, Indices_Counts
from exactdiag.general.cpp_classes cimport State_Amplitude


cdef extern from './symmetry_ladder.h':  
    # The _ladder suffix is because the build system has some problem with correctly identifying headers locations.
    cdef cppclass Symmetries_Single_Spin_Half_2leg_Ladder nogil:
        Symmetries_Single_Spin_Half_2leg_Ladder(pos_int num_rungs, const vector[int]& quantum_numbers) noexcept nogil

    cdef cppclass State_Index_Amplitude_Translator(I_State_Index_Amplitude_Translator) nogil:
        State_Index_Amplitude_Translator() noexcept nogil:
            pass # cython requires default constructor
        State_Index_Amplitude_Translator (
            int num_holes, int num_down_spins, int num_spin_states, 
            const vector[vector[[MATRIX_INDEX_TYPE_t]]]& combinatorics_table, const shared_ptr[I_Symmetry_Generator]& symmetries
        ) noexcept nogil
        size_t get_num_minor_sparse_states() noexcept nogil
        shared_ptr[I_Symmetry_Generator] get_symmetries() noexcept nogil
        vector[state_int] sparse_index_to_state(MATRIX_INDEX_TYPE_t sparse_index) noexcept nogil
        MATRIX_INDEX_TYPE_t state_to_sparse_index(const state_int* state) noexcept nogil
        MATRIX_INDEX_TYPE_t check_major_index_supports_lowest_states(MATRIX_INDEX_TYPE_t major_index) noexcept nogil
        void find_hole_positions(MATRIX_INDEX_TYPE_t hole_index, vector[pos_int]& hole_positions) noexcept nogil
