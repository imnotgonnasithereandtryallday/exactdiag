from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

cimport cython

from exactdiag.general.types_and_constants cimport state_int, VALUE_TYPE_t, MATRIX_INDEX_TYPE_t
from exactdiag.general.group.symmetry_generator cimport I_Symmetry_Generator, Py_Symmetry_Generator


cdef extern from "./symmetry.h":
    cdef cppclass I_State_Index_Amplitude_Translator nogil:

        MATRIX_INDEX_TYPE_t get_num_minor_sparse_states() noexcept nogil
        shared_ptr[I_Symmetry_Generator] get_symmetries() noexcept nogil

        vector[state_int] sparse_index_to_state(MATRIX_INDEX_TYPE_t sparse_index) noexcept nogil
        MATRIX_INDEX_TYPE_t state_to_sparse_index(const state_int* state) noexcept nogil
        MATRIX_INDEX_TYPE_t check_major_index_supports_lowest_states(MATRIX_INDEX_TYPE_t major_index) noexcept nogil  
        Indices_Counts get_lowest_sparse_indices_from_major_index(MATRIX_INDEX_TYPE_t major_index) noexcept nogil
        Interval get_sparse_index_range_from_major_index(MATRIX_INDEX_TYPE_t major_index) noexcept nogil                                          
        void get_lowest_sparse_amplitude_from_state(const state_int* state, Index_Amplitude& index_amplitude) noexcept nogil

    cdef struct Index_Amplitude:
        MATRIX_INDEX_TYPE_t ind
        VALUE_TYPE_t amplitude

    cdef struct Indices_Counts:
        vector[MATRIX_INDEX_TYPE_t] indices
        vector[int] counts

    cdef struct Interval:
        MATRIX_INDEX_TYPE_t start
        MATRIX_INDEX_TYPE_t end


@cython.final
cdef class Py_State_Index_Amplitude_Translator:
    """Wrapper around I_State_Index_Amplitude_Translator for inspecting states.
    
    For heavy calculations, use I_State_Index_Amplitude_Translator directly.
    """
    cdef shared_ptr[I_State_Index_Amplitude_Translator] cpp_shared_ptr

    cpdef inline MATRIX_INDEX_TYPE_t get_num_minor_sparse_states(self) nogil:
        return self.cpp_shared_ptr.get().get_num_minor_sparse_states()

    cpdef inline Py_Symmetry_Generator get_symmetries(self):
        cdef Py_Symmetry_Generator py_obj = Py_Symmetry_Generator()
        py_obj.cpp_shared_ptr = self.cpp_shared_ptr.get().get_symmetries()
        return py_obj

    cpdef inline vector[state_int] sparse_index_to_state(self, MATRIX_INDEX_TYPE_t sparse_index) nogil:
        return self.cpp_shared_ptr.get().sparse_index_to_state(sparse_index)

    cpdef inline MATRIX_INDEX_TYPE_t state_to_sparse_index(self, const state_int[::1] state) nogil:
        return self.cpp_shared_ptr.get().state_to_sparse_index(&state[0])

    cpdef inline MATRIX_INDEX_TYPE_t check_major_index_supports_lowest_states(self, MATRIX_INDEX_TYPE_t major_index) nogil:
        return self.cpp_shared_ptr.get().check_major_index_supports_lowest_states(major_index)

    cpdef inline Indices_Counts get_lowest_sparse_indices_from_major_index(self, MATRIX_INDEX_TYPE_t major_index) nogil:
        return self.cpp_shared_ptr.get().get_lowest_sparse_indices_from_major_index(major_index)

    cpdef inline Interval get_sparse_index_range_from_major_index(self, MATRIX_INDEX_TYPE_t major_index) nogil:
        return self.cpp_shared_ptr.get().get_sparse_index_range_from_major_index(major_index)  

    cpdef inline void get_lowest_sparse_amplitude_from_state(self, const state_int[::1] state, Index_Amplitude& index_amplitude) nogil:
        self.cpp_shared_ptr.get().get_lowest_sparse_amplitude_from_state(&state[0], index_amplitude)
