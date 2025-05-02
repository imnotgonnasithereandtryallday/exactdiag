from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr

cimport cython

from exactdiag.general.types_and_constants cimport MATRIX_INDEX_TYPE_t
from exactdiag.general.symmetry cimport I_State_Index_Amplitude_Translator

import pathlib


cdef extern from "./basis_indexing.h":
    cdef cppclass Basis_Index_Map nogil:
        MATRIX_INDEX_TYPE_t dense_to_sparse_len
        vector[MATRIX_INDEX_TYPE_t] dense_to_sparse
        vector[int] num_sparse_states_in_dense

        MATRIX_INDEX_TYPE_t get_sparse(MATRIX_INDEX_TYPE_t dense_index) noexcept nogil

        MATRIX_INDEX_TYPE_t get_dense(MATRIX_INDEX_TYPE_t sparse_index) noexcept nogil

        int get_num_sparse_in_dense(MATRIX_INDEX_TYPE_t dense_index) noexcept nogil

        MATRIX_INDEX_TYPE_t get_num_sparse_in_dense_from_sparse(const MATRIX_INDEX_TYPE_t sparse_index) noexcept nogil

        MATRIX_INDEX_TYPE_t get_num_states() noexcept nogil

    cdef shared_ptr[Basis_Index_Map] get_basis_map(const string& path, MATRIX_INDEX_TYPE_t num_major_only_states, 
                              const I_State_Index_Amplitude_Translator& state_translator, unsigned num_threads) noexcept nogil

@cython.final
cdef class Py_Basis_Index_Map:
    """Wrapper around Basis_Index_Map for inspecting a basis.
    
    For heavy calculations, use Basis_Index_Map directly.
    """
    cdef shared_ptr[Basis_Index_Map] cpp_shared_ptr

    cpdef inline MATRIX_INDEX_TYPE_t get_sparse(self, MATRIX_INDEX_TYPE_t dense_index) noexcept nogil:
        return self.cpp_shared_ptr.get().get_sparse(dense_index)

    cpdef inline MATRIX_INDEX_TYPE_t get_dense(self, MATRIX_INDEX_TYPE_t sparse_index) noexcept nogil:
        return self.cpp_shared_ptr.get().get_dense(sparse_index)

    cpdef inline int get_num_sparse_in_dense(self, MATRIX_INDEX_TYPE_t dense_index) noexcept nogil:
        return self.cpp_shared_ptr.get().get_num_sparse_in_dense(dense_index)

    cpdef inline MATRIX_INDEX_TYPE_t get_num_sparse_in_dense_from_sparse(self, MATRIX_INDEX_TYPE_t sparse_index) noexcept nogil:
        return self.cpp_shared_ptr.get().get_num_sparse_in_dense_from_sparse(sparse_index)

    cpdef inline MATRIX_INDEX_TYPE_t get_num_states(self) noexcept nogil:
        return self.cpp_shared_ptr.get().get_num_states()
