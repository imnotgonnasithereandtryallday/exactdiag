from exactdiag.general.column_functions cimport I_Lambda_Column
from exactdiag.general.matrix_rules_utils cimport Wrapped_Column_Func
from exactdiag.general.types_and_constants cimport MATRIX_INDEX_TYPE_t, VALUE_INDEX_TYPE_t, VALUE_TYPE_t


cdef class _Tmp_To_Perm_Transcriber:
    cdef int max_values_per_column, num_unique_values
    cdef MATRIX_INDEX_TYPE_t num_nonzero_elements, last_filled_column_index, num_columns
    cdef MATRIX_INDEX_TYPE_t[:] tmp_row_inds, tmp_column_pointers
    cdef VALUE_TYPE_t[:] tmp_values
    cdef VALUE_INDEX_TYPE_t[:] tmp_value_inds        

    cdef void organize_chunk_results(self, long chunk_size, const MATRIX_INDEX_TYPE_t* chunk_inds, \
                                    const VALUE_TYPE_t* chunk_values, const int* chunk_num_results) noexcept nogil

cdef void calc_sparse_matrix(Wrapped_Column_Func column_func, MATRIX_INDEX_TYPE_t num_columns, 
                                    int max_values_per_column, int num_cores, matrix_signature)