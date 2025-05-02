from libcpp cimport bool   

from exactdiag.general.types_and_constants cimport MATRIX_INDEX_TYPE_t, VALUE_INDEX_TYPE_t, VALUE_TYPE_t


cpdef void sparse_matrix_dot(const VALUE_TYPE_t[:] vec_in, const VALUE_TYPE_t[:] vals, 
                const VALUE_INDEX_TYPE_t[:] value_inds, const MATRIX_INDEX_TYPE_t[:] minor_inds, const MATRIX_INDEX_TYPE_t[:] major_pointers,
                const bool is_row_major, const bool is_sorted, const bool is_hermitean_triangle, const VALUE_TYPE_t weight,
                const int num_threads, VALUE_TYPE_t[:] vec_out, const bool first_set_to_zero) noexcept nogil

cpdef void block_diagonal_sparse_matrix_dot(const VALUE_TYPE_t[:] vec_in, const MATRIX_INDEX_TYPE_t[:] block_indices, const VALUE_TYPE_t[:] vals, 
                const VALUE_INDEX_TYPE_t[:] value_inds, const MATRIX_INDEX_TYPE_t[:] minor_inds, const MATRIX_INDEX_TYPE_t[:] major_pointers,
                const bool is_row_major, const bool is_sorted, const bool is_hermitean_triangle, const VALUE_TYPE_t weight,
                const int num_threads, VALUE_TYPE_t[:] vec_out, const bool first_set_to_zero) noexcept nogil
