from libcpp cimport bool   
from libcpp.vector cimport vector
from libc.math cimport sqrt
from libc.string cimport memset
cimport cython       

from exactdiag.general.types_and_constants cimport MATRIX_INDEX_TYPE_t, VALUE_INDEX_TYPE_t, VALUE_TYPE_t  

from cython.parallel import prange, threadid, parallel
 



cdef extern from './cpp_sparse_dot.h':
    VALUE_TYPE_t matvec_hermitian_conjugate_pass_column_major(const MATRIX_INDEX_TYPE_t column_ind, const int num_nonzero_elements, \
        const VALUE_TYPE_t* const vals, const VALUE_INDEX_TYPE_t* const value_inds, const MATRIX_INDEX_TYPE_t* const row_inds, \
        const VALUE_TYPE_t* const vec_in, const bool is_sorted) nogil

    void matvec_explicit_elements_pass_column_major(const MATRIX_INDEX_TYPE_t range_start, const MATRIX_INDEX_TYPE_t range_end, 
        const MATRIX_INDEX_TYPE_t vec_in_length, const MATRIX_INDEX_TYPE_t* const column_pointers, 
        const VALUE_TYPE_t* const vals, const VALUE_INDEX_TYPE_t* const value_inds, const MATRIX_INDEX_TYPE_t* const row_inds, 
        const VALUE_TYPE_t* const vec_in, const bool is_sorted, const bool is_triangular, 
        VALUE_TYPE_t* const vec_out, const VALUE_TYPE_t weight) nogil

    VALUE_TYPE_t matvec_explicit_elements_pass_row_major(const MATRIX_INDEX_TYPE_t row_index, const MATRIX_INDEX_TYPE_t* const row_pointers, \
        const VALUE_TYPE_t* const vals, const VALUE_INDEX_TYPE_t* const value_inds, const MATRIX_INDEX_TYPE_t* const column_inds, \
        const VALUE_TYPE_t* const vec_in) nogil

    void matvec_hermitian_conjugate_pass_row_major(const MATRIX_INDEX_TYPE_t range_start, const MATRIX_INDEX_TYPE_t range_end, 
        const MATRIX_INDEX_TYPE_t vec_in_length, const MATRIX_INDEX_TYPE_t* const row_pointers, 
        const VALUE_TYPE_t* const vals, const VALUE_INDEX_TYPE_t* const value_inds, const MATRIX_INDEX_TYPE_t* const column_inds, 
        const VALUE_TYPE_t* const vec_in, const bool is_sorted,  
        VALUE_TYPE_t* const vec_out, const VALUE_TYPE_t weight) nogil

    void matvec_block_diagonal_row_major(const MATRIX_INDEX_TYPE_t row_start, const MATRIX_INDEX_TYPE_t row_end, const MATRIX_INDEX_TYPE_t* const row_pointers, \
        const VALUE_TYPE_t* const vals, const VALUE_INDEX_TYPE_t* const value_inds, const MATRIX_INDEX_TYPE_t* const column_inds, \
        const VALUE_TYPE_t* const vec_in, const bool is_sorted, const bool is_triangular, VALUE_TYPE_t* const vec_out) nogil
        

cpdef void sparse_matrix_dot(const VALUE_TYPE_t[:] vec_in, const VALUE_TYPE_t[:] vals, 
                const VALUE_INDEX_TYPE_t[:] value_inds, const MATRIX_INDEX_TYPE_t[:] minor_inds, const MATRIX_INDEX_TYPE_t[:] major_pointers,
                const bool is_row_major, const bool is_sorted, const bool is_hermitian_triangle, const VALUE_TYPE_t weight,
                const int num_threads, VALUE_TYPE_t[:] vec_out, const bool first_set_to_zero) noexcept nogil:  
    cdef MATRIX_INDEX_TYPE_t column_index, row_index
    cdef MATRIX_INDEX_TYPE_t vec_in_length = len(vec_in) 
    cdef MATRIX_INDEX_TYPE_t vec_out_length = len(vec_out) 
    cdef VALUE_INDEX_TYPE_t len_vals = len(vals)
    cdef int num_nonzero_elements_in_column
    
    cdef int chunks_per_thread = 20, chunk_ind, thread_id
    cdef MATRIX_INDEX_TYPE_t chunk_size = vec_out_length // (chunks_per_thread*num_threads) + 1
    cdef MATRIX_INDEX_TYPE_t num_chunks = vec_out_length // chunk_size + 1
    cdef VALUE_TYPE_t add = 0 if first_set_to_zero else 1
    cdef MATRIX_INDEX_TYPE_t chunk_start, chunk_end, shift

    # Equal distribution of the number of rows for the explicit pass is not great:
    # It takes much longer to compute the last row compared to the first.
    cdef vector[MATRIX_INDEX_TYPE_t] explicit_pass_ranges = vector[MATRIX_INDEX_TYPE_t](num_threads+1)
    cdef MATRIX_INDEX_TYPE_t area_triang_range = vec_out_length**2 / num_threads   
    explicit_pass_ranges[0] = 0
    for chunk_ind in range(1,num_threads):
        explicit_pass_ranges[chunk_ind] = (vec_out_length // num_threads + 1) * chunk_ind
    explicit_pass_ranges[num_threads] = vec_out_length
    
    if is_row_major:
        with nogil, parallel(num_threads=num_threads):  
            for chunk_ind in prange(num_chunks, schedule='dynamic', chunksize=1):
                chunk_start = chunk_ind * chunk_size
                chunk_end = min(chunk_start + chunk_size, vec_out_length)
                if first_set_to_zero:
                    memset(&vec_out[chunk_start], 0, (chunk_end-chunk_start) * sizeof(VALUE_TYPE_t))
            for row_index in prange(vec_out_length, schedule='dynamic', chunksize=10000):
                vec_out[row_index] += weight*matvec_explicit_elements_pass_row_major(row_index, 
                                &major_pointers[0], &vals[0], &value_inds[0], &minor_inds[0], &vec_in[0])    
        
            if is_hermitian_triangle:
                thread_id = threadid()
                matvec_hermitian_conjugate_pass_row_major(explicit_pass_ranges[thread_id], explicit_pass_ranges[thread_id+1], vec_in_length, 
                            &major_pointers[0], &vals[0], &value_inds[0], &minor_inds[0], &vec_in[0], is_sorted, &vec_out[0], weight)
        return

    # column-major 
    with nogil, parallel(num_threads=num_threads):  
        for chunk_ind in prange(num_chunks, schedule='dynamic', chunksize=1):
            chunk_start = chunk_ind * chunk_size
            chunk_end = min(chunk_start + chunk_size, vec_out_length)
            if first_set_to_zero:
                memset(&vec_out[chunk_start], 0, (chunk_end-chunk_start) * sizeof(VALUE_TYPE_t))
                # (1-first_set_to_zero)*vec_out[column_index] + weight*matvec does not work -- the first term can give nan
            if is_hermitian_triangle:
                for column_index in range(chunk_start,chunk_end):
                    shift = major_pointers[column_index]
                    num_nonzero_elements_in_column = major_pointers[column_index+1] - shift
                    vec_out[column_index] += weight * matvec_hermitian_conjugate_pass_column_major(column_index, num_nonzero_elements_in_column, 
                                                        &vals[0], &value_inds[shift], &minor_inds[shift], &vec_in[0], is_sorted)
        
        thread_id = threadid()
        matvec_explicit_elements_pass_column_major(explicit_pass_ranges[thread_id], explicit_pass_ranges[thread_id+1], vec_in_length, 
                         &major_pointers[0], &vals[0], &value_inds[0], &minor_inds[0], &vec_in[0], is_sorted, is_hermitian_triangle, &vec_out[0], weight)



cpdef void block_diagonal_sparse_matrix_dot(const VALUE_TYPE_t[:] vec_in, const MATRIX_INDEX_TYPE_t[:] block_indices, const VALUE_TYPE_t[:] vals, 
                const VALUE_INDEX_TYPE_t[:] value_inds, const MATRIX_INDEX_TYPE_t[:] minor_inds, const MATRIX_INDEX_TYPE_t[:] major_pointers,
                const bool is_row_major, const bool is_sorted, const bool is_hermitian_triangle, const VALUE_TYPE_t weight,
                const int num_threads, VALUE_TYPE_t[:] vec_out, const bool first_set_to_zero) noexcept nogil:  
    
    cdef MATRIX_INDEX_TYPE_t block_index, num_blocks = len(block_indices)
    if not is_row_major:
        # not implemented
        return
    for block_index in prange(num_blocks, schedule='dynamic', chunksize=1):
        matvec_block_diagonal_row_major(block_indices[block_index], block_indices[block_index+1], &major_pointers[0], \
                &vals[0], &value_inds[0], &minor_inds[0], &vec_in[0], is_sorted, is_hermitian_triangle, &vec_out[0])



''' # jak podavat sub bloky? kazdy blok muze mit jiny pocet sub bloku, kazdy sub blok ma 2 indexy - start a end
cpdef void shuffled_block_diagonal_sparse_matrix_dot(const VALUE_TYPE_t[:] vec_in, const MATRIX_INDEX_TYPE_t[:,::1] subblocks, const VALUE_TYPE_t[:] vals, 
                const VALUE_INDEX_TYPE_t[:] value_inds, const MATRIX_INDEX_TYPE_t[:] minor_inds, const MATRIX_INDEX_TYPE_t[:] major_pointers,
                const bool is_row_major, const bool is_sorted, const bool is_hermitian_triangle, const VALUE_TYPE_t weight,
                const int num_threads, VALUE_TYPE_t[:] vec_out, const bool first_set_to_zero) nogil:  

    cdef MATRIX_INDEX_TYPE_t block_index, num_blocks = subblocks.shape[0]
    if not is_row_major:
        # not implemented
        return
    for block_index in prange(num_blocks, schedule='dynamic', chunksize=1):
        matvec_shuffled_block_diagonal_row_major(subblocks, &major_pointers[0], \
                &vals[0], &value_inds[0], &minor_inds[0], &vec_in[0], is_sorted, is_hermitian_triangle, &vec_out[0])

'''

