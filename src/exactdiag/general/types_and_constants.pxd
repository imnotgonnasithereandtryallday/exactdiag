cdef extern from './types_and_constants.h':
    ctypedef short state_int
    ctypedef short pos_int

cdef double pi

cdef state_int HOLE_VALUE # -------------------- these are model specific!
cdef state_int SPIN_UP_VALUE
cdef state_int SPIN_DOWN_VALUE

cdef state_int OCCUPIED_VALUE
cdef state_int EMPTY_VALUE

cdef object np_pos_int
cdef object np_state_int

# FIXME: clean up

cimport numpy as np
ctypedef long long MATRIX_INDEX_TYPE_t #np.int64_t 
ctypedef short VALUE_INDEX_TYPE_t #np.int16_t
ctypedef double complex VALUE_TYPE_t #np.complex_t
cdef VALUE_INDEX_TYPE_t MAX_NUM_UNIQUIE_VALUES
# FIXME: _t means type. TYPE_t is redundant!