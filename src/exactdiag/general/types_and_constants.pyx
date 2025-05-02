import numpy as np

# NOTE: make sure these match those in the .h file

# state stuff
cdef double pi = np.pi

cdef object np_pos_int = np.int16
cdef object np_state_int = np.int16

cdef state_int HOLE_VALUE = 0
cdef state_int SPIN_UP_VALUE = 1
cdef state_int SPIN_DOWN_VALUE = 2

cdef state_int OCCUPIED_VALUE = 1
cdef state_int EMPTY_VALUE = 0


# matrix stuff
# these are not cdefed -- cdef object cannot be imported 
# (only cimported which is not possible in .py)
MATRIX_INDEX_TYPE = np.int64
VALUE_INDEX_TYPE = np.int16
VALUE_TYPE = np.cdouble
MAX_NUM_UNIQUIE_VALUES = np.iinfo(VALUE_INDEX_TYPE).max
SYMMETRY_OPTIONS = ['hermitian','sorted']
