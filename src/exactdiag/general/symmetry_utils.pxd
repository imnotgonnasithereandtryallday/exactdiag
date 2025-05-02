from libcpp.vector cimport vector
from exactdiag.general.types_and_constants cimport state_int, pos_int


cdef extern from "./symmetry.h":
    cdef int dot_product_without_conj(const int* vec1, const int* vec2, int length) noexcept nogil
    cdef pos_int find_first[T](const vector[T]& array, const T& value) noexcept nogil
    cdef int python_mod(int dividend, int divisor) noexcept nogil
