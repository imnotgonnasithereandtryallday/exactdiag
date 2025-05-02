# http://docs.cython.org/en/latest/src/userguide/wrapping_CPlusPlus.html#add-public-attributes

from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp cimport bool

cimport cython

from exactdiag.general.types_and_constants cimport state_int, pos_int


cdef extern from './cpp_classes.h':
    cdef cppclass Product_Generator nogil:
        int iteration
        int length
        int* tops
        vector[int] result

        Product_Generator() except +
        Product_Generator(const int* tops, int length) except +
        Product_Generator(const int* tops, int length, int skip_first) except +
        Product_Generator(const Product_Generator & that)  except +

        Product_Generator& operator=(const Product_Generator& that)
        vector[int]& operator*()
        bool operator==(const Product_Generator& that) const
        bool operator!=(const Product_Generator& that) const
        vector[int]& operator++()
        vector[int] operator++(int)

        void begin()
        vector[int] next()



    cdef cppclass State_Amplitude nogil:
        state_int* state
        double complex amplitude
        pos_int length

        State_Amplitude() except +
        State_Amplitude(int length, double complex amplitude) except +
        State_Amplitude(int length, const state_int* state, double complex amplitude) except+
        State_Amplitude(const State_Amplitude & state_amplitude) except +
        #State_Amplitude(State_Amplitude && state_amplitude) except +
        
        State_Amplitude& operator=(const State_Amplitude& that)
        #State_Amplitude& operator=(const State_Amplitude&& that)

@cython.final
cdef class Py_State_Amplitude:
    """Wrapper around State_Amplitude for inspecting states.
    
    For heavy calculations, use State_Amplitude directly.
    """
    cdef State_Amplitude cpp_instance