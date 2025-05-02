from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp cimport bool

cimport cython

from exactdiag.general.types_and_constants cimport np_state_int, state_int, pos_int

import numpy as np


@cython.final
cdef class Py_State_Amplitude:
    def __init__(self, state, double complex amplitude):
        cdef int length = len(state)
        cdef state_int[:] state_buf = np.zeros(length, dtype=np_state_int)
        self.cpp_instance = State_Amplitude(length, &state_buf[0], amplitude)

    def __repr__(self):
        state = ",".join(f'{s}' for s in self.state)
        return f"{type(self)}([{state}], {self.amplitude})"

    @property
    def state(self) -> list[int]:
        return [self.cpp_instance.state[i] for i in range(self.cpp_instance.length)]
    
    @property
    def amplitude(self) -> complex:
        return self.cpp_instance.amplitude

    @property
    def length(self) -> int:
        return self.cpp_instance.length
