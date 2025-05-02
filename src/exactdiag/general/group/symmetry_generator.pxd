# http://docs.cython.org/en/latest/src/userguide/wrapping_CPlusPlus.html#add-public-attributes
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr, shared_ptr
cimport cython

from exactdiag.general.types_and_constants cimport state_int, pos_int, VALUE_TYPE_t
from exactdiag.general.cpp_classes cimport State_Amplitude
from exactdiag.general.cpp_classes cimport Py_State_Amplitude

# FIXME: except + or noexcept nogil?
cdef extern from './symmetry_generator.h':
    cdef cppclass I_Symmetry_Generator nogil:
        I_Symmetry_Generator() except +

        const vector[int]& get_shift_periodicities() except +
        size_t get_num_shifts() except +
        int get_basis_length() except +

        # cython does not know virtual keyword?
        unique_ptr[I_Symmetry_Generator] clone() except +
        pos_int get_index_by_relative_index(int initial_index, int relative_index) except +
        pos_int get_index_by_relative_shift(int initial_index, const int* relative_shift) except +
        void get_unit_shifts_from_index(int index, int* shifts) except +
        pos_int get_index_from_unit_shifts(const int* shifts) except +
        # size_t get_flat_shift_index(const int* const shifts) except +
        
        int get_symmetry_states_from_state(const state_int* state, bool check_lowest, State_Amplitude* states_amplitudes) except +
        int get_symmetry_states_from_state(const state_int* state, bool check_lowest, int num_repeat, int stride, State_Amplitude* states_amplitudes) except +
        
        void translate_by_symmetry(const state_int* state, const int* symmetry_indices, State_Amplitude& state_amplitude) except +
        void translate_by_symmetry(const state_int* state, const int* symmetry_indices, int num_repeat, int stride, State_Amplitude& state_amplitude) except +
        VALUE_TYPE_t get_q_dot_r(pos_int index_of_node_at_r, const vector[int]& symmetry_qs) except +


    cdef cppclass Direct_Sum(I_Symmetry_Generator) nogil:
        vector[unique_ptr[I_Symmetry_Generator]] generators

        Direct_Sum() except +
        #Direct_Sum(vector[T]& generators) except +
        #Direct_Sum(vector[unique_ptr[T]] generators) except +
        Direct_Sum(vector[unique_ptr[I_Symmetry_Generator]]& generators) except +
        unique_ptr[I_Symmetry_Generator] clone() except +

        pos_int get_index_from_unit_shifts(const int* shifts) except +
        void get_unit_shifts_from_index(int index, int* shifts) except +
        int get_symmetry_states_from_state(const state_int* state, bool check_lowest, int external_num_repeat, 
                                                int external_stride, State_Amplitude* states_amplitudes) except +
        void translate_by_symmetry(const state_int* state, const int* symmetry_indices, int external_num_repeat, 
                                        int external_stride, State_Amplitude& state_amplitude) except +

@cython.final
cdef class Py_Symmetry_Generator:
    """Wrapper around I_Symmetry_Generator for inspecting states.
    
    For heavy calculations, use I_Symmetry_Generator directly.
    """
    cdef shared_ptr[I_Symmetry_Generator] cpp_shared_ptr

    cpdef inline vector[int] get_shift_periodicities(self) nogil:
        return vector[int](self.cpp_shared_ptr.get().get_shift_periodicities())

    cpdef inline size_t get_num_shifts(self) nogil:
        return self.cpp_shared_ptr.get().get_num_shifts()

    cpdef inline int get_basis_length(self) nogil:
        return self.cpp_shared_ptr.get().get_basis_length()

    cpdef inline Py_Symmetry_Generator clone(self):
        cpy = Py_Symmetry_Generator()
        cpy.cpp_shared_ptr = self.cpp_shared_ptr
        return cpy

    cpdef inline pos_int get_index_by_relative_index(self, int initial_index, int relative_index) nogil: 
        return self.cpp_shared_ptr.get().get_index_by_relative_index(initial_index, relative_index)

    cpdef inline pos_int get_index_by_relative_shift(self, int initial_index, const int[::1] relative_shift) nogil: 
        return self.cpp_shared_ptr.get().get_index_by_relative_shift(initial_index, &relative_shift[0])

    cpdef inline void get_unit_shifts_from_index(self, int index, int[::1] shifts) nogil: 
        self.cpp_shared_ptr.get().get_unit_shifts_from_index(index, &shifts[0])
        
    cpdef inline pos_int get_index_from_unit_shifts(self, const int[::1] shifts) nogil: 
        return self.cpp_shared_ptr.get().get_index_from_unit_shifts(&shifts[0])

    cpdef inline int get_symmetry_states_from_state(self, const state_int[::1] state, bool check_lowest, int num_repeat, int stride, Py_State_Amplitude[::1] states_amplitudes): 
        cdef vector[State_Amplitude] vec = vector[State_Amplitude](len(states_amplitudes))
        for i in range(len(states_amplitudes)):
            vec[i] = states_amplitudes[i].cpp_instance
        cdef int num_found
        with nogil:
            num_found = self.cpp_shared_ptr.get().get_symmetry_states_from_state(&state[0], check_lowest, num_repeat, stride, &vec[0])
        for i in range(len(states_amplitudes)):
            states_amplitudes[i].cpp_instance = vec[i]
        return num_found

    cpdef inline void translate_by_symmetry(self, const state_int[::1] state, const int[::1] symmetry_indices, int num_repeat, int stride, Py_State_Amplitude state_amplitude) nogil: 
        self.cpp_shared_ptr.get().translate_by_symmetry(&state[0], &symmetry_indices[0], num_repeat, stride, state_amplitude.cpp_instance)
        
    cpdef inline VALUE_TYPE_t get_q_dot_r(self, pos_int index_of_node_at_r, const vector[int]& symmetry_qs) nogil: 
        return self.cpp_shared_ptr.get().get_q_dot_r(index_of_node_at_r, symmetry_qs)
        