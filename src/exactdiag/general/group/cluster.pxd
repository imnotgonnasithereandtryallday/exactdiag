from libcpp.vector cimport vector
from .permutation cimport Permutations
from .commutation_counter cimport State_Hierarchy, Commutation_Counter_Factory
from ..types_and_constants cimport pos_int

#cdef Permutations get_rectangular_cluster(x_length, y_length, State_Hierarchy hierarchy, Commutation_Counter_Factory commutation_counter_factory)


cdef extern from './cluster.h':
    cdef Permutations get_rectangular_cluster(pos_int x_length, pos_int y_length, const State_Hierarchy& hierarchy,
                                     const vector[float]& quantum_numbers,
                                     const Commutation_Counter_Factory& commutation_counter_factory) noexcept nogil