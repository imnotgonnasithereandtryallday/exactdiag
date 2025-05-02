from libcpp cimport bool
from libcpp.vector cimport vector

from exactdiag.general.types_and_constants cimport state_int, pos_int
from exactdiag.general.column_functions cimport is_valid_state_type, get_new_state_type


cdef is_valid_state_type is_valid_state_hole_hopping, is_valid_state_spin_swap, is_valid_state_no_holes, \
                         is_valid_state_all_spin_up, is_valid_state_all_spin_down, is_valid_state_all_holes, \
                         is_valid_state_unequal_spins_2holes

cdef get_new_state_type add_spin_down, add_spin_up, add_hole, remove_two_add_up_down


cdef int get_anticommutation_sign(const state_int* const state, const pos_int state_length, const vector[pos_int]& indices) noexcept nogil

cdef void get_weights_spin_projection(const state_int* const state, const vector[pos_int]& operator_indices, 
                                    double complex& explicit_diagonal_weight, double complex& off_diagonal_weight) noexcept nogil
cdef void get_weights_annihilation_part_singlet_singlet(const state_int* const state, const vector[pos_int]& operator_indices, 
                                    double complex& explicit_diagonal_weight, double complex& off_diagonal_weight) noexcept nogil
