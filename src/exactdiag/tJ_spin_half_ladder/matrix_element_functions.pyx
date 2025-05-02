"""Collection of methods for validation, weighing, and mutation of states
for use in calculation of matrix elements.
"""
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.string cimport memcpy
from libc.stdlib cimport malloc, calloc, free

cimport cython   

from exactdiag.general.types_and_constants cimport HOLE_VALUE, SPIN_DOWN_VALUE, SPIN_UP_VALUE, state_int, pos_int
from exactdiag.general.group.commutation_counter cimport Commutation_Counter_Factory
from exactdiag.general.column_functions cimport count_commutations


cdef bool is_valid_state_hole_hopping(const state_int* const state, const pos_int* const indices, const int length) noexcept nogil:
    return state[indices[0]] == HOLE_VALUE and state[indices[1]] != HOLE_VALUE

cdef bool is_valid_state_spin_swap(const state_int* const state, const pos_int* const indices, const int length) noexcept nogil:
    # includes the n_i n_j term
    # Sz_i Sz_j - n_i n_j / 4 is zero for Sz_i = Sz_j
    return state[indices[0]] != state[indices[1]] and state[indices[0]] != HOLE_VALUE and state[indices[1]] != HOLE_VALUE

# TODO: The following are too similar -- fit for a lambda?
cdef bool is_valid_state_no_holes(const state_int* const state, const pos_int* const indices, const int length) noexcept nogil:
    cdef int i
    for i in range(length):
        if state[indices[i]] == HOLE_VALUE:
            return False
    return True

cdef bool is_valid_state_all_spin_up(const state_int* const state, const pos_int* const indices, const int length) noexcept nogil:
    cdef int i
    for i in range(length):
        if state[indices[i]] != SPIN_UP_VALUE:
            return False
    return True

cdef bool is_valid_state_all_spin_down(const state_int* const state, const pos_int* const indices, const int length) noexcept nogil:
    cdef int i
    for i in range(length):
        if state[indices[i]] != SPIN_DOWN_VALUE:
            return False
    return True

cdef bool is_valid_state_all_holes(const state_int* const state, const pos_int* const indices, const int length) noexcept nogil: 
    cdef int i
    for i in range(length):
        if state[indices[i]] != HOLE_VALUE:
            return False
    return True

cdef bool is_valid_state_unequal_spins_2holes(const state_int* const state, const pos_int* const indices, const int length) noexcept nogil:
    # Can be used to do singlet-singlet, singlet-triplet (Sz=0), and triplet-triplet correlations.
    # The distinction is in input indices and weight function.
    if indices[0] == indices[1]:
        return False
    cdef bool first_hole_correct = indices[0] == indices[2] or indices[0] == indices[3] or state[indices[0]] == HOLE_VALUE
    cdef bool second_hole_correct = indices[1] == indices[2] or indices[1] == indices[3] or state[indices[1]] == HOLE_VALUE
    cdef bool holes_correct = first_hole_correct and second_hole_correct
    cdef bool spins_correct = state[indices[2]] != state[indices[3]] and state[indices[2]] != HOLE_VALUE and state[indices[3]] != HOLE_VALUE
    return holes_correct and spins_correct


# FIXME: use commutation counter symmetry
cdef inline int get_anticommutation_sign(const state_int* const state, const pos_int state_length, const vector[pos_int]& indices) noexcept nogil:
    cdef bool use_equals = False
    cdef int count = count_commutations(state, state_length, indices, HOLE_VALUE, use_equals)
    return 1 if (count%2)==0 else -1

cdef void get_weights_spin_projection(const state_int* const state, const vector[pos_int]& operator_indices, 
                                    double complex& explicit_diagonal_weight, double complex& off_diagonal_weight) noexcept nogil:
    # Does not check for holes
    cdef pos_int i
    cdef double complex weight_factor = 1
    for i in range(operator_indices.size()):
        if state[operator_indices[i]] == SPIN_DOWN_VALUE:
            weight_factor *= -1
    # Cython 0.29 does not allow direct assignment to a reference
    # https://github.com/cython/cython/issues/1863
    (&explicit_diagonal_weight)[0] *= weight_factor
    (&off_diagonal_weight)[0] *= weight_factor    

cdef void get_weights_annihilation_part_singlet_singlet(const state_int* const state, const vector[pos_int]& operator_indices, 
                                    double complex& explicit_diagonal_weight, double complex& off_diagonal_weight) noexcept nogil:
    # singlet-singlet: \Delta_i^\dagger (y)  \Delta_j (x)
    # \Delta_i (x) = (c_{i \uparrow} c_{i+x \downarrow} - c_{i \downarrow} c_{i+x \uparrow})
    # the sign of the creation part has to be included in the input weights
    cdef pos_int i
    cdef double complex weight_factor = 1
    if state[operator_indices[2]] == SPIN_DOWN_VALUE:
        weight_factor = -1
    (&explicit_diagonal_weight)[0] *= weight_factor
    (&off_diagonal_weight)[0] *= weight_factor


# TODO: the following are too similar -- fit for a lambda?
cdef void add_spin_down(const state_int* const state_in, state_int* const state_out, const pos_int state_length, const vector[pos_int]& indices) noexcept nogil:
    # Does not check for validity of the final state
    cdef pos_int i
    memcpy(state_out,state_in,state_length*sizeof(state_int))
    for i in range(indices.size()):
        state_out[indices[i]] = SPIN_DOWN_VALUE

cdef void add_spin_up(const state_int* const state_in, state_int* const state_out, const pos_int state_length, const vector[pos_int]& indices) noexcept nogil:
    # Does not check for validity of the final state
    cdef pos_int i
    memcpy(state_out,state_in,state_length*sizeof(state_int))
    for i in range(indices.size()):
        state_out[indices[i]] = SPIN_UP_VALUE

cdef void add_hole(const state_int* const state_in, state_int* const state_out, const pos_int state_length, const vector[pos_int]& indices) noexcept nogil:
    # Does not check for validity of the final state
    cdef pos_int i
    memcpy(state_out,state_in,state_length*sizeof(state_int))
    for i in range(indices.size()):
        state_out[indices[i]] = HOLE_VALUE


cdef void remove_two_add_up_down(const state_int* const state_in, state_int* const state_out, const pos_int state_length, const vector[pos_int]& indices) noexcept nogil:
    # Can be used to do singlet-singlet, singlet-triplet (Sz=0), and triplet-triplet correlations.
    # The distinction between the uses is in the input indices and the weight function.
    # indices[-1] acts first: 3 and 2 are annihilated, 1 (up) and 0 (down) are created
    # if indices.size() > 4, the pattern repeats 
    cdef pos_int i
    cdef int remainder = 0, length = indices.size()
    cdef pos_int* changes = [HOLE_VALUE, HOLE_VALUE, SPIN_UP_VALUE, SPIN_DOWN_VALUE]
    memcpy(state_out,state_in,state_length*sizeof(state_int))
    for i in range(length):
        state_out[indices[length-1-i]] = changes[remainder]
        remainder += 1 - 4*(remainder//3)

