from libcpp.vector cimport vector

from exactdiag.general.group.symmetry_generator cimport I_Symmetry_Generator
from exactdiag.general.symmetry cimport Py_Symmetry_Generator
from exactdiag.general.types_and_constants cimport (HOLE_VALUE, SPIN_DOWN_VALUE, SPIN_UP_VALUE, 
                                   state_int, np_state_int, pos_int, np_pos_int, pi, VALUE_TYPE_t)

from itertools import product

import numpy as np

from exactdiag.general.types_and_constants import MATRIX_INDEX_TYPE, VALUE_TYPE


cpdef combinations_from_start_end_points(startpoints, endpoint_shifts, startpoint_weights, endpoint_weights, \
                                        Py_Symmetry_Generator py_symmetries, reverse=False, validator=None):
    """Combine inputs to create a list of indices and a list of weights.

    Each element of the first list gives indices, each index presumably corresponding to 
    a single combination of operators. These indices can be used for state validation and 
    to determine the final state after a set of operators act.
    If more than one startpoints is given, 
    
    startpoints: a list state-position indices. 
    endpoint_shifts: a list of lists of shifts. 
                     One instance of shifts determines the relative position 
                     of one operator with respect to any startpoint.
                     The length of one element of the outer list corresponds to
                     the number of nodes the combination of operator acts on less one 
                     (the remaining one is given by a startpoint).
                     The inner lists are required to share length.
    startpoint_weights: numpy array of pairs (diagonal weight, off-diagonal weight).
                        Each pair corresponds to one element of startpoints.
    endpoint_weights: numpy array of pairs (diagonal weight, off-diagonal weight).
                      Each pair corresponds to one element of the outer list in endpoint_shifts.
    reverse: If True, the order of the output inner lists of indices is reversed.
             I.e. the operators form the same combinations but are applied in reverse order.
    validator: a callable or None. If not None, each combination of indices is validated.
               If the validator returns a falsy value, the combination is not included in the results.
    """
    cdef const I_Symmetry_Generator* symmetries = py_symmetries.cpp_shared_ptr.get()
    end_lengths = np.array([len(end_set) for end_set in endpoint_shifts])
    if np.any(end_lengths != end_lengths[0]):
        message = f'uneaqual lengths in endpoint_shifts.\n' + \
                  f'endpoint lengths: {end_lengths}'
        raise ValueError(message)
    cdef size_t combination_length = 1 + end_lengths[0]

    cdef size_t num_combinations = len(startpoints) * len(endpoint_shifts)
    if num_combinations == 0:
        return np.empty((0,0), dtype=np_pos_int), np.empty((0,2), dtype=VALUE_TYPE)

    cdef vector[vector[pos_int]] index_combinations = vector[vector[pos_int]](num_combinations)
    list_of_shifts = np.empty([combination_length-1, symmetries.get_num_shifts()], dtype=np.int32)
    cdef int[:] shifts = np.empty(symmetries.get_num_shifts(), dtype=np.int32)
    np_shifts = np.frombuffer(shifts, dtype=np.int32)
    skipped = []
    cdef int num_skipped = 0
    # shifts cannot be cython-typed in the 'for ... in enumerate' line
    # but we need it to not be python object to take its address in get_index_by_relative_shift.
    for i, (first, list_of_shifts[:,:]) in enumerate(product(startpoints,endpoint_shifts)):
        index_combinations[i-num_skipped] = vector[pos_int](combination_length)
        index_combinations[i-num_skipped][0] = first
        for j,np_shifts[:] in enumerate(list_of_shifts):
            index_combinations[i-num_skipped][j+1] = symmetries.get_index_by_relative_shift(first, &shifts[0])
        if validator is not None and not validator(index_combinations[i-num_skipped]):
            num_skipped += 1
            skipped.append(i)
    index_combinations.resize(num_combinations-num_skipped)

    num_skipped = 0
    cdef vector[vector[VALUE_TYPE_t]] weights = vector[vector[VALUE_TYPE_t]](num_combinations-num_skipped, vector[VALUE_TYPE_t](2))
    for i,(s,e) in enumerate(product(startpoint_weights,endpoint_weights)):
        if i in skipped[num_skipped:]:
            num_skipped += 1
            continue
        weights[i-num_skipped][0] = (s*e)[0]
        weights[i-num_skipped][1] = (s*e)[1]

    cdef vector[pos_int] tmp
    if reverse:
        for i in range(index_combinations.size()):
            # cyhton cannot create vector from a pair of iterators?
            tmp = index_combinations[i]
            for j in range(combination_length):
                index_combinations[i][j] = tmp[combination_length-1-j]
    return index_combinations, weights


cpdef get_iqr_weights(index_combinations, Py_Symmetry_Generator py_symmetries, base_weights, q):
    """Return base_weights multiplied by exp(iqr).
    
    index_combinations: either a list of single state-positions or of lists of state-positions.
    For each outer dimension of index_combinations, multiplies both weights by exp(-iqr), 
    where r is the position from the first index in each combination.
    base_weights: 2-tuple where the first element is the weight of the explicitly diagonal contribution,
                  and the second element is the weight of the off-diagonal contribution.
    q is given in integer values.
    """
    cdef const I_Symmetry_Generator* symmetries = py_symmetries.cpp_shared_ptr.get()
    index_combinations = np.array(index_combinations, dtype=np_pos_int)
    if len(index_combinations.shape) > 2:
        raise ValueError(f'Expected index_combinations to be 1- or 2-dimensional list. Got {index_combinations}')
    cdef int i,j
    cdef VALUE_TYPE_t q_dot_r = 0
    cdef vector[int] q_vec = vector[int](len(q))
    for i in range(len(q)):
        q_vec[i] = q[i]
    weights = np.empty((index_combinations.shape[0],2), dtype=VALUE_TYPE)
    
    if len(index_combinations.shape) == 1:
        for i in range(index_combinations.shape[0]):
            q_dot_r = symmetries.get_q_dot_r(index_combinations[i], q_vec)
            weights[i] = np.array(base_weights) * np.exp(-1j*q_dot_r) 
    else:
        for i in range(index_combinations.shape[0]):
            q_dot_r = symmetries.get_q_dot_r(index_combinations[i,0], q_vec)
            weights[i,:] = np.array(base_weights) * np.exp(-1j*q_dot_r) 
    return weights


def set_combinatorics(size):
    """Return a combinatorics table where comb[j,i] = number of ways to arrange j elements to i places."""
    comb = np.zeros((size+1,size+1),dtype=MATRIX_INDEX_TYPE,order='C')  
    # one dim could be halved, also only lower triangle
    # the +1 is needed for calculating total number of states
    for i in range(size+1): 
        t = 1 # i choose j backwards
        for j in range(i//2+1): 
            if (i != j and i*j != 0):
                t *= (i+1-j)/j
            comb[j,i] = int(np.around(t))
            comb[i-j,i] = int(np.around(t))
    return comb

class Validator_Unique_Pairs:
    """Keeps track of seen unordered pairs."""
    def __init__(self, ordered):
        self.seen = []
        self.ordered = ordered
    
    def __call__(self, pair) -> bool:
        """Return False if a pair has already been seen; Add the pair to the seen list."""
        pair = tuple(pair)
        if pair in self.seen or (not self.ordered and tuple(reversed(pair)) in self.seen):
            return False
        self.seen.append(pair)
        return True