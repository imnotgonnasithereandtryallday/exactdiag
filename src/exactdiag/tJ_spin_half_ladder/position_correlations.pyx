# ------------ needs generalization
import numpy as np
cimport numpy as np
cimport cython
from ..general.symmetry cimport Abstract_State_Index_Amplitude_Translator as State_Translator, Index_Amplitude
from ..general.basis_indexing cimport Basis_Index_Map
from ..general.types_and_constants cimport HOLE_VALUE, SPIN_DOWN_VALUE, SPIN_UP_VALUE, state_int, pos_int, np_pos_int, MATRIX_INDEX_TYPE_t, VALUE_TYPE_t
from ..general.types_and_constants import VALUE_TYPE
from .symmetry cimport State_Index_Amplitude_Translator as State_Translator, Symmetries
from .matrix_setup import setup_excitation_operator
from libcpp.vector cimport vector
from libcpp cimport bool  
from ..logger import Dummy_Logger
from ..general.lanczos_diagonalization import save_spectrum, load_spectrum, get_lowest_eigenpairs
from ..utils import load_calculate_save
from functools import partial

cdef extern from "complex.h":
    double norm(VALUE_TYPE_t) nogil
    VALUE_TYPE_t sqrt(VALUE_TYPE_t) nogil

ctypedef int (*get_value_type)(state_int* const state, pos_int[:] positions) nogil
cdef Dummy_Aggregator DUMMY_AGGREGATOR = Dummy_Aggregator() 
# nogil function does not allow aggregator iniciation in function default argument

def get_hole_spin_projection_correlations(hamiltonian_kwargs, eigenpair_kwargs, spectrum_kwargs, loggers=[Dummy_Logger()]*2, **kwargs):
    def calculate(**kwargs):
        state_translator, basis_map = hamiltonian_kwargs['get_translators'](logger=loggers[1])
        eigvals, eigvecs = get_lowest_eigenpairs(**eigenpair_kwargs,logger=loggers[1])
        num_degenerate_states = 0
        for j in range(len(eigvals)):
            # average over degenerate ground states
            if (abs(eigvals[0]-eigvals[j]) > 1e-6):
                break
            num_degenerate_states += 1
            gs = eigvecs[:,j]
            if j == 0:
                varied_shifts, spectrum = calculate_hole_spin_projection_correlations(state_translator, basis_map, gs, **spectrum_kwargs)
            else:
                _, tmp_spectrum = calculate_hole_spin_projection_correlations(state_translator, basis_map, gs, **spectrum_kwargs)
                spectrum += tmp_spectrum
        return varied_shifts, spectrum / num_degenerate_states
    varied_shifts, spectrum = load_calculate_save(load_fun=load_spectrum, calc_fun=calculate, save_fun=save_spectrum, logger=loggers[0], **spectrum_kwargs)
    return varied_shifts, spectrum

def get_singlet_singlet_correlations(hamiltonian_kwargs, eigenpair_kwargs, spectrum_kwargs, loggers=[Dummy_Logger()]*2, **kwargs):
    def calculate(**kwargs):
        cdef State_Translator state_translator
        state_translator, basis_map = hamiltonian_kwargs['get_translators'](logger=loggers[1])
        symmetries = state_translator.symmetries    # ------------ the fact that i need this here also suggest that a split from other operators is warranted
        eigvals, eigvecs = get_lowest_eigenpairs(**eigenpair_kwargs,logger=loggers[1])
        num_nodes = hamiltonian_kwargs['num_nodes']
        fixed_distances = spectrum_kwargs['fixed_distances']
        op_kwargs = spectrum_kwargs['operator_params']
        get_operator = lambda **kwargs: op_kwargs['setup_func'](**op_kwargs, **kwargs, logger=loggers[0])[0]()
        num_degenerate_states = 0
        for j in range(len(eigvals)):
            # average over degenerate ground states
            if (abs(eigvals[0]-eigvals[j]) > 1e-6):
                break
            num_degenerate_states += 1
            gs = eigvecs[:,j]
            if j == 0:
                varied_shifts, spectrum = calculate_singlet_singlet_correlations(num_nodes, fixed_distances, gs, symmetries, get_operator)
            else:
                _, tmp_spectrum = calculate_singlet_singlet_correlations(num_nodes, fixed_distances, gs, symmetries, get_operator)
                spectrum += tmp_spectrum
        return varied_shifts, spectrum / num_degenerate_states
    varied_shifts, spectrum = load_calculate_save(load_fun=load_spectrum, calc_fun=calculate, save_fun=save_spectrum, logger=loggers[0], **spectrum_kwargs)
    return varied_shifts, spectrum

def calculate_hole_spin_projection_correlations(State_Translator state_translator, Basis_Index_Map basis_map, VALUE_TYPE_t[:] eigvec, \
                                                bool spin_or_hole, int[:,:] fixed_distances=np.empty([0,0], dtype=np.int32), **kwargs):
    # for hole correlations spin_or_hole is False
    # for a 2-hole/Sz correlation, fixed_distances is empty
    # for a 3-hole/Sz correlation, fixed_distances = [x], where x is the shift between the two fixed holes
    cdef pos_int distance, num_nodes = state_translator.symmetries.num_nodes
    cdef int i, dist_ind
    cdef int num_fixed = fixed_distances.shape[0] if fixed_distances.shape[1] > 0 else 0
    position_combinations = np.empty((num_nodes,num_nodes,2+num_fixed),dtype=np_pos_int)
    cdef int[:,:] varied_shifts = np.empty([num_nodes,state_translator.symmetries.num_shifts], dtype=np.int32)
    
    if spin_or_hole:
        aggregator = DUMMY_AGGREGATOR
        get_value = get_spin_projection_value
    else:
        aggregator = Hole_Index_Aggregator(state_translator.num_spin_states)
        get_value = get_holes_present_value

    corrs = np.empty(num_nodes)
    for distance in range(num_nodes):
        for reference_index in range(num_nodes):
            position_combinations[distance, reference_index, 0] = reference_index
            position_combinations[distance, reference_index, 1] = state_translator.symmetries.get_index_by_relative_index(reference_index, distance)
            for dist_ind in range(num_fixed):
                position_combinations[distance, reference_index, 2+dist_ind] = state_translator.symmetries.get_index_by_relative_shift(reference_index, &fixed_distances[dist_ind,0])
        state_translator.symmetries.get_unit_shifts_from_index(distance, &varied_shifts[distance,0])

    for distance in range(num_nodes):
        # how to create private memoryview or slice in nogil for prange?? -------------------------------------- 
        #---- change the position_combinations type to vectors?
        corrs[distance] = calculate_position_correlations(state_translator, basis_map, eigvec, position_combinations[distance,:,:], get_value, aggregator)
    return varied_shifts, corrs


def calculate_singlet_singlet_correlations(int num_nodes, int[:,:] fixed_distances, VALUE_TYPE_t[:] eigvec, Symmetries symmetries, get_operator):
    cdef int num_shifts = symmetries.num_shifts
    cdef int[:,:] shifts = np.empty([3,num_shifts], dtype=np.int32)
    varied_shifts = np.empty([num_nodes,num_shifts], dtype=np.int32)
    cdef int i
    corrs = np.empty(num_nodes)
    shifts[:2,:] = fixed_distances
    for i in range(num_nodes):
        symmetries.get_unit_shifts_from_index(i, &shifts[2,0])
        operator = get_operator(fixed_distances=shifts)
        corrs[i] = np.real(np.vdot(eigvec, operator.dot(eigvec)))
        varied_shifts[i,:] = shifts[2,:]
    return varied_shifts, corrs


cdef class Dummy_Aggregator:
    cdef Index_Amplitude call(self, MATRIX_INDEX_TYPE_t first_dense_index, VALUE_TYPE_t[:] eigvec, Basis_Index_Map basis_map) nogil:
        cdef Index_Amplitude index_amplitude 
        index_amplitude.ind = first_dense_index+1
        index_amplitude.amplitude = eigvec[first_dense_index]
        return index_amplitude

@cython.final
cdef class Hole_Index_Aggregator(Dummy_Aggregator):
    cdef MATRIX_INDEX_TYPE_t num_spin_states

    def __init__(self, MATRIX_INDEX_TYPE_t num_spin_states):
        self.num_spin_states = num_spin_states

    cdef Index_Amplitude call(self, MATRIX_INDEX_TYPE_t first_dense_index, VALUE_TYPE_t[:] eigvec, Basis_Index_Map basis_map) nogil:
        cdef MATRIX_INDEX_TYPE_t dense_index, hole_index = basis_map.get_sparse(first_dense_index) // self.num_spin_states
        cdef MATRIX_INDEX_TYPE_t max_dense_index = min(basis_map.get_num_states(), first_dense_index+self.num_spin_states)
        # num_spin_states is the gap between hole states in sparse_index, dense index has at most this spacing
        cdef VALUE_TYPE_t value = 0
        cdef Index_Amplitude index_amplitude

        for dense_index in range(first_dense_index, max_dense_index):
            if basis_map.get_sparse(dense_index) // self.num_spin_states != hole_index:
                index_amplitude.ind = dense_index
                index_amplitude.amplitude = sqrt(value)
                return index_amplitude
            value += norm(eigvec[dense_index])
        index_amplitude.ind = max_dense_index
        index_amplitude.amplitude = sqrt(value)
        return index_amplitude

cdef double calculate_position_correlations(State_Translator state_translator, Basis_Index_Map basis_map, VALUE_TYPE_t[:] eigvec, \
                            pos_int[:,::1] position_combinations, get_value_type get_value, Dummy_Aggregator aggregator=DUMMY_AGGREGATOR) nogil:    
    # symmetry translations need to be included because the choice of 'lowest' sparse state in the dense representation is arbitrary
    # they are to be included in position_combinations
    cdef MATRIX_INDEX_TYPE_t dense_index = 0, sparse_index, max_dense_index = basis_map.get_num_states()
    cdef vector[state_int] state
    cdef int combination_index
    cdef pos_int[:] position_combination
    cdef Index_Amplitude next_dense_index_amplitude
    cdef double corr = 0
        
    while dense_index < max_dense_index:
        sparse_index = basis_map.get_sparse(dense_index)
        # aggregate can skip a number of iterations
        next_dense_index_amplitude = aggregator.call(dense_index, eigvec, basis_map)
        dense_index = next_dense_index_amplitude.ind
        state = state_translator.sparse_index_to_state(sparse_index)
        amplitude_norm = norm(next_dense_index_amplitude.amplitude)
        for combination_index in range(position_combinations.shape[0]):
            position_combination = position_combinations[combination_index,:]
            corr += amplitude_norm * get_value(&state[0], position_combination)
    return corr        


cdef int get_spin_projection_value(state_int* const state, pos_int[:] positions) nogil: 
    cdef int value = 1
    cdef pos_int i, ip
    for i in range(len(positions)):
        ip = positions[i]
        if state[ip] == HOLE_VALUE:
            return 0
        if state[ip] == SPIN_DOWN_VALUE:
            value *= -1
    return value

cdef int get_holes_present_value(state_int* const state, pos_int[:] positions) nogil:
    cdef pos_int i
    for i in range(len(positions)):
        if state[positions[i]] != HOLE_VALUE:
            return 0
    return 1







