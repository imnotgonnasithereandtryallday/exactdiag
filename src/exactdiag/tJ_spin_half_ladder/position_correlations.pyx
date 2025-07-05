# TODO: needs generalization
from libcpp.memory cimport shared_ptr
import numpy as np
cimport numpy as np
cimport cython
from exactdiag.general.group.symmetry_generator cimport Py_Symmetry_Generator
from ..general.symmetry cimport Index_Amplitude, Py_State_Index_Amplitude_Translator as Py_State_Translator, I_State_Index_Amplitude_Translator
from ..general.basis_indexing cimport Basis_Index_Map, Py_Basis_Index_Map
from ..general.types_and_constants cimport HOLE_VALUE, SPIN_DOWN_VALUE, SPIN_UP_VALUE, state_int, pos_int, np_pos_int, MATRIX_INDEX_TYPE_t, VALUE_TYPE_t
from ..general.types_and_constants import VALUE_TYPE
from .symmetry cimport State_Index_Amplitude_Translator as State_Translator, Symmetries_Single_Spin_Half_2leg_Ladder as Symmetries
from .matrix_setup import get_position_correlation_operator
from libcpp.vector cimport vector
from libcpp cimport bool
from ..general.lanczos_diagonalization import save_spectrum, load_spectrum, get_lowest_eigenpairs
from ..utils import load_calculate_save

from exactdiag.tJ_spin_half_ladder import configs

cdef extern from "complex.h":
    double norm(VALUE_TYPE_t) nogil
    VALUE_TYPE_t sqrt(VALUE_TYPE_t) nogil

ctypedef int (*get_value_type)(state_int* const state, pos_int[:] positions) nogil
cdef Dummy_Aggregator DUMMY_AGGREGATOR = Dummy_Aggregator() 
# nogil function does not allow aggregator iniciation in function default argument

def get_hole_spin_projection_correlations(config: configs.Combined_Position_Config):
    def calculate():
        state_translator, basis_map, symmetries = config.hamiltonian.get_translators()
        eigvals, eigvecs = get_lowest_eigenpairs(config)
        if not config.spectrum.fixed_distances:
            fixed_distances = np.empty([0,len(config.hamiltonian.periodicities)], dtype=np.int32)
        else:
            fixed_distances = np.array([shift.to_npint32() for shift in config.spectrum.fixed_distances], dtype=np.int32)
        spin_or_hole = config.spectrum.name == "Sz_correlations"
        num_degenerate_states = 0
        for j in range(len(eigvals)):
            # average over degenerate ground states
            if (abs(eigvals[0]-eigvals[j]) > 1e-6):  # TODO: Unify this tolerance across files.
                break
            num_degenerate_states += 1
            gs = eigvecs[:,j]
            if j == 0:
                varied_shifts, spectrum = calculate_hole_spin_projection_correlations(state_translator, basis_map, gs, spin_or_hole, fixed_distances)
            else:
                _, tmp_spectrum = calculate_hole_spin_projection_correlations(state_translator, basis_map, gs, spin_or_hole, fixed_distances)
                spectrum += tmp_spectrum
        return varied_shifts, spectrum / num_degenerate_states
    path = config.get_spectrum_path()
    varied_shifts, spectrum = load_calculate_save(path, load_fun=load_spectrum, calc_fun=calculate, save_fun=save_spectrum)
    return varied_shifts, spectrum

def get_singlet_singlet_correlations(config: configs.Combined_Position_Config):
    def calculate(**kwargs):
        state_translator, basis_map, py_symmetries = config.hamiltonian.get_translators()
        eigvals, eigvecs = get_lowest_eigenpairs(config)
        num_nodes = config.hamiltonian.num_nodes
        fixed_distances = np.array([shift.to_npint32() for shift in config.spectrum.fixed_distances], dtype=np.int32)
        num_degenerate_states = 0
        get_operaor = lambda fixed_distances: get_position_correlation_operator(config, fixed_distances)[0]()
        for j in range(len(eigvals)):
            # average over degenerate ground states
            if (abs(eigvals[0]-eigvals[j]) > 1e-6):  # TODO: set a single threshold for all functions.
                break
            num_degenerate_states += 1
            gs = eigvecs[:,j]
            if j == 0:
                varied_shifts, spectrum = calculate_singlet_singlet_correlations(num_nodes, fixed_distances, gs, py_symmetries, get_operaor)
            else:
                _, tmp_spectrum = calculate_singlet_singlet_correlations(num_nodes, fixed_distances, gs, py_symmetries, get_operaor)
                spectrum += tmp_spectrum
        return varied_shifts, spectrum / num_degenerate_states
    path = config.get_spectrum_path()
    varied_shifts, spectrum = load_calculate_save(path, load_fun=load_spectrum, calc_fun=calculate, save_fun=save_spectrum)
    return varied_shifts, spectrum

def calculate_hole_spin_projection_correlations(Py_State_Translator py_state_translator, Py_Basis_Index_Map basis_map, VALUE_TYPE_t[:] eigvec, \
                                                bool spin_or_hole, int[:,:] fixed_distances):
    # for hole correlations spin_or_hole is False
    # for a 2-hole/Sz correlation, fixed_distances is empty
    # for a 3-hole/Sz correlation, fixed_distances = [x], where x is the shift between the two fixed holes
    # TODO: spin_or_hole to enum.
    # TODO: Can we unify with calculate_singlet_singlet_correlations?
    cdef pos_int distance, num_nodes = py_state_translator.get_symmetries().get_basis_length()
    cdef int i, dist_ind
    cdef int num_fixed = fixed_distances.shape[0]
    position_combinations = np.empty((num_nodes,num_nodes,2+num_fixed),dtype=np_pos_int)
    cdef int[:,:] varied_shifts = np.empty([num_nodes,py_state_translator.get_symmetries().get_num_shifts()], dtype=np.int32)
    
    if spin_or_hole:
        aggregator = DUMMY_AGGREGATOR
        get_value = get_spin_projection_value
    else:
        aggregator = Hole_Index_Aggregator(py_state_translator.get_num_minor_sparse_states())
        get_value = get_holes_present_value

    corrs = np.empty(num_nodes)
    for distance in range(num_nodes):
        for reference_index in range(num_nodes):
            position_combinations[distance, reference_index, 0] = reference_index
            position_combinations[distance, reference_index, 1] = py_state_translator.get_symmetries().get_index_by_relative_index(reference_index, distance)
            for dist_ind in range(num_fixed):
                position_combinations[distance, reference_index, 2+dist_ind] = py_state_translator.get_symmetries().cpp_shared_ptr.get().get_index_by_relative_shift(reference_index, &fixed_distances[dist_ind,0])
        py_state_translator.get_symmetries().cpp_shared_ptr.get().get_unit_shifts_from_index(distance, &varied_shifts[distance,0])

    for distance in range(num_nodes):
        # TODO: How to create private memoryview or slice in nogil for prange?
        # TODO: Change the position_combinations type to vectors?
        corrs[distance] = calculate_position_correlations(py_state_translator.cpp_shared_ptr, basis_map.cpp_shared_ptr, eigvec, position_combinations[distance,:,:], get_value, aggregator)
    return varied_shifts, corrs


def calculate_singlet_singlet_correlations(int num_nodes, int[:,:] fixed_distances, VALUE_TYPE_t[:] eigvec, Py_Symmetry_Generator symmetries, get_operator):
    cdef size_t num_shifts = symmetries.cpp_shared_ptr.get().get_num_shifts()
    cdef int[:,:] shifts = np.empty([3,num_shifts], dtype=np.int32)
    varied_shifts = np.empty([num_nodes,num_shifts], dtype=np.int32)
    cdef int i
    corrs = np.empty(num_nodes)
    shifts[:2,:] = fixed_distances
    for i in range(num_nodes):
        symmetries.cpp_shared_ptr.get().get_unit_shifts_from_index(i, &shifts[2,0])
        operator = get_operator(shifts)
        corrs[i] = np.real(np.vdot(eigvec, operator.dot(eigvec)))
        varied_shifts[i,:] = shifts[2,:]
    return varied_shifts, corrs


cdef class Dummy_Aggregator:
    cdef Index_Amplitude call(self, MATRIX_INDEX_TYPE_t first_dense_index, VALUE_TYPE_t[:] eigvec, shared_ptr[Basis_Index_Map] basis_map) nogil:
        cdef Index_Amplitude index_amplitude 
        index_amplitude.ind = first_dense_index+1
        index_amplitude.amplitude = eigvec[first_dense_index]
        return index_amplitude

@cython.final
cdef class Hole_Index_Aggregator(Dummy_Aggregator):
    cdef MATRIX_INDEX_TYPE_t num_spin_states

    def __init__(self, MATRIX_INDEX_TYPE_t num_spin_states):
        self.num_spin_states = num_spin_states

    cdef Index_Amplitude call(self, MATRIX_INDEX_TYPE_t first_dense_index, VALUE_TYPE_t[:] eigvec, shared_ptr[Basis_Index_Map] basis_map) nogil:
        cdef MATRIX_INDEX_TYPE_t dense_index, hole_index = basis_map.get().get_sparse(first_dense_index) // self.num_spin_states
        cdef MATRIX_INDEX_TYPE_t max_dense_index = min(basis_map.get().get_num_states(), first_dense_index+self.num_spin_states)
        # num_spin_states is the gap between hole states in sparse_index, dense index has at most this spacing
        cdef VALUE_TYPE_t value = 0
        cdef Index_Amplitude index_amplitude

        for dense_index in range(first_dense_index, max_dense_index):
            if basis_map.get().get_sparse(dense_index) // self.num_spin_states != hole_index:
                index_amplitude.ind = dense_index
                index_amplitude.amplitude = sqrt(value)
                return index_amplitude
            value += norm(eigvec[dense_index])
        index_amplitude.ind = max_dense_index
        index_amplitude.amplitude = sqrt(value)
        return index_amplitude

cdef double calculate_position_correlations(shared_ptr[I_State_Index_Amplitude_Translator] state_translator, shared_ptr[Basis_Index_Map] basis_map, VALUE_TYPE_t[:] eigvec, \
                            pos_int[:,::1] position_combinations, get_value_type get_value, Dummy_Aggregator aggregator=DUMMY_AGGREGATOR) nogil:    
    # symmetry translations need to be included because the choice of 'lowest' sparse state in the dense representation is arbitrary
    # they are to be included in position_combinations
    cdef MATRIX_INDEX_TYPE_t dense_index = 0, sparse_index, max_dense_index = basis_map.get().get_num_states()
    cdef vector[state_int] state
    cdef int combination_index
    cdef pos_int[:] position_combination
    cdef Index_Amplitude next_dense_index_amplitude
    cdef double corr = 0
        
    while dense_index < max_dense_index:
        sparse_index = basis_map.get().get_sparse(dense_index)
        # aggregate can skip a number of iterations
        next_dense_index_amplitude = aggregator.call(dense_index, eigvec, basis_map)
        dense_index = next_dense_index_amplitude.ind
        state = state_translator.get().sparse_index_to_state(sparse_index)
        amplitude_norm = norm(next_dense_index_amplitude.amplitude)
        for combination_index in range(position_combinations.shape[0]):
            position_combination = position_combinations[combination_index,:]
            corr += amplitude_norm * get_value(&state[0], position_combination)
    return corr        


cdef int get_spin_projection_value(state_int* const state, pos_int[:] positions) nogil: 
    """Return the product of spin projection values over given positions."""
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
    """Return whether a hole is present at any of the given positions."""
    cdef pos_int i
    for i in range(len(positions)):
        if state[positions[i]] != HOLE_VALUE:
            return 0
    return 1







