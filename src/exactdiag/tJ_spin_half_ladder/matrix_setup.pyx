from libcpp cimport bool  
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

from exactdiag.general.types_and_constants cimport (pos_int, np_pos_int, state_int, np_state_int, HOLE_VALUE, 
                                            SPIN_UP_VALUE, SPIN_DOWN_VALUE, VALUE_TYPE_t, MATRIX_INDEX_TYPE_t)
from exactdiag.general.column_functions cimport (I_Lambda_Matrix_Elements, Lambda_Matrix_Elements, I_Lambda_Column, Lambda_Column,
                                    swap, dont_change, is_valid_indices_triangular,
                                    count_commutations_type, calculate_weights_type,
                                    is_valid_indices_type, is_valid_state_type, get_new_state_type)
from exactdiag.general.matrix_rules_utils cimport Wrapped_Column_Func
from exactdiag.general.basis_indexing cimport Py_Basis_Index_Map, get_basis_map, Basis_Index_Map
from exactdiag.general.symmetry cimport Py_State_Index_Amplitude_Translator, I_State_Index_Amplitude_Translator as I_State_Trans
from exactdiag.general.symmetry_utils cimport python_mod
from exactdiag.general.group.symmetry_generator cimport I_Symmetry_Generator, Py_Symmetry_Generator
from exactdiag.tJ_spin_half_ladder.matrix_element_functions cimport (is_valid_state_spin_swap, is_valid_state_hole_hopping, is_valid_state_no_holes, 
                            get_weights_spin_projection, get_anticommutation_sign, 
                            is_valid_state_all_holes, is_valid_state_all_spin_up, is_valid_state_all_spin_down, 
                            add_hole, add_spin_down, add_spin_up, remove_two_add_up_down, 
                            get_weights_annihilation_part_singlet_singlet, is_valid_state_unequal_spins_2holes)
from exactdiag.tJ_spin_half_ladder.symmetry cimport Symmetries_Single_Spin_Half_2leg_Ladder, State_Index_Amplitude_Translator as Ladder_Trans

from functools import partial
from dataclasses import asdict
import pathlib
import copy
import logging

import numpy as np

from exactdiag.general.sparse_matrices import get_sparse_matrices, get_sparse_matrix, FILE_NAMES
from exactdiag.general.cython_sparse_matrices import Signature
from exactdiag.general.matrix_rules_utils import combinations_from_start_end_points, get_iqr_weights, Validator_Unique_Pairs
from exactdiag.tJ_spin_half_ladder import configs


_logger = logging.getLogger(__name__)


def setup_hamiltonian(config: "config.Hamiltonian_Config"):
    """Return a callable that returns the Hamiltonian.
    
    The callable, when called, loads matrices if they already exist or calculates and saves them.
    """
    _logger.debug(f"Started setting up Hamilonian with {config}.")
    cdef Py_Symmetry_Generator py_sym
    cdef Py_State_Index_Amplitude_Translator py_state_creator
    cdef Py_Basis_Index_Map py_basis
    py_state_creator, py_basis, py_sym = config.get_translators()  
    # TODO: Here we load the basis map even if the matrix is already saved just to know what shape we are looking for.
    #       Should we load just based on name without checking the shape? We do not check the shape when loading eigenpairs.

    # We define info shared with all the terms
    cdef Basis_Index_Map* bmap = py_basis.cpp_shared_ptr.get()
    cdef MATRIX_INDEX_TYPE_t length = bmap.get_num_states()
    shape = [int(length)]*2
    matrix_symmetry_strings = ['hermitian', 'sorted'] # TODO: bit flags?
    if config.triangle_only:
        matrix_symmetry_strings.append('triangle_only')
    shared_signature = {  # FIXME: clean up
        'symmetry_strings': matrix_symmetry_strings,
        'shape': shape,
        'initial_system_info': config.get_symmetry_info(),
        'final_system_info': None,
        'major': config.major,
        **FILE_NAMES,
    }
    
    # Here we create the rules for the calculation of the matrix elements 
    # Hamiltonian has operators in pairs -- we define the distance between the nodes the 
    # operators act on by shifts, take the initial position to be all the nodes, 
    # and create a matrix for each term with nonzero input weight.

    # TODO: The only thing that makes this setup specific for ladder clusters with a specific node ordering 
    #       are the 'shifts' (and 'max_num_values_per_column'?) values in the following dictionaries.
    #       Find a good way to make these a part of config.

    # The following max_num_values_per_column values could be further limited, but slight overestimation is ok.
    tl_shift_info = {
        'shifts': [(1,0), (-1,0)], 
        'matrix_name': 'tl',
        'torJ': True, # TODO: Find a better name?
        'weights': (0, -1), # the minus sign of the hole hopping operator is accounted for here
        'max_num_values_per_column': 2 * config.num_holes # the two comes from the length of shifts  # TODO: tie them up better
    }
    tr_shift_info = {
        'shifts': [(0,1), (0,-1)],
        'validator': Validator_Unique_Pairs(ordered=True), # validator enforces the non-periodicity in the rung hopping
        'matrix_name': 'tr',
        'torJ': True,
        'weights': (0, -1), # the minus sign of the hole hopping operator is accounted for here
        'max_num_values_per_column': min(config.num_holes, config.num_rungs)
    }
    jl_shift_info = {
        'shifts': [(1,0)],
        'matrix_name': 'jl',
        'torJ': False,
        'weights': (-0.5, 0.5), 
        'max_num_values_per_column': min(config.num_spins, config.num_rungs) + 1 # +1 for explicit diagonal value
    }
    jr_shift_info = {
        'shifts': [(0,1)],
        'validator': Validator_Unique_Pairs(ordered=False),
        'matrix_name': 'jr',
        'torJ': False,
        'weights': (-0.5, 0.5), 
        'max_num_values_per_column': min(config.num_spins, config.num_rungs) + 1 # +1 for explicit diagonal value
    }
    infos = [tl_shift_info, tr_shift_info, jl_shift_info, jr_shift_info]
    set_of_matrix_kwargs = [
        _get_tJ_lambda_kwargs(
            config.get_calc_folder(),
            py_state_creator, py_basis, py_sym, shared_signature=shared_signature,
            num_threads=config.num_threads, **lamb_kw
        ) 
        for lamb_kw in infos
    ]
    
    set_of_nonzero_matrix_kwargs = []
    nonzero_weights = {}
    for dic in set_of_matrix_kwargs: # TODO: what is this mess?
        name = dic['signature']['matrix_name'] 
        weight = getattr(config.weights, name, 0)
        if weight != 0:
            set_of_nonzero_matrix_kwargs.append(dic)
            nonzero_weights[name] = weight
    _logger.debug(f"Finished setting up Hamilonian with {config}.")
    return partial(get_sparse_matrices, set_of_matrix_kwargs=set_of_nonzero_matrix_kwargs, weights=nonzero_weights)


def _get_tJ_lambda_kwargs(
        symmetry_block_folder: pathlib.Path,
        Py_State_Index_Amplitude_Translator py_state_creator, 
        Py_Basis_Index_Map py_basis_map, 
        Py_Symmetry_Generator py_symmetries, 
        shifts, weights,
        max_num_values_per_column, torJ, matrix_name, 
        shared_signature, num_threads, validator=None
    ):
    # Each node is considered as an initial position of the operators.
    # weights do not change with start/end points.
    cdef shared_ptr[I_State_Trans] state_creator = <shared_ptr[I_State_Trans]> py_state_creator.cpp_shared_ptr
    cdef shared_ptr[Basis_Index_Map] basis_map = <shared_ptr[Basis_Index_Map]> py_basis_map.cpp_shared_ptr
    cdef shared_ptr[I_Symmetry_Generator] symmetries = <shared_ptr[I_Symmetry_Generator]> py_symmetries.cpp_shared_ptr

    num_nodes = symmetries.get().get_basis_length()
    startpoints = [i for i in range(num_nodes)]
    endpoint_shifts = [[s] for s in shifts] # added extra layer since combinations_from_start_end_points supports 
                                            # combinations of more that 2 operators
    startpoint_weights = np.ones((len(startpoints),2))
    endpoint_weights = np.array([weights]*len(endpoint_shifts)) 
    cdef vector[vector[pos_int]] operator_index_combinations
    cdef vector[vector[VALUE_TYPE_t]] combination_weights
    operator_index_combinations, combination_weights = combinations_from_start_end_points( \
                                startpoints, endpoint_shifts, startpoint_weights, endpoint_weights, \
                                py_symmetries, reverse=True, validator=validator)
    cdef count_commutations_type* get_anticommutation
    if torJ:
        is_valid_state = is_valid_state_hole_hopping 
        get_anticommutation = get_anticommutation_sign
    else:
        is_valid_state = is_valid_state_spin_swap
        get_anticommutation = NULL
    cdef calculate_weights_type* calculate_weights = NULL
    commutes_with_symmetries = True
    cdef is_valid_indices_type* is_valid_indices_check = NULL
    if 'triangle_only' in shared_signature['symmetry_strings']:
        is_valid_indices_check = is_valid_indices_triangular
    cdef shared_ptr[I_Lambda_Matrix_Elements] pair_func = shared_ptr[I_Lambda_Matrix_Elements](new Lambda_Matrix_Elements(state_creator,
                                                    basis_map, num_nodes, swap,
                                                    is_valid_state, 
                                                    is_valid_indices_check, 
                                                    calculate_weights, get_anticommutation,
                                                    operator_index_combinations, combination_weights))
    cdef shared_ptr[I_Lambda_Column] column_func = shared_ptr[I_Lambda_Column](new Lambda_Column(basis_map, state_creator, pair_func, commutes_with_symmetries))
    cdef Wrapped_Column_Func py_column_func = Wrapped_Column_Func()
    py_column_func.cpp_shared_ptr = column_func
    signature = Signature(
        matrix_name=matrix_name,
        folder_name=str(symmetry_block_folder / matrix_name),
        max_values_per_column=max_num_values_per_column,
        **shared_signature
    )
    matrix_kwargs = { # TODO: remove the duplicates that are already in signature.
        'column_func': py_column_func,
        'max_values_per_column': max_num_values_per_column,
        'signature': signature,
        'shape': shared_signature['shape'],
        'num_threads': num_threads,
    }
    return matrix_kwargs


def setup_excitation_operator(initial_config: "config.Limited_Spectrum_Config"): 
    """Return a callable that returns the operator, 
    info about the final symmetry block and the Hamiltonian of the final block.
    
    The callable, when called, loads matrices if they already exist or calculates and saves them.
    """
    # FIXME: separate position correlations

    num_rungs = initial_config.hamiltonian.num_rungs
    num_holes = initial_config.hamiltonian.num_holes 
    cdef pos_int num_nodes = initial_config.hamiltonian.num_nodes
    num_spins = num_nodes - num_holes
    kx,ky = initial_config.hamiltonian.symmetry_qs.to_npint32()
    operator_symmetry_qs = initial_config.spectrum.operator_symmetry_qs
    qx,qy = operator_symmetry_qs.to_npint32() if operator_symmetry_qs is not None else (0,0)
    
    final_config = copy.deepcopy(initial_config)
    combinatorics_table = initial_config.hamiltonian.combinatorics_table

    cdef Py_Symmetry_Generator py_symmetries
    cdef Py_State_Index_Amplitude_Translator py_state_creator
    cdef Py_Basis_Index_Map py_initial_basis
    py_state_creator, py_initial_basis, py_symmetries = initial_config.hamiltonian.get_translators()
    cdef shared_ptr[Basis_Index_Map] initial_basis_map = <shared_ptr[Basis_Index_Map]> py_initial_basis.cpp_shared_ptr
    cdef shared_ptr[I_Symmetry_Generator] symmetries = <shared_ptr[I_Symmetry_Generator]> py_symmetries.cpp_shared_ptr
    cdef shared_ptr[I_State_Trans] initial_state_creator = <shared_ptr[I_State_Trans]> py_state_creator.cpp_shared_ptr

    cdef count_commutations_type* get_anticommutation
    cdef is_valid_indices_type* is_valid_indices
    cdef is_valid_state_type* is_valid_state
    cdef get_new_state_type* generate_new_state
    cdef calculate_weights_type* calculate_weights
    cdef int num_shifts = symmetries.get().get_num_shifts()
    cdef vector[vector[pos_int]] operator_index_combinations
    cdef vector[vector[VALUE_TYPE_t]] combination_weights

    # TODO: Again, as in setup_hamiltonian, only the endpoint_shifts (and possibly the get_iqr_weights function?) 
    #       limit the use of this function to the specific ladder clusters.
    #       Generalize.

    name = initial_config.spectrum.name     
    if name == configs.Spectrum_Name.SZQ:
        operator_index_combinations = np.array([(i,) for i in range(num_nodes)], dtype=np_pos_int) 
        # explicitly diagonal only if qs == 0
        weights = (0, 0.5 / np.sqrt(num_nodes))
        combination_weights = get_iqr_weights(operator_index_combinations, py_symmetries, base_weights=weights, q=(qx,qy))
        max_num_values_per_column = 1
        get_anticommutation = NULL
        is_valid_indices = NULL 
        is_valid_state = is_valid_state_no_holes
        generate_new_state = dont_change
        calculate_weights = get_weights_spin_projection
        commutes_with_symmetries = True

    elif name == configs.Spectrum_Name.CURRENT_RUNG:
        startpoints = np.array([i for i in range(num_nodes)], dtype=np_pos_int)
        endpoint_shifts = [[(0,1)]] 
        qy += 1 # the sign difference between hopping directions is taken care of by the qy+=1 change in the final state symmetry
        startpoint_weights = get_iqr_weights(startpoints, py_symmetries, base_weights=(0,1j), q=(qx,qy))
        endpoint_weights = np.array([(0,1)])
        operator_index_combinations, combination_weights = combinations_from_start_end_points( \
                                startpoints, endpoint_shifts, startpoint_weights, endpoint_weights, \
                                py_symmetries, reverse=True)
        max_num_values_per_column = len(endpoint_shifts) * num_holes
        get_anticommutation = NULL
        is_valid_indices = NULL
        is_valid_state = is_valid_state_hole_hopping
        generate_new_state = swap
        calculate_weights = NULL
        commutes_with_symmetries = True

    elif name == configs.Spectrum_Name.CURRENT_LEG:
        startpoints = np.array([i for i in range(num_nodes)], dtype=np_pos_int)
        endpoint_shifts = [[(1,0)], [(-1,0)]]
        startpoint_weights = get_iqr_weights(startpoints, py_symmetries, base_weights=(0,1j), q=(qx,qy))
        endpoint_weights = np.array([(0,1),(0,-1)])
        operator_index_combinations, combination_weights = combinations_from_start_end_points( \
                                startpoints, endpoint_shifts, startpoint_weights, endpoint_weights, \
                                py_symmetries, reverse=True)
        max_num_values_per_column = len(endpoint_shifts) * num_holes
        get_anticommutation = get_anticommutation_sign
        is_valid_indices = NULL
        is_valid_state = is_valid_state_hole_hopping
        generate_new_state = swap
        calculate_weights = NULL
        commutes_with_symmetries = True

    elif name == configs.Spectrum_Name.SPECTRAL_FUNCTION_PLUS:
        # adds electron with spin down if the initial spin projection > 0, 
        # otherwise the added electron has spin up
        # so that the final spin projection is close to zero and non-negative (if possible)
        operator_index_combinations = np.array([(i,) for i in range(num_nodes)], dtype=np_pos_int)
        weights = (0,1/np.sqrt(num_nodes))
        combination_weights = get_iqr_weights(operator_index_combinations, py_symmetries, base_weights=weights, q=(qx,qy))
        max_num_values_per_column = num_holes
        get_anticommutation = get_anticommutation_sign
        is_valid_indices = NULL
        is_valid_state = is_valid_state_all_holes
        calculate_weights = NULL
        commutes_with_symmetries = True
        final_config.hamiltonian.num_holes -= 1
        if initial_config.hamiltonian.total_spin_projection > 0:
            final_config.hamiltonian.total_spin_projection -= 1
            generate_new_state = add_spin_down
        else:
            final_config.hamiltonian.total_spin_projection += 1
            generate_new_state = add_spin_up
        
    elif name == configs.Spectrum_Name.SPECTRAL_FUNCTION_MINUS:
        # removes electron with spin down if the initial spin projection <= 0, 
        # otherwise the removed electron has spin up
        # so that the final spin projection is close to zero and non-negative (if possible)

        # qx,qy define the change in the state's momentum. a given qx,qy corresponds to c_{-qx.-qy} and the sign in the exponential
        # is the same as for c_{qx,qy}^\dagger
        operator_index_combinations = np.array([(i,) for i in range(num_nodes)],dtype=np_pos_int)
        weights = (0,1/np.sqrt(num_nodes))
        combination_weights = get_iqr_weights(operator_index_combinations, py_symmetries, base_weights=weights,q=(qx,qy))
        get_anticommutation = get_anticommutation_sign
        is_valid_indices = NULL
        calculate_weights = NULL
        generate_new_state = add_hole
        commutes_with_symmetries = True
        final_config.hamiltonian.num_holes += 1
        if initial_config.hamiltonian.total_spin_projection <= 0:
            max_num_values_per_column = initial_config.hamiltonian.num_down_spins
            final_config.hamiltonian.total_spin_projection += 1
            is_valid_state = is_valid_state_all_spin_down
        else:
            max_num_values_per_column = num_spins - initial_config.hamiltonian.num_down_spins
            final_config.hamiltonian.total_spin_projection -= 1
            is_valid_state = is_valid_state_all_spin_up
    else:
        raise ValueError(f'{name} not supported')

    final_config.hamiltonian.symmetry_qs.leg = (kx + qx + (num_rungs-1)//2) % num_rungs - (num_rungs-1)//2
    final_config.hamiltonian.symmetry_qs.rung = python_mod(ky+qy, 2)

    get_excited_H = final_config.hamiltonian.setup()
    cdef Py_State_Index_Amplitude_Translator py_final_state_creator
    cdef Py_Basis_Index_Map py_final_basis_map
    py_final_state_creator, py_final_basis_map, _ = final_config.hamiltonian.get_translators()
    # TODO: Here we load the basis map even if the matrix is already saved just to know what shape we are looking for.
    #       Should we load just based on name without checking the shape? We do not check the shape when loading eigenpairs.
    cdef shared_ptr[Basis_Index_Map] final_basis_map = <shared_ptr[Basis_Index_Map]> py_final_basis_map.cpp_shared_ptr
    cdef shared_ptr[I_State_Trans] final_state_creator = <shared_ptr[I_State_Trans]> py_final_state_creator.cpp_shared_ptr

    cdef shared_ptr[I_Lambda_Matrix_Elements] matel_func = shared_ptr[I_Lambda_Matrix_Elements](new Lambda_Matrix_Elements(final_state_creator, final_basis_map,
                                                    num_nodes, generate_new_state,
                                                    is_valid_state, is_valid_indices, 
                                                    calculate_weights, get_anticommutation,
                                                    operator_index_combinations, combination_weights))
    cdef shared_ptr[I_Lambda_Column] column_func = shared_ptr[I_Lambda_Column](new Lambda_Column(initial_basis_map, initial_state_creator, matel_func, commutes_with_symmetries))
    cdef Wrapped_Column_Func py_column_func = Wrapped_Column_Func()
    py_column_func.cpp_shared_ptr = column_func

    symmetry_strings = ['sorted']  
    shape = [final_basis_map.get().get_num_states(), initial_basis_map.get().get_num_states()]
    q_string = f'_{operator_symmetry_qs.to_npint32()}' if operator_symmetry_qs is not None else ''
    full_name = f'{name}{q_string}'         
    signature = Signature(
        matrix_name=full_name,
        shape=shape,
        symmetry_strings=symmetry_strings,
        max_values_per_column=max_num_values_per_column,
        major='column',
        initial_system_info=initial_config.hamiltonian.get_symmetry_info(),
        final_system_info=final_config.hamiltonian.get_symmetry_info(),
        folder_name=str(initial_config.hamiltonian.get_calc_folder() / full_name),
        **FILE_NAMES,
    )
    operator_kwargs = {  # TODO: remove the duplicates that are already in signature.
        'column_func': py_column_func,
        'max_values_per_column': max_num_values_per_column,
        'shape': shape,
        'signature': signature,
        'num_threads': initial_config.spectrum.num_threads,
    }
    get_operator = partial(get_sparse_matrix, **operator_kwargs)
    _logger.debug(f"Finished setting up operator {name} with {initial_config}.")
    return get_operator, final_config.hamiltonian, get_excited_H


def py_get_ladder_translators(config: "config.Hamiltonian_Config"
    ) -> tuple[Py_State_Index_Amplitude_Translator, Py_Basis_Index_Map, Py_Symmetry_Generator]: 
    _logger.debug(f"Started calculating translators with {config}.")
    py_table = config.combinatorics_table
    cdef vector[size_t] shape = vector[size_t](2)
    for i in range(2):
        shape[i] = py_table.shape[i]
    cdef vector[vector[MATRIX_INDEX_TYPE_t]] combinatorics_table = vector[vector[MATRIX_INDEX_TYPE_t]](shape[0], vector[MATRIX_INDEX_TYPE_t](shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            combinatorics_table[i][j] = py_table[i,j]

    # Cython does not know std::filesystem::path, we use string instead.
    path = config.get_calc_folder() / "Sz_local-momentum_map.dat"
    cdef string file_path = bytes(f"{path!s}", encoding='utf-8')  # TODO: where should this be decided?

    cdef Py_Symmetry_Generator py_sym = Py_Symmetry_Generator()
    cdef Py_State_Index_Amplitude_Translator py_trans = Py_State_Index_Amplitude_Translator()
    cdef Py_Basis_Index_Map py_basis = Py_Basis_Index_Map()
    py_sym.cpp_shared_ptr = shared_ptr[I_Symmetry_Generator](new Symmetries_Single_Spin_Half_2leg_Ladder(config.num_rungs, config.symmetry_qs.to_npint32()))
    py_trans.cpp_shared_ptr = shared_ptr[I_State_Trans](
        new Ladder_Trans(
                config.num_holes, config.num_down_spins, config.num_spin_states, combinatorics_table, py_sym.cpp_shared_ptr
                # TODO: is num_spin_states used incorrectly in get_num_minor_sparse_states?
    ))
    cdef size_t num_hole_only_states = combinatorics_table[config.num_holes][config.num_nodes]
    py_basis.cpp_shared_ptr = get_basis_map(file_path, num_hole_only_states, 
                               py_trans.cpp_shared_ptr.get()[0], config.num_threads)
    _logger.debug(f"Finished calculating translators with {config}.")
    return py_trans, py_basis, py_sym


def get_position_correlation_operator(config: "configs.Limited_Position_Correlation_Config", free_shift: "configs.Position_Shift"):
    num_rungs = config.hamiltonian.num_rungs
    num_holes = config.hamiltonian.num_holes 
    cdef pos_int num_nodes = config.hamiltonian.num_nodes
    num_spins = num_nodes - num_holes
    
    final_config = copy.deepcopy(config)

    cdef Py_Symmetry_Generator py_symmetries
    cdef Py_State_Index_Amplitude_Translator py_state_creator
    cdef Py_Basis_Index_Map py_initial_basis
    py_state_creator, py_initial_basis, py_symmetries = config.hamiltonian.get_translators()
    cdef shared_ptr[Basis_Index_Map] initial_basis_map = <shared_ptr[Basis_Index_Map]> py_initial_basis.cpp_shared_ptr
    cdef shared_ptr[I_Symmetry_Generator] symmetries = <shared_ptr[I_Symmetry_Generator]> py_symmetries.cpp_shared_ptr
    cdef shared_ptr[I_State_Trans] initial_state_creator = <shared_ptr[I_State_Trans]> py_state_creator.cpp_shared_ptr

    cdef count_commutations_type* get_anticommutation
    cdef is_valid_indices_type* is_valid_indices
    cdef is_valid_state_type* is_valid_state
    cdef get_new_state_type* generate_new_state
    cdef calculate_weights_type* calculate_weights
    cdef vector[vector[pos_int]] operator_index_combinations
    cdef vector[vector[VALUE_TYPE_t]] combination_weights


    if config.correlations.name != 'singlet-singlet':
        raise ValueError(f"Unsupported operator {config.correlations.name}")

    # singlet-singlet: \Delta_i^\dagger (y)  \Delta_j (x)
    # \Delta_i (x) = 1/sqrt(2) * (c_{i \uparrow} c_{i+x \downarrow} - c_{i \downarrow} c_{i+x \uparrow})
    fixed_distances = np.array([shift.to_npint32() for shift in config.correlations.fixed_distances] + [free_shift.to_npint32()], dtype=np.int32)
    # fixed_distances = separation in [creation singlet, annihilation singlet, between singlets]
    #                   given by shifts [rung, leg]
    #                = shifts corresponding to  [y, x, j-i]

    # The annihilation part \Delta_j (x) is given by just one combination of two shifts
    # corresponding to j, j+x. Both combinations of sign and spins are taken into account in 
    # the get_weights_annihilation_part_singlet_singlet function.

    # The startpoint i and the first shift in the first element of endpoint_shifts creates
    # c_{i \downarrow}^\dagger c_{i+y \uparrow}^\dagger
    # which corresponds to the second term in \Delta_i^\dagger(y)
    # (the anticommutation cancels the sign)

    # The startpoint i and the first shift in the second element of endpoint_shifts creates
    # c_{i \downarrow}^\dagger c_{i-y \uparrow}^\dagger
    # which corresponds to the first term in \Delta_{i-y}^\dagger(y) 
    # Therefore, the other shifts are also adjusted. 
    startpoints = [i for i in range(num_nodes)]
    startpoints_weights = np.array([(0,0.5/num_nodes)]*num_nodes)
    endpoint_shifts = np.array([(fixed_distances[0], fixed_distances[2], fixed_distances[1] + fixed_distances[2]), \
                        (-fixed_distances[0], fixed_distances[2] - fixed_distances[0], fixed_distances[1] + fixed_distances[2] - fixed_distances[0])])
    endpoint_weights = np.array([(0,1)]*endpoint_shifts.shape[0])
    operator_index_combinations, combination_weights = combinations_from_start_end_points( \
                            startpoints, endpoint_shifts, startpoints_weights, endpoint_weights, \
                            py_symmetries, reverse=False)
    max_num_values_per_column = 4*num_spins 
    get_anticommutation = get_anticommutation_sign
    is_valid_indices = NULL
    is_valid_state = is_valid_state_unequal_spins_2holes
    generate_new_state = remove_two_add_up_down
    calculate_weights = get_weights_annihilation_part_singlet_singlet
    commutes_with_symmetries = True

    get_excited_H = config.hamiltonian.setup()

    cdef shared_ptr[I_Lambda_Matrix_Elements] matel_func = shared_ptr[I_Lambda_Matrix_Elements](new Lambda_Matrix_Elements(initial_state_creator, initial_basis_map,
                                                    num_nodes, generate_new_state,
                                                    is_valid_state, is_valid_indices, 
                                                    calculate_weights, get_anticommutation,
                                                    operator_index_combinations, combination_weights))
    cdef shared_ptr[I_Lambda_Column] column_func = shared_ptr[I_Lambda_Column](new Lambda_Column(initial_basis_map, initial_state_creator, matel_func, commutes_with_symmetries))
    cdef Wrapped_Column_Func py_column_func = Wrapped_Column_Func()
    py_column_func.cpp_shared_ptr = column_func

    symmetry_strings = ['sorted']  
    shape = [initial_basis_map.get().get_num_states(), initial_basis_map.get().get_num_states()]
    shift_string = f'_shifts{fixed_distances.tolist()}'
    full_name = f'{config.correlations.name}{shift_string}'
    signature = Signature(
        matrix_name=full_name,
        shape=shape,
        symmetry_strings=symmetry_strings,
        max_values_per_column=max_num_values_per_column,
        major='column',
        initial_system_info=config.hamiltonian.get_symmetry_info(),
        final_system_info=None,
        folder_name=str(config.hamiltonian.get_calc_folder() / full_name),
        **FILE_NAMES,
    )
    operator_kwargs = {  # TODO: remove the duplicates that are already in signature.
        'column_func': py_column_func,
        'max_values_per_column': max_num_values_per_column,
        'shape': shape,
        'signature': signature,
        'num_threads': config.correlations.num_threads,
    }
    get_operator = partial(get_sparse_matrix, **operator_kwargs)
    _logger.debug(f"Finished setting up operator {config.correlations.name} with {config}.")
    return get_operator, final_config.hamiltonian, get_excited_H