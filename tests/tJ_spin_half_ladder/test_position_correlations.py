import functools
import itertools
import json
import pathlib
import typing

import numpy as np

from exactdiag.general.sparse_matrices import Sparse_Matrix

# from exactdiag.general.types_and_constants import HOLE_VALUE
from exactdiag.tJ_spin_half_ladder import api
import exactdiag.tJ_spin_half_ladder.configs as lc


def test_hole_correlations_against_precalculated_paulina():
    """Calculated 2-hole correlations should match values in Paulina's dissertation."""
    # NOTE: Paulina has the values for 9 rungs and 2 holes wrong: the probabilities do not even add up to 1.
    folder = pathlib.Path(__file__).parent / "precalculated_position_correlations"
    diffs = _get_correlation_diffs(folder / "hole_correlations_paulina.txt", normalize=True)
    listed_precision = 1e-3
    assert all([np.all(np.abs(diff) < listed_precision) for diff in diffs])


def test_hole_correlations_against_precalculated_scco():
    """Calculated 2- and 3-hole correlations should match values in hte SCCO paper."""
    folder = pathlib.Path(__file__).parent / "precalculated_position_correlations"
    diffs = _get_correlation_diffs(folder / "hole_correlations_scco.txt", normalize=True)
    listed_precision = 1e-3
    passed = [np.all(np.abs(diff) < listed_precision) for diff in diffs]
    # TODO: Refactor to get a more usefull message
    assert all(passed), passed


def _get_correlation_diffs(path: pathlib.Path, normalize: bool) -> list[np.ndarray]:
    with open(path, mode="r", encoding="utf-8") as fp:
        precalc = json.load(fp)

    diffs = []
    for d in precalc:
        target = d.pop("probabilities")
        config = lc.Full_Position_Correlation_Config(**d)
        spectrum = api.get_position_correlations(config)
        probabilities = spectrum.spectrum.flatten()
        if normalize:
            # The sum of probabilities does not sum to the number of holes if there are fixed holes:
            # it sums to the probability that the holes are in the fixed coordination
            # times some combinatorics based on number of holes.
            # We normalize the fixed probabilities to 1 as they are not given in the articles.
            probabilities /= probabilities[0]
        diffs.append(probabilities[: len(target)] - target)
    return diffs


def test_singlet_singlet_correlations_against_precalculated():
    """Calculated singlet-singlet correlations should match values published in my dissertation."""
    # NOTE: Takes around 15 minutes on my pc.
    folder = pathlib.Path(__file__).parent / "precalculated_position_correlations"
    diffs = _get_correlation_diffs(folder / "singlet_correlations_scco.txt", normalize=False)
    listed_precision = 1e-3
    passed = [np.all(np.abs(diff) < listed_precision) for diff in diffs]
    assert all(passed), passed


def test_hole_matrix():
    """Matrix of projection to states should filter states with undesirable hole positions."""

    HOLE_VALUE = 0  # FIXME: Change it to be importable into regular python files.

    for num_rungs in [4, 5, 6]:
        for num_holes in range(4):  # We check all permutations, so num_holes cant go much higher.
            for fixed_distances, free_shift in _yield_distances(2 * num_rungs, num_holes):
                config = lc.Limited_Position_Correlation_Config(
                    hamiltonian={
                        "num_rungs": num_rungs,
                        "num_holes": num_holes,
                        "weights": {"tl": 0.45, "tr": 0.36, "jl": 0.15, "jr": 0.12},
                        "symmetry_qs": {"leg": 0, "rung": 0},
                        "num_threads": 8,
                    },
                    correlations={"name": "hole_correlations", "fixed_distances": fixed_distances},
                )
                matrix = config.setup_excitation_operator(free_shift)[0]()
                state_translator, basis_map, symmetries = config.hamiltonian.get_translators()
                relative_distances = [shift.to_npint32() for shift in config.correlations.fixed_distances] + [
                    free_shift.to_npint32()
                ]
                check_projection = functools.partial(
                    _are_values_at_relative_distances,
                    state_value=HOLE_VALUE,
                    relative_distances=relative_distances,
                    state_translator=state_translator,
                    basis_map=basis_map,
                    symmetries=symmetries,
                )
                assert _is_matrix_projection(matrix, check_projection, num_holes), (config, free_shift)


def _are_values_at_relative_distances(
    dense_index, state_value, relative_distances, state_translator, basis_map, symmetries
):
    sparse_index = basis_map.get_sparse(dense_index)
    sparse_state = state_translator.sparse_index_to_state(sparse_index)
    value_positions = [i for i, v in enumerate(sparse_state) if v == state_value]
    if not value_positions:
        return False
    for initial_index in value_positions:
        found = [False] * len(relative_distances)
        for i, relative_distance in enumerate(relative_distances):
            expected_pos = symmetries.get_index_by_relative_shift(initial_index, relative_distance)
            if expected_pos not in value_positions:
                break
            found[i] = True
        if all(found):
            # TODO: We could sum the matches here to get the value of the matrix element.
            return True
    return False


def _yield_distances(num_nodes, num_choices):
    for num_fixed in range(num_choices - 1):
        for fixed in itertools.product(*[list(range(num_nodes))] * num_fixed):
            fixed = [lc.Position_Shift(leg=i // 2, rung=i % 2) for i in fixed]  # TODO: Should not assume node ordering.
            for free in range(num_nodes):
                free = lc.Position_Shift(leg=free // 2, rung=free % 2)
                yield fixed, free


def _is_matrix_projection(matrix: Sparse_Matrix, is_state_in_subspace: typing.Callable, max_value: int) -> bool:
    # Only checks for projections diagonal in the matrix's basis.
    # square matrix
    # all nonzero values are ... not only ones -- the matrix is not really defined as a true projection
    # nonzero values only on diagonal
    # each row has a value decided by is_state_in_subspace
    if matrix.shape[0] != matrix.shape[1]:
        return False
    # The matrix cannot be empty so a single zero element is saved if there are no non-zero elements.
    is_empty = matrix.vals.tolist() == [0] and matrix.major_pointers[-1] == 1
    # The values can be whole numbers bigger than one if there are i.e. more holes than the fixed number.
    # TODO: This makes the matrix technically not a projection. Improve naming.
    if not is_empty and not set(matrix.vals.tolist()) <= set(range(1, max_value + 1)):
        return False
    for dense_index, major_pointer in enumerate(matrix.major_pointers[:-1]):
        end = matrix.major_pointers[dense_index + 1]
        is_in = is_state_in_subspace(dense_index)
        if major_pointer == end:
            if is_in:
                return False
            continue
        if end != major_pointer + 1:
            return False
        if dense_index != matrix.minor_inds[major_pointer]:
            return False
        if not is_in:
            return False
    return True


def test_sz_matrix():
    """Matrix should filter states with undesirable hole positions and give correct diagonal values."""
    for num_rungs in [4, 5, 6]:
        for num_holes in range(4):
            max_num_operators = 3  # We check all permutations, so max_num_operators cant go much higher.
            for fixed_distances, free_shift in _yield_distances(2 * num_rungs, max_num_operators):
                config = lc.Limited_Position_Correlation_Config(
                    hamiltonian={
                        "num_rungs": num_rungs,
                        "num_holes": num_holes,
                        "weights": {"tl": 0.45, "tr": 0.36, "jl": 0.15, "jr": 0.12},
                        "symmetry_qs": {"leg": 0, "rung": 0},
                        "num_threads": 8,
                    },
                    correlations={"name": "Sz_correlations", "fixed_distances": fixed_distances},
                )
                matrix = config.setup_excitation_operator(free_shift)[0]()
                state_translator, basis_map, symmetries = config.hamiltonian.get_translators()
                relative_distances = [shift.to_npint32() for shift in config.correlations.fixed_distances] + [
                    free_shift.to_npint32()
                ]
                get_element_value = functools.partial(
                    _get_sz_value,
                    relative_distances=relative_distances,
                    state_translator=state_translator,
                    basis_map=basis_map,
                    symmetries=symmetries,
                )
                assert _is_matrix_sz(matrix, get_element_value, num_holes), (config, free_shift)


def _get_sz_value(dense_index, relative_distances, state_translator, basis_map, symmetries):
    # TODO: Can it be unified with holes?
    SPIN_DOWN_VALUE = 2  # FIXME: Change it to be importable into regular python files.
    SPIN_UP_VALUE = 1
    sparse_index = basis_map.get_sparse(dense_index)
    sparse_state = state_translator.sparse_index_to_state(sparse_index)
    matrix_value = 0
    for initial_index in range(len(sparse_state)):
        if sparse_state[initial_index] == SPIN_UP_VALUE:
            matrix_value_contribution = 1
        elif sparse_state[initial_index] == SPIN_DOWN_VALUE:
            matrix_value_contribution = -1
        else:
            continue
        for i, relative_distance in enumerate(relative_distances):
            expected_pos = symmetries.get_index_by_relative_shift(initial_index, relative_distance)
            found_value = sparse_state[expected_pos]
            if found_value == SPIN_DOWN_VALUE:
                matrix_value_contribution *= -1
            elif found_value != SPIN_UP_VALUE:
                matrix_value_contribution = 0
                break
        matrix_value += matrix_value_contribution
    return matrix_value


def _is_matrix_sz(matrix: Sparse_Matrix, get_element_value: typing.Callable, max_value: int) -> bool:
    # Only checks for projections diagonal in the matrix's basis.
    # square matrix
    # all nonzero values are 1
    # nonzero values only on diagonal
    # each row has a value decided by is_state_in_subspace
    if matrix.shape[0] != matrix.shape[1]:
        return False
    # The matrix cannot be empty so a single zero element is saved if there are no non-zero elements.
    is_empty = matrix.vals.tolist() == [0] and matrix.major_pointers[-1] == 1
    for dense_index, major_pointer in enumerate(matrix.major_pointers[:-1]):
        end = matrix.major_pointers[dense_index + 1]
        expected_value = get_element_value(dense_index)
        if major_pointer == end:
            if expected_value != 0:
                return False
            continue
        if end != major_pointer + 1:
            return False
        if not is_empty and dense_index != matrix.minor_inds[major_pointer]:
            return False
        if (is_empty and expected_value != 0) or expected_value != matrix.vals[matrix.value_inds[major_pointer]]:
            return False
    return True
