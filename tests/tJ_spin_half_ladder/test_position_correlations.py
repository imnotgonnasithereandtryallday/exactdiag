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
    folder = pathlib.Path(__file__).parent / "precalculated"
    diffs = _get_correlation_diffs(folder / "hole_correlations_paulina.txt", normalize=True)
    listed_precision = 1e-3
    assert all([np.all(np.abs(diff) < listed_precision) for diff in diffs])


def test_hole_correlations_against_precalculated_scco():
    """Calculated 2- and 3-hole correlations should match values in hte SCCO paper."""
    folder = pathlib.Path(__file__).parent / "precalculated"
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
    folder = pathlib.Path(__file__).parent / "precalculated"
    diffs = _get_correlation_diffs(folder / "singlet_correlations_scco.txt", normalize=False)
    listed_precision = 1e-3
    passed = [np.all(np.abs(diff) < listed_precision) for diff in diffs]
    assert all(passed), passed


def test_hole_matrix():
    """Matrix of projection to states should filter states with undesirable hole positions."""
    # TODO: generalize and run for more configs.
    config = lc.Limited_Position_Correlation_Config(
        hamiltonian={
            "num_rungs": 8,
            "num_holes": 2,
            "weights": {"tl": 0.45, "tr": 0.36, "jl": 0.15, "jr": 0.12},
            "symmetry_qs": {"leg": 0, "rung": 0},
            "num_threads": 8,
        },
        correlations={"name": "hole_correlations", "fixed_distances": []},
    )
    free_shift = lc.Position_Shift(leg=1, rung=1)
    matrix = config.setup_excitation_operator(free_shift)[0]()
    state_translator, basis_map, symmetries = config.hamiltonian.get_translators()
    relative_distances = [shift.to_npint32() for shift in config.correlations.fixed_distances] + [
        free_shift.to_npint32()
    ]

    def is_state_in_subspace(dense_index):
        sparse_index = basis_map.get_sparse(dense_index)
        sparse_state = state_translator.sparse_index_to_state(sparse_index)
        HOLE_VALUE = 0  # FIXME: Change it to be importable into regular python files.
        hole_positions = [i for i, v in enumerate(sparse_state) if v == HOLE_VALUE]
        if not hole_positions:
            return False
        for relative_distance in relative_distances:
            hole_pos = symmetries.get_index_by_relative_shift(hole_positions[0], relative_distance)  # FIXME: It is not always relative to the first!
            if hole_pos not in hole_positions[1:]:
                return False
        return True

    assert _is_matrix_projection(matrix, is_state_in_subspace)


def test_sz_matrix():
    pass


def _is_matrix_projection(matrix: Sparse_Matrix, is_state_in_subspace: typing.Callable) -> bool:
    # Only checks for projections diagonal in the matrix's basis.
    # square matrix
    # all nonzero values are 1
    # nonzero values only on diagonal
    # each row has a value decided by is_state_in_subspace
    if matrix.shape[0] != matrix.shape[1]:
        return False
    if matrix.vals.tolist() != [1]:
        return False
    for dense_index, major_pointer in enumerate(matrix.major_pointers[:-1]):
        end = matrix.major_pointers[dense_index + 1]
        if major_pointer == end:
            continue
        if end != major_pointer + 1:
            return False
        if dense_index != matrix.minor_inds[major_pointer]:
            return False
        if not is_state_in_subspace(dense_index):
            return False
    return True
