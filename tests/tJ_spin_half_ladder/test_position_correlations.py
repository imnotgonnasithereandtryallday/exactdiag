import json
import pathlib

import numpy as np

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
        config = lc.Combined_Position_Config(**d)
        spectrum = api.get_position_correlations(config)
        probabilities = spectrum.spectrum.flatten()
        if normalize:
            # The sum of probabilities does not sum to the number of holes if there are fixed holes:
            # it sums to the probability that the holes are in the fixed coordination
            # times some combinatorics based on number of holes.
            # We normalize the fixed probabilities to 1 as they are not given in the articles.
            probabilities /= probabilities[0]
        diffs.append(probabilities[:len(target)] - target)
    return diffs


def test_singlet_singlet_correlations_against_precalculated():
    """Calculated singlet-singlet correlations should match values published in my dissertation."""
    # NOTE: Takes around 15 minutes on my pc.
    folder = pathlib.Path(__file__).parent / "precalculated"
    diffs = _get_correlation_diffs(folder / "singlet_correlations_scco.txt", normalize=False)
    listed_precision = 1e-3
    passed = [np.all(np.abs(diff) < listed_precision) for diff in diffs]
    assert all(passed), passed
