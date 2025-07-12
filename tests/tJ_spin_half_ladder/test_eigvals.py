import numpy as np

from exactdiag.general.lanczos_diagonalization import get_lowest_eigenpairs
from tests.tJ_spin_half_ladder.test_basis_size import collect_all, Precalculated


def test_against_precalculated():
    """The calculation should reproduce eigenvalues previously deemed correct."""
    paths = collect_all()
    expected_num_files = 4
    count = 0
    for file in paths:
        pre = Precalculated.load(file)
        if pre.eigvals is None:
            continue
        count += 1
        config = pre.to_he_config()
        vals, vecs = get_lowest_eigenpairs(config)
        assert np.all(np.abs(vals - pre.eigvals) < 1e-8), np.abs(vals - pre.eigvals)
    assert count == expected_num_files


def test_equivalent_systems():
    pass
