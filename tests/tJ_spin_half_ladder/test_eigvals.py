import numpy as np

from exactdiag.general.lanczos_diagonalization import get_lowest_eigenpairs
from tests.tJ_spin_half_ladder.test_basis_size import collect, collect_all, Precalculated

EIGENVALUE_TOLERANCE = 1e-8


def test_against_precalculated():
    """The calculation should reproduce eigenvalues previously deemed correct."""
    paths = collect_all()
    expected_num_files = 51
    num_files = 0
    for file in paths:
        pre = Precalculated.load(file)
        if pre.eigvals is None:
            continue
        num_files += 1
        config = pre.to_he_config()
        vals, vecs = get_lowest_eigenpairs(config)
        assert np.all(np.abs(vals - pre.eigvals) < EIGENVALUE_TOLERANCE), np.abs(vals - pre.eigvals)
    assert num_files == expected_num_files


def test_negative_k_leg():
    """Eigenvalues should be the same for k as for (-k[0], k[1])."""
    # The system chosen to have varied number of states for different momenta.
    collected = collect(num_rungs=6, num_holes=0, num_down_spins=6)
    expected_num_files = 8
    assert len(collected) == expected_num_files
    for pre in collected:
        k = pre.symmetry_qs
        if k[0] == 0:
            continue
        config = pre.to_he_config()
        config.hamiltonian.symmetry_qs.leg *= -1
        vals, vecs = get_lowest_eigenpairs(config)
        assert np.all(np.abs(vals - pre.eigvals) < EIGENVALUE_TOLERANCE), np.abs(vals - pre.eigvals)


def test_k_leg_periodicity():
    """Eigenvalues should remain the same when shifting k by a multiple of its periodicity."""
    # NOTE: We do not test config.symmetry_qs.rung, as that is validated to be in {0, 1}.
    # The system chosen to have varied number of states for different momenta.
    collected = collect(num_rungs=6, num_holes=0, num_down_spins=6)
    expected_num_files = 8
    assert len(collected) == expected_num_files
    multiples = range(-4, 5)
    for pre in collected:
        k = pre.symmetry_qs
        if k == (0, 0):
            continue
        config = pre.to_he_config()
        periodicities = tuple(config.hamiltonian.periodicities)
        for leg_multiple in multiples:
            config.hamiltonian.symmetry_qs.leg = k[0] + leg_multiple * periodicities[0]
            vals, vecs = get_lowest_eigenpairs(config)
            assert np.all(np.abs(vals - pre.eigvals) < EIGENVALUE_TOLERANCE), np.abs(vals - pre.eigvals)


def test_equivalent_symmetries():
    pass
