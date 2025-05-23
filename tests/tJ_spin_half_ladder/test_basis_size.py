import pathlib
import json
import glob
from typing import Self
from pydantic.dataclasses import dataclass

import scipy.special

import exactdiag.tJ_spin_half_ladder.configs as lc


def test_against_precalculated():
    """The calculation should reproduce basis sizes previously deemed correct."""
    collected = collect_all()
    expected_num_files = 68
    assert len(collected) == expected_num_files
    for file in collected:
        pre = Precalculated.load(file)
        config = pre.to_hamiltonian_config()
        _, py_basis_map, _ = config.get_translators()
        assert pre.basis_size == py_basis_map.get_num_states(), (file, pre.basis_size, py_basis_map.get_num_states())


def test_negative_k_leg():
    """Basis size should be the same for k as for (-k[0], k[1])."""
    # The system chosen to have varied number of states for different momenta.
    collected = collect(num_rungs=6, num_holes=0, num_down_spins=6)
    expected_num_files = 8
    assert len(collected) == expected_num_files
    for params in collected:
        k = params.symmetry_qs
        if k[0] == 0:
            continue
        config = params.to_hamiltonian_config()
        config.symmetry_qs.leg *= -1
        _, py_basis_map, _ = config.get_translators()
        assert py_basis_map.get_num_states() == params.basis_size


def test_k_leg_periodicity():
    """Basis size should remain the same when shifting k by a multiple of its periodicity."""
    # NOTE: We do not test config.symmetry_qs.rung, as that is validated to be in {0, 1}.
    # The system chosen to have varied number of states for different momenta.
    collected = collect(num_rungs=6, num_holes=0, num_down_spins=6)
    expected_num_files = 8
    assert len(collected) == expected_num_files
    multiples = range(-4, 5)
    for params in collected:
        k = params.symmetry_qs
        if k == (0, 0):
            continue
        config = params.to_hamiltonian_config()
        periodicities = tuple(config.periodicities)
        for leg_multiple in multiples:
            config.symmetry_qs.leg = k[0] + leg_multiple * periodicities[0]
            _, py_basis_map, _ = config.get_translators()
            assert py_basis_map.get_num_states() == params.basis_size


def test_sum_over_k():
    """Summed number of states should give values calculated using combinatorics."""
    # NOTE: We test the saved files, nothing is calculated.
    # We assume that there are no files with negative or reducible momenta.
    # Counting negative momenta: all our files contribute twice except
    # those with k_leg == 0 or (k_leg == num_rungs//2 if num_rungs is even).
    collected = collect_all()
    rungs_holes_down_to_pre = {}
    for fpath in collected:
        pre = Precalculated.load(fpath)
        rungs_holes_down = (pre.num_rungs, pre.num_holes, pre.num_down_spins)
        if rungs_holes_down not in rungs_holes_down_to_pre:
            rungs_holes_down_to_pre[rungs_holes_down] = []
        multiplier = 1 + (pre.symmetry_qs[0] != 0 and (pre.symmetry_qs[0] != pre.num_rungs // 2 or pre.num_rungs % 2))
        rungs_holes_down_to_pre[rungs_holes_down].append(multiplier * pre.basis_size)

    rungs_holes_down_to_pre = {
        k: sizes for k, sizes in rungs_holes_down_to_pre.items() if len(sizes) == 2 * (k[0] // 2 + 1)
    }
    expected_num_sets = 7
    assert len(rungs_holes_down_to_pre) == expected_num_sets
    for (num_rungs, num_holes, num_down_spins), sizes in rungs_holes_down_to_pre.items():
        num_spin_states = scipy.special.comb(2 * num_rungs - num_holes, num_down_spins, exact=True)
        num_hole_states = scipy.special.comb(2 * num_rungs, num_holes, exact=True)
        assert num_spin_states * num_hole_states == sum(sizes)


def test_sum_over_holes():
    pass


def test_sum_over_down_spins():
    pass


def test_equivalent_systems():
    pass


def test_inter_symmetry():
    """General and specialized symmetries should give the same basis sizes."""


def collect_all():
    # FIXME: Shares name with collect() but the return type differs.
    folder = pathlib.Path(__file__).parent / "precalculated"
    return glob.glob(f"{(folder / '*')!s}")


def collect(num_rungs: int, num_holes: int, num_down_spins: int):
    folder = pathlib.Path(__file__).parent / "precalculated"
    mkx = num_rungs // 2 + 1
    mky = 2
    collected = []
    for kx in range(mkx):
        for ky in range(mky):
            file = f"rungs{num_rungs}_holes{num_holes}_k[{kx},{ky}]_down{num_down_spins}.txt"
            collected.append(Precalculated.load(folder / file))
    return collected


@dataclass
class Precalculated:
    num_rungs: int
    num_holes: int
    num_down_spins: int
    symmetry_qs: tuple[int, int]
    basis_size: int
    weights: lc.Weights | None = None  # TODO: wrap eigvals and weights into a class?
    eigvals: tuple[float, ...] | None = None

    @classmethod
    def load(cls, path: pathlib.Path | str) -> Self:
        with open(path, mode="r", encoding="utf-8") as fp:
            d = json.load(fp)
        return cls(**d)

    def to_hamiltonian_config(self) -> lc.Hamiltonian_Config:
        weights = self.weights if self.weights is not None else lc.Weights(tl=1, tr=1, jl=1, jr=1)
        num_spins = 2 * self.num_rungs - self.num_holes
        total_spin_projection = num_spins - 2 * self.num_down_spins
        symmetry_qs = lc.Quantum_Numbers(leg=self.symmetry_qs[0], rung=self.symmetry_qs[1])
        return lc.Hamiltonian_Config(
            num_rungs=self.num_rungs,
            num_holes=self.num_holes,
            num_threads=10,
            weights=weights,
            symmetry_qs=symmetry_qs,
            total_spin_projection=total_spin_projection,
        )

    def to_he_config(self) -> lc.Config:
        h_config = self.to_hamiltonian_config()
        e_config = lc.Eigenpair_Config(
            num_eigenpairs=len(self.eigvals),
            num_threads=10,
        )
        return lc.Config(
            hamiltonian=h_config,
            eigenpair=e_config,
        )
