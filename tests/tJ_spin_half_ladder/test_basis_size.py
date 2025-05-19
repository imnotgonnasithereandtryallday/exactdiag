import pathlib
import json
import glob
from typing import Self
from pydantic.dataclasses import dataclass


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


def test_negative_k():
    pass


def test_sum_over_k():
    pass


def test_sum_over_holes():
    pass


def test_sum_over_down_spins():
    pass


def test_equivalent_systems():
    pass


def test_inter_symmetry():
    """General and specialized symmetries should give the same basis sizes."""

def collect_all():
    folder = pathlib.Path(__file__).parent / "precalculated"
    return glob.glob(f"{(folder / '*')!s}")


def collect(num_rungs: int, num_holes: int, num_down_spins: int):
    folder = pathlib.Path(__file__).parent / "precalculated"
    mkx = num_rungs // 2
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
