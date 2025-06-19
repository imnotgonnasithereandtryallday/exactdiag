"""Contains classes for configuring calculations. using
an anisotropic spin-1/2 nearest-neighbour t-J Hamiltonian on a ladder lattice.
"""  # noqa: D205

from dataclasses import asdict
import pathlib
from typing import Literal, override
import logging

import numpy as np
from pydantic.dataclasses import dataclass, Field

from exactdiag.general.matrix_rules_utils import set_combinatorics
import exactdiag.general.configs as gc
from exactdiag.general.configs import Eigenpair_Config  # noqa: F401 - Reexporting for convenience.
from exactdiag.general.types_and_constants import MATRIX_INDEX_TYPE, VALUE_INDEX_TYPE
from exactdiag.general.sparse_matrices import Sparse_Matrix
from exactdiag.tJ_spin_half_ladder.matrix_setup import (
    setup_hamiltonian,
    py_get_ladder_translators,
    setup_excitation_operator,
)

_logger = logging.getLogger(__name__)


@dataclass(config={"validate_assignment": True, "extra": "forbid"})
class Weights(gc.Weights_Base):
    """Weights of the Hamiltonian terms."""

    tl: float
    tr: float
    jl: float
    jr: float


@dataclass(config={"validate_assignment": True, "extra": "forbid"})
class Quantum_Numbers(gc.Quantum_Numbers_Base):  # noqa: D101 docstring inherited
    leg: int
    rung: Literal[0, 1]


@dataclass(config={"arbitrary_types_allowed": True, "validate_assignment": True, "extra": "forbid"})
class Hamiltonian_Config(gc.Hamiltonian_Config_Base):
    r"""Information necessary to construct a Hamiltonian.

    symmetry_qs: [conserved quasi-momentum along the leg, mirror symmetry parity] of the Hamiltonian block or None.
        The momentum is in the units of $2\pi/a$, parity is 0 for even, 1 for odd.
        If None, indicates that momenta in range(num_rungs//2) and both parities should be considered.
    total_spin_projection: value of the conserved spin of the Hamiltonian block
        calculated as a difference of the numbers of up and down spins, i.e $sum_i 2S^z_i$.

    If you wish to save the results of your calculations to a different location,
    overwrite the get_calc_folder method.
    """

    # TODO: decouple tj model from ladder clusters
    # TODO: Collect sparse basis conserved qunatities into separate class?
    num_rungs: int
    num_holes: int
    weights: Weights
    symmetry_qs: Quantum_Numbers | None = None
    total_spin_projection: int = None  # noqa # TODO: How to specify that the value is filled in if not given but the initialized object always has it?
    combinatorics_table: np.ndarray = Field(init=False)

    # TODO: How to handle validations when changing num_holes or total_spin_projection?
    def __post_init__(self):
        if self.total_spin_projection is None:
            self.total_spin_projection = self._get_default_spin_projection()
        elif (self.num_spins - self.total_spin_projection) % 2 != 0:
            raise ValueError(
                f"Incompatible total_spin_projection {self.total_spin_projection}/2 with {self.num_spins} spins."
            )
        self.combinatorics_table = set_combinatorics(self.num_nodes)

    def __repr__(self):
        attrs = asdict(self)
        attrs = {k:v for k,v in attrs.items() if v is not None}
        attrs["combinatorics_table"] = "shape[{self.combinatorics_table.shape}]"
        attrs_string = ", ".join(f"{k}={v}" for k,v in attrs.items())
        return f"{type(self).__name__}({attrs_string})"

    def _get_default_spin_projection(self) -> int:
        return self.num_spins % 2

    @property
    def num_spin_states(self) -> int:
        return self.combinatorics_table[self.num_down_spins, self.num_spins]

    @property
    def num_spins(self) -> int:
        return self.num_nodes - self.num_holes

    @property
    def num_down_spins(self) -> int:
        return (self.num_spins - self.total_spin_projection) // 2

    @property
    def periodicities(self) -> tuple[int, int]:
        return (self.num_rungs, 2)

    @property
    def num_nodes(self) -> int:
        return 2 * self.num_rungs

    def _get_relative_path(self) -> pathlib.Path:
        if self.symmetry_qs is None:
            raise ValueError("symmetry_qs must not be None to specify a path.")
        rungs_holes = f"rungs{self.num_rungs}_holes{self.num_holes}"
        qs_spins = f"kx{self.symmetry_qs.leg}_ky{self.symmetry_qs.rung}_down_spins{self.num_down_spins}"
        folder = pathlib.Path(rungs_holes) / qs_spins
        return folder

    @override
    def get_translators(self):  # noqa: D102 docstring inherited
        # TODO: Make the typing visible to python.
        # NOTE: This is here as a first step to decouple matrix_setup from cluster info.
        return py_get_ladder_translators(self)

    @override
    def setup(self):  # noqa: D102 docstring inherited
        return setup_hamiltonian(self)

    @override
    def get_symmetry_info(self):  # noqa: D102 docstring inherited
        symmetry_qs = None if self.symmetry_qs is None else [int(i) for i in self.symmetry_qs.to_npint32()]
        dic = {
            "num_rungs": self.num_rungs,
            "num_holes": self.num_holes,
            "symmetry_qs": symmetry_qs,
        }
        if self.total_spin_projection != self._get_default_spin_projection():
            dic["total_spin_projection"] = self.total_spin_projection
        return dic


@dataclass(config={"validate_assignment": True, "extra": "forbid"})
class Spectrum_Config(gc.Spectrum_Config_Base):
    r"""Information necessary, in addition to a Hamiltonian_Config instance, to calculate an operator spectrum.

    operator_symmetry_qs: [conserved quasi-momentum along the leg, mirror symmetry parity] of the operator block or None.
        The momentum is in the units of $2\pi/a$, parity is 0 for even, 1 for odd.
        If None, indicates that momenta in range(num_rungs//2) and both parities should be considered.
    num_threads: Setting to None indicates to copy the attribute from hamiltonian config when put into Config.
    """

    # TODO: There must be a better way to copy the num_threads.
    name: str  # TODO: to enum
    omega_max: float
    operator_symmetry_qs: Quantum_Numbers | None
    broadening: float
    omega_min: float = 0
    omega_steps: int = 1000
    num_lanczos_vecs: int = 250

    def get_omegas(self):
        return np.linspace(self.omega_min, self.omega_max, self.omega_steps)

@dataclass(config={"validate_assignment": True, "extra": "forbid"})
class Position_Correlation_Config(gc.Spectrum_Config_Base):
    # FIXME: add docstring
    name: str
    fixed_distances: None

@dataclass
class Config(gc.Combined_Config_Base):  # noqa: D101 - docstring inherited.
    hamiltonian: Hamiltonian_Config
    spectrum: Spectrum_Config | None = None

    def __post_init__(self):
        super().__post_init__()
        _logger.info(f"Config loaded:\n{_parse_hamiltonian_kwargs_intro_message(self.hamiltonian)}")

    @override
    def get_spectrum_path(self, operator_name_suffix: str = ""):  # noqa: D102 - docstring inherited.
        if self.spectrum is None:
            raise ValueError("Spectrum_Config is required.")
        system_folder = self.hamiltonian.get_calc_folder()
        folder = system_folder / f"{self.spectrum.name}_spectra"
        suffix = _get_spectrum_figure_suffix(self.hamiltonian, self.spectrum, operator_name_suffix)
        path = folder / f"{self.spectrum.name}{suffix}.npz"
        return path

    @override
    def setup_excitation_operator(self):  # noqa: D102 - docstring inherited.
        if self.spectrum is None:
            raise ValueError("Spectrum_Config is required.")
        return setup_excitation_operator(initial_config=self)


@dataclass
class Combined_Position_Config(gc.Combined_Config_Base):  # noqa: D101 - docstring inherited.
    hamiltonian: Hamiltonian_Config
    correlation: Position_Correlation_Config | None = None
    # FIXME: What is the best workflow here?


def _parse_hamiltonian_kwargs_intro_message(config: Hamiltonian_Config) -> str:
    first_line = f"{config.num_rungs} rungs, {config.num_holes} holes, k={config.symmetry_qs}, Sz={config.total_spin_projection}/2"
    spin_projection_states = config.combinatorics_table[config.num_down_spins, config.num_nodes - config.num_holes]
    hole_states = config.combinatorics_table[config.num_holes, config.num_nodes]
    total_projection_states = spin_projection_states * hole_states
    momentum_states_approx = total_projection_states // config.num_nodes
    second_line = f"number of states satisfying spin projection = {total_projection_states}, number of states further satifying momentum ~ {momentum_states_approx}"
    momentum_states_per_Gb = momentum_states_approx / 1e9  # noqa: N806
    eigenvectors_size = 2 * 64 * momentum_states_per_Gb / 8
    elements_in_column = 3 * config.num_nodes
    index_element_size = np.empty(0, dtype=MATRIX_INDEX_TYPE).itemsize  # TODO: how to use it on the dtype directly?
    value_element_size = np.empty(0, dtype=VALUE_INDEX_TYPE).itemsize  # TODO: how to use it on the dtype directly?
    hamiltonian_size = (
        (elements_in_column + 1) * index_element_size + elements_in_column * value_element_size
    ) * momentum_states_per_Gb
    basis_size = momentum_states_per_Gb * (index_element_size + 4)
    third_line = f"single eigenvector size: {eigenvectors_size:.2f} GB, full Hamiltonian size < {hamiltonian_size:.2f} GB, basis size: {basis_size:.2f} GB"
    full_string = f"{first_line}\n{second_line}" + (
        f"\n{third_line}" if np.amax([hamiltonian_size, eigenvectors_size, basis_size]) > 1 else ""
    )
    return full_string


def _get_spectrum_figure_suffix(
    hamiltonian_config: Hamiltonian_Config, spectrum_config: Spectrum_Config, operator_name_suffix: str
):
    weight_string = "_".join([f"{name}{weight}" for name, weight in asdict(hamiltonian_config.weights).items()])
    operator_name_suffix = f"_{operator_name_suffix}" if operator_name_suffix else ""
    if spectrum_config.operator_symmetry_qs is None:
        q_string = ""
    else:
        q_string = f"_q{spectrum_config.operator_symmetry_qs.to_npint32()}"
    ws_string = f"_ws{spectrum_config.omega_min:g},{spectrum_config.omega_max:g},{spectrum_config.omega_steps:g}"
    broad_string = f"_broad{spectrum_config.broadening}"
    vecs_string = f"_vecs{spectrum_config.num_lanczos_vecs}"
    suffix = f"{operator_name_suffix}{q_string}{ws_string}{broad_string}{vecs_string}_{weight_string}"
    return suffix
