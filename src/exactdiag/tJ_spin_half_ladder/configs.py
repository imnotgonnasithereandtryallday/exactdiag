"""Contains classes for configuring calculations. using
an anisotropic spin-1/2 nearest-neighbour t-J Hamiltonian on a ladder lattice.
"""  # noqa: D205

from dataclasses import asdict
import enum
import logging
import pathlib
from typing import Literal, override

import numpy as np
import numpydantic
from pydantic import field_validator
from pydantic.dataclasses import dataclass, Field

from exactdiag.general.matrix_rules_utils import set_combinatorics
import exactdiag.general.configs as gc
from exactdiag.general.types_and_constants import MATRIX_INDEX_TYPE, VALUE_INDEX_TYPE
from exactdiag.tJ_spin_half_ladder import matrix_setup

_logger = logging.getLogger(__name__)


@dataclass(kw_only=True, config={"validate_assignment": True, "extra": "forbid"})
class Weights(gc.Weights_Base):
    """Weights of the Hamiltonian terms."""

    tl: float
    tr: float
    jl: float
    jr: float


@dataclass(kw_only=True, config={"validate_assignment": True, "extra": "forbid"})
class Quantum_Numbers(gc.Quantum_Numbers_Base):  # noqa: D101 docstring inherited
    leg: int
    rung: Literal[0, 1]


@dataclass(kw_only=True, config={"validate_assignment": True, "extra": "forbid"})
class Hamiltonian_Config(gc.Hamiltonian_Config_Base[Quantum_Numbers]):
    r"""Information necessary to construct a Hamiltonian.

    symmetry_qs: [conserved quasi-momentum along the leg, mirror symmetry parity] of the Hamiltonian block or None.
        The momentum is in the units of $2\pi/a$, parity is 0 for even, 1 for odd.
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
    total_spin_projection: int = None  # noqa # TODO: How to specify that the value is filled in if not given but the initialized object always has it?
    combinatorics_table: numpydantic.NDArray[
        numpydantic.Shape["* num_nodes_plus_one, * num_nodes_plus_one"], int  # noqa: F722 - Not a forward type annotation.
    ] = Field(init=False)

    @field_validator("weights", mode="before")
    @classmethod
    def _validate_weights(cls, value):
        # TODO: Can this be defined on the Weights class?
        if isinstance(value, Weights):
            return value
        return Weights(**value)

    @field_validator("symmetry_qs", mode="before")
    @classmethod
    def _validate_qs(cls, value):
        if isinstance(value, Quantum_Numbers):
            return value
        return Quantum_Numbers(**value)

    # TODO: How to handle validations when changing num_holes or total_spin_projection?
    def __post_init__(self):
        if self.total_spin_projection is None:
            self.total_spin_projection = self._get_default_spin_projection()
        elif (self.num_spins - self.total_spin_projection) % 2 != 0:
            raise ValueError(
                f"Incompatible total_spin_projection {self.total_spin_projection}/2 with {self.num_spins} spins."
            )
        self.combinatorics_table = set_combinatorics(self.num_nodes)
        _logger.info(f"Config created:\n{_parse_hamiltonian_kwargs_intro_message(self)}")

    def __repr__(self):
        attrs = asdict(self)
        attrs = {k: v for k, v in attrs.items() if v is not None}
        attrs["combinatorics_table"] = "shape[{self.combinatorics_table.shape}]"
        attrs_string = ", ".join(f"{k}={v}" for k, v in attrs.items())
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
    def get_translators(self):
        # TODO: Make the typing visible to python.
        # NOTE: This is here as a first step to decouple matrix_setup from cluster info.
        return matrix_setup.py_get_ladder_translators(self)

    @override
    def setup(self):
        return matrix_setup.setup_hamiltonian(self)

    @override
    def get_symmetry_info(self):
        symmetry_qs = None if self.symmetry_qs is None else [int(i) for i in self.symmetry_qs.to_npint32()]
        dic = {
            "num_rungs": self.num_rungs,
            "num_holes": self.num_holes,
            "symmetry_qs": symmetry_qs,
        }
        if self.total_spin_projection != self._get_default_spin_projection():
            dic["total_spin_projection"] = self.total_spin_projection
        return dic


@dataclass(kw_only=True, config={"validate_assignment": True, "extra": "forbid"})
class Eigenpair_Config(gc.Eigenpair_Config[Hamiltonian_Config]):  # noqa: D101 - docstring inherited.
    # Mostly just a type alias but pydantic does not validate the Hamiltonian_Config automatically.
    # So the full configs have to inherit it again.

    @field_validator("hamiltonian", mode="before")
    @classmethod
    def _validate_hamiltonian(cls, value):
        if isinstance(value, Hamiltonian_Config):
            return value
        return Hamiltonian_Config(**value)


class Spectrum_Name(enum.StrEnum):
    """Names of the available operators."""

    CURRENT_RUNG = enum.auto()
    CURRENT_LEG = enum.auto()
    SZQ = "Szq"
    SPECTRAL_FUNCTION_PLUS = enum.auto()
    SPECTRAL_FUNCTION_MINUS = enum.auto()
    OFFIDAG_SPEC_FUNC_PLUS = "offdiagonal_spectral_function_plus"
    OFFIDAG_SPEC_FUNC_MINUS = "offdiagonal_spectral_function_minus"


@dataclass(kw_only=True, config={"validate_assignment": True, "extra": "forbid"})
class Spectrum_Part(gc.Spectrum_Part[Spectrum_Name, Quantum_Numbers]):  # noqa: D101 - docstring inherited.
    # Just a type alias but pydantic does not validate name and operator_symmetry_qs automatically.
    @field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, value):
        if isinstance(value, Spectrum_Name):
            return value
        return Spectrum_Name(value)

    @field_validator("operator_symmetry_qs", mode="before")
    @classmethod
    def _validate_qs(cls, value):
        if isinstance(value, Quantum_Numbers):
            return value
        return Quantum_Numbers(**value)


@dataclass(kw_only=True, config={"validate_assignment": True, "extra": "forbid"})
class Limited_Spectrum_Config(gc.Limited_Spectrum_Config_Base[Hamiltonian_Config, Spectrum_Part]):  # noqa: D101 - docstring inherited.
    @field_validator("hamiltonian", mode="before")
    @classmethod
    def _validate_hamiltonian(cls, value):
        if isinstance(value, Hamiltonian_Config):
            return value
        return Hamiltonian_Config(**value)

    @field_validator("spectrum", mode="before")
    @classmethod
    def _validate_spectrum(cls, value):
        if isinstance(value, Spectrum_Part):
            return value
        return Spectrum_Part(**value)

    @override
    def setup_excitation_operator(self):
        return matrix_setup.setup_excitation_operator(self)


@dataclass(kw_only=True, config={"validate_assignment": True, "extra": "forbid"})
class Full_Spectrum_Config(  # noqa: D101 - docstring inherited
    gc.Full_Spectrum_Config_Base[Hamiltonian_Config, Spectrum_Part],
    Limited_Spectrum_Config,
    Eigenpair_Config,
):
    # @field_validator("eigenpair", mode="before")
    # @classmethod
    # def _validate_eigenpair(cls, value):
    #     if isinstance(value, Eigenpair_Part):
    #         return value
    #     return Eigenpair_Config(**value)

    @override
    def get_spectrum_path(self, operator_name_suffix=""):
        system_folder = self.hamiltonian.get_calc_folder()
        folder = system_folder / f"{self.spectrum.name}_spectra"
        suffix = _get_spectrum_figure_suffix(self.hamiltonian, self.spectrum, operator_name_suffix)
        path = folder / f"{self.spectrum.name}{suffix}.npz"
        return path


class Position_Correlation_Name(enum.StrEnum):
    """Names of the available operators."""

    HOLE_CORRELATIONS = "hole_correlations"
    SZ_CORRELATIONS = "Sz_correlations"
    SINGLET_SINGLET = "singlet-singlet"


@dataclass(kw_only=True, config={"validate_assignment": True, "extra": "forbid"})
class Position_Shift(gc.Position_Shift_Base):  # noqa: D101 docstring inherited
    leg: int
    rung: Literal[0, 1]


@dataclass(kw_only=True, config={"validate_assignment": True, "extra": "forbid"})
class Position_Correlation_Part(gc.Position_Correlation_Part[Position_Correlation_Name, Position_Shift]):
    """Information necessary calculate a position correlation operator.

    fixed_distances: fixed relative shifts between operator terms.
        For hole- and Sz-correlations, a state is projected using len(fixed_distances)+1 number operators.
        Their poistions relative to the first one are defined by fixed_distances.
        Then a number operator (times some combinatorics based on the total number of particles) is evaluated
        for each position separately.
        For singlet-singlet correlations, fixed_distances defines the structure of the two singlet operators;
        all relative positions of the singlet operators are computed.
    num_threads: Setting to None indicates to copy the attribute from hamiltonian config when put into Config.
    """

    @field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, value):
        if isinstance(value, Position_Correlation_Name):
            return value
        return Position_Correlation_Name(value)

    @field_validator("fixed_distances", mode="before")
    @classmethod
    def _validate_distances(cls, value):
        return [e if isinstance(e, Position_Shift) else Position_Shift(**e) for e in value]


@dataclass(kw_only=True, config={"validate_assignment": True, "extra": "forbid"})
class Limited_Position_Correlation_Config(  # noqa: D101 docstring inherited
    gc.Limited_Position_Correlation_Config_Base[Hamiltonian_Config, Position_Correlation_Name, Position_Shift]
):
    @field_validator("hamiltonian", mode="before")
    @classmethod
    def _validate_hamiltonian(cls, value):
        if isinstance(value, Hamiltonian_Config):
            return value
        return Hamiltonian_Config(**value)

    @field_validator("correlations", mode="before")
    @classmethod
    def _validate_correlations(cls, value):
        if isinstance(value, Position_Correlation_Part):
            return value
        return Position_Correlation_Part(**value)

    @override
    def setup_excitation_operator(self, free_shift):
        """Return callables to set up the excitation operator.

        Returns a callable to get the operator, info about its final symmetry block,
        and a callable to construct the Hamiltonian in that block.

        Args:
        free_shift: relative position of the operators not specified by the fixed shifts of the Hamiltonian.
                    For `n`-hole correlations, `n-1` relative positions are fixed and one is free.
                    For singlet-singlet correlations, the intra-singlet structure is fixed
                    and the relative position of the singlet operators is free.

        """
        if self.name != Position_Correlation_Name.SINGLET_SINGLET:
            raise NotImplementedError()  # FIXME: Implement.
        return matrix_setup.get_position_correlation_operator(self, free_shift)


@dataclass(kw_only=True, config={"validate_assignment": True, "extra": "forbid"})
class Full_Position_Correlation_Config(  # noqa: D101 - docstring inherited.
    gc.Full_Position_Correlation_Config_Base[Hamiltonian_Config, Position_Correlation_Name, Position_Shift],
    Limited_Position_Correlation_Config,
    Eigenpair_Config,
):
    # @field_validator("eigenpair", mode="before")
    # @classmethod
    # def _validate_eigenpair(cls, value):
    #     if isinstance(value, Eigenpair_Config):
    #         return value
    #     return Eigenpair_Config(**value)

    @override
    def get_spectrum_path(self, operator_name_suffix=""):
        system_folder = self.hamiltonian.get_calc_folder()
        folder = system_folder / f"{self.correlations.name}_spectra"
        suffix = _get_position_correlation_figure_suffix(self.hamiltonian, self.correlations, operator_name_suffix)
        path = folder / f"{self.correlations.name}{suffix}.npz"
        return path


def _parse_hamiltonian_kwargs_intro_message(config: Hamiltonian_Config) -> str:
    first_line = (
        f"{config.num_rungs} rungs, {config.num_holes} holes"
        f", k={config.symmetry_qs}, Sz={config.total_spin_projection}/2"
    )
    spin_projection_states = config.combinatorics_table[config.num_down_spins, config.num_nodes - config.num_holes]
    hole_states = config.combinatorics_table[config.num_holes, config.num_nodes]
    total_projection_states = spin_projection_states * hole_states
    momentum_states_approx = total_projection_states // config.num_nodes
    second_line = (
        f"number of states satisfying spin projection = {total_projection_states}"
        f", number of states further satifying momentum ~ {momentum_states_approx}"
    )
    momentum_states_per_Gb = momentum_states_approx / 1e9  # noqa: N806
    eigenvectors_size = 2 * 64 * momentum_states_per_Gb / 8
    elements_in_column = 3 * config.num_nodes
    index_element_size = np.empty(0, dtype=MATRIX_INDEX_TYPE).itemsize  # TODO: how to use it on the dtype directly?
    value_element_size = np.empty(0, dtype=VALUE_INDEX_TYPE).itemsize  # TODO: how to use it on the dtype directly?
    hamiltonian_size = (
        (elements_in_column + 1) * index_element_size + elements_in_column * value_element_size
    ) * momentum_states_per_Gb
    basis_size = momentum_states_per_Gb * (index_element_size + 4)
    third_line = (
        f"single eigenvector size: {eigenvectors_size:.2f} GB"
        f", full Hamiltonian size < {hamiltonian_size:.2f} GB, basis size: {basis_size:.2f} GB"
    )
    full_string = f"{first_line}\n{second_line}" + (
        f"\n{third_line}" if np.amax([hamiltonian_size, eigenvectors_size, basis_size]) > 1 else ""
    )
    return full_string


def _get_spectrum_figure_suffix(
    hamiltonian_config: Hamiltonian_Config, spectrum_config: gc.Spectrum_Part, operator_name_suffix: str
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


def _get_position_correlation_figure_suffix(
    hamiltonian_config: Hamiltonian_Config, correlation_config: gc.Position_Correlation_Part, operator_name_suffix: str
):
    weight_string = "_".join([f"{name}{weight}" for name, weight in asdict(hamiltonian_config.weights).items()])
    operator_name_suffix = f"_{operator_name_suffix}" if operator_name_suffix else ""
    fixed_string = f"{[shift.to_npint32().tolist() for shift in correlation_config.fixed_distances]}"
    suffix = f"{operator_name_suffix}_fixed{fixed_string}_{weight_string}"
    return suffix
