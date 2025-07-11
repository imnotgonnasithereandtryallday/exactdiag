import abc
from collections.abc import Callable
from dataclasses import asdict
import enum
import json
import os
import pathlib
import typing

import numpy as np
from pydantic import model_validator
from pydantic.dataclasses import dataclass

from exactdiag.general import basis_indexing
from exactdiag.general import symmetry
from exactdiag.general import sparse_matrices
from exactdiag.general.group import symmetry_generator

# TODO: Setting config on pydantic dataclasses and then inheriting from them
#       has some very stupid interactions.
#       We would like the config to be {"validate_assignment": True, "extra": "forbid"}
#       on all of these. We give it only to those classes that do not make sense to subclass.


@dataclass(kw_only=True)
class Weights_Base:
    """Weights of the Hamiltonian terms.

    Children should also be dataclasses.
    """


@dataclass(kw_only=True)
class Quantum_Numbers_Base:
    """Quantum numbers conserved in the dense basis but not the sparse one.

    Determines the symmetry group representation of a Hamiltonian block.
    """

    def to_npint32(self):
        """Return quantum numbers as a numpy array of int32."""
        return np.array(list(asdict(self).values()), dtype=np.int32)


@dataclass(kw_only=True)
class Position_Shift_Base:
    """Describes a relative node position."""

    def to_npint32(self):
        """Return shift as a numpy array of int32."""
        return np.array(list(asdict(self).values()), dtype=np.int32)


Name = typing.TypeVar("Name", bound=enum.StrEnum)
Quantum_Numbers_co = typing.TypeVar("Quantum_Numbers_co", bound=Quantum_Numbers_Base, covariant=True)
Position_Shift_co = typing.TypeVar("Position_Shift_co", bound=Position_Shift_Base, covariant=True)


class _Json_Loadable:
    @classmethod
    def load(cls, path: pathlib.Path | str) -> typing.Self:
        """Load from a JSON file."""
        with open(path, mode="r", encoding="utf-8") as fp:
            d = json.load(fp)
        return cls(**d)


@dataclass(kw_only=True)
class Hamiltonian_Config_Base[Quantum_Numbers_co](_Json_Loadable):
    r"""Information necessary to construct a Hamiltonian."""

    CALCULATIONS_FOLDER_NAME: typing.ClassVar[str] = (
        "calculations"  # TODO: Change to instance var to allow for simple cross-validation of basis_maps
    )
    FIGURES_FOLDER_NAME: typing.ClassVar[str] = "figures"

    weights: Weights_Base
    symmetry_qs: Quantum_Numbers_co
    num_threads: int
    major: typing.Literal["row", "column"] = "row"  # TODO: to bit flag? enum?
    triangle_only: bool = True

    def get_calc_folder(self) -> pathlib.Path:
        """Return the folder the eigenpairs, matrices and spectra should be saved to."""
        path = self.CALCULATIONS_FOLDER_NAME / self._get_relative_path()
        os.makedirs(path, exist_ok=True)
        return path

    def get_figures_folder(self) -> pathlib.Path:
        """Return the folder the figures should be saved to."""
        path = self.FIGURES_FOLDER_NAME / self._get_relative_path()
        os.makedirs(path, exist_ok=True)
        return path

    @abc.abstractmethod
    def _get_relative_path(self) -> pathlib.Path:
        """Return path to the folder of the block relative to the root calculations / figures folder."""

    @abc.abstractmethod
    def get_translators(
        self,
    ) -> "tuple[symmetry.Py_State_Index_Amplitude_Translator, basis_indexing.Py_Basis_Index_Map, symmetry_generator.Py_Symmetry_Generator]":
        """Return wrapped State_Index_Amplitude_Translator, Basis_Index_Map, and Symmetry_Generator instances."""

    def setup(self) -> Callable[[], "sparse_matrices.Added_Sparse_Matrices"]:
        """Return a callable that returns the Hamiltonian.

        The callable, when called, loads matrices if they already exist or calculates and saves them.
        """

    def get_symmetry_info(self) -> dict[str, typing.Any]:
        """Return a JSON-serializable dictionary that determines the symmetry block."""


@dataclass(kw_only=True, config={"validate_assignment": True, "extra": "forbid"})
class _Eigenpair_Part:
    """Information necessary to diagonalize a Hamiltonian.

    num_eigenpairs: the number of desired eigenpairs.
    num_recalculated_vecs_per_iteration: The implementation uses the Implicitly Restarted Arnoldi Method.
        Increasing this number should speed the diagonalization up, but increases memory requirements.
    num_threads: Setting to None indicates to copy the attribute from hamiltonian config when put into Config.
    """

    num_eigenpairs: int
    num_recalculated_vecs_per_iteration: int = 4
    num_threads: int | None = None


Hamiltonian_co = typing.TypeVar("Hamiltonian_co", bound=Hamiltonian_Config_Base, covariant=True)


@dataclass(kw_only=True)
class Eigenpair_Config[Hamiltonian_co](_Json_Loadable):
    """Information necessary to calculate a Hamiltonian and partially diagonalize it."""

    hamiltonian: Hamiltonian_co
    eigenpair: _Eigenpair_Part

    @model_validator(mode="after")
    def __fill_in_num_threads(self):
        if self.eigenpair.num_threads is None:
            self.eigenpair.num_threads = self.hamiltonian.num_threads
        return self

    def get_eigenpair_paths(self) -> tuple[pathlib.Path, pathlib.Path]:
        """Return paths to the files with the eigenvalues and eigenvectors."""
        hamiltonian_string = "_".join([
            f"{name}{weight:g}" for name, weight in asdict(self.hamiltonian.weights).items()
        ])
        suffix = f"_{hamiltonian_string}_eigenpairs{self.eigenpair.num_eigenpairs}.npy"
        eigen_folder = self.hamiltonian.get_calc_folder() / "eigenpairs"
        eigval_path = eigen_folder / f"eigenvals{suffix}"
        eigvec_path = eigen_folder / f"eigenvecs{suffix}"
        return eigval_path, eigvec_path


@dataclass(kw_only=True, config={"validate_assignment": True, "extra": "forbid"})
class Operator_Part_Base[Name]:
    """Information necessary, in addition to a Hamiltonian_Config instance, to calculate an operator.

    num_threads: Setting to None indicates to copy the attribute from hamiltonian config when put into Config.
    """

    name: Name
    num_threads: int | None = None


@dataclass(kw_only=True, config={"validate_assignment": True, "extra": "forbid"})
class Spectrum_Part[Name, Quantum_Numbers_co](Operator_Part_Base[Name]):
    r"""Information necessary, in addition to a Hamiltonian_Config instance, to calculate an operator.

    operator_symmetry_qs: change in the quantum numbers between the initial and final hamiltonian blocks.
    num_threads: Setting to None indicates to copy the attribute from hamiltonian config when put into Config.
    """

    operator_symmetry_qs: Quantum_Numbers_co
    broadening: float
    omega_max: float
    omega_min: float = 0
    omega_steps: int = 1000
    num_lanczos_vecs: int = 250

    def get_omegas(self):
        """Return a numpy array of uniformly spaced energies."""
        return np.linspace(self.omega_min, self.omega_max, self.omega_steps)


SP_co = typing.TypeVar("SP_co", bound=Spectrum_Part, covariant=True)


@dataclass(kw_only=True)
class Limited_Spectrum_Config_Base[Hamiltonian_co, SP_co](_Json_Loadable):
    """Information necessary to calculate an operator."""

    hamiltonian: Hamiltonian_co
    spectrum: SP_co

    @model_validator(mode="after")
    def __fill_in_num_threads(self):
        if self.spectrum.num_threads is None:
            self.spectrum.num_threads = self.hamiltonian.num_threads
        return self

    @abc.abstractmethod
    def setup_excitation_operator(
        self,
    ) -> tuple[
        Callable[[], "sparse_matrices.Sparse_Matrix"],
        Hamiltonian_Config_Base,
        Callable[[], "sparse_matrices.Added_Sparse_Matrices"],
    ]:
        """Return callables to set up the excitation operator.

        Returns a callable to get the operator, info about its final symmetry block,
        and a callable to construct the Hamiltonian in that block.
        """


@dataclass(kw_only=True)
class Full_Spectrum_Config_Base[Hamiltonian_co, SP_co](
    Limited_Spectrum_Config_Base[Hamiltonian_co, SP_co], Eigenpair_Config[Hamiltonian_co]
):
    """Information necessary to calculate an operator and its spectrum."""

    @abc.abstractmethod
    def get_spectrum_path(self, operator_name_suffix: str = "") -> pathlib.Path:
        """Return path the spectrum should be saved to."""


@dataclass(kw_only=True, config={"validate_assignment": True, "extra": "forbid"})
class Position_Correlation_Part[Name, Position_Shift_co](Operator_Part_Base[Name]):
    r"""Information necessary, in addition to a Hamiltonian_Config instance, to get a position correlation operator.

    fixed_distances: fixed relative shifts between operator terms.
    num_threads: Setting to None indicates to copy the attribute from hamiltonian config when put into Config.
    """

    fixed_distances: list[Position_Shift_co]


@dataclass(kw_only=True, config={"validate_assignment": True, "extra": "forbid"})
class Limited_Position_Correlation_Config_Base[Hamiltonian_co, Name, Position_Shift_co](_Json_Loadable):
    """Information necessary calculate a position correlation operator."""

    hamiltonian: Hamiltonian_co
    correlations: Position_Correlation_Part[Name, Position_Shift_co]

    @abc.abstractmethod
    def setup_excitation_operator(
        self, free_shift: Position_Shift_co
    ) -> tuple[
        Callable[[], "sparse_matrices.Sparse_Matrix"],
        Hamiltonian_co,
        Callable[[], "sparse_matrices.Added_Sparse_Matrices"],
    ]:
        """Return callables to set up the excitation operator.

        Returns a callable to get the operator, info about its final symmetry block,
        and a callable to construct the Hamiltonian in that block.

        Args:
        free_shift: relative position of the operators not specified by the fixed shifts of the Hamiltonian.

        """


@dataclass(kw_only=True, config={"validate_assignment": True, "extra": "forbid"})
class Full_Position_Correlation_Config_Base[Hamiltonian_co, Name, Position_Shift_co](
    Limited_Position_Correlation_Config_Base[Hamiltonian_co, Name, Position_Shift_co], Eigenpair_Config[Hamiltonian_co]
):
    r"""Information necessary calculate and evaluate position correlation operator."""

    @abc.abstractmethod
    def get_spectrum_path(self, operator_name_suffix: str = "") -> pathlib.Path:
        """Return path the evaluated correlations should be saved to."""
