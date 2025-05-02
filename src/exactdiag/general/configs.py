import abc
from collections.abc import Callable
from dataclasses import asdict
import pathlib
import typing
import json
import os

import numpy as np
from pydantic.dataclasses import dataclass

from exactdiag.general import basis_indexing
from exactdiag.general import symmetry
from exactdiag.general import sparse_matrices
from exactdiag.general.group import symmetry_generator

# TODO: Setting config on pydantic dataclasses and then inheriting from them
#       has some very stupid interactions.
#       We would like the config to be {"validate_assignment": True, "extra": "forbid"}
#       on all of these. We give it only to Eigenpairs since there is no reason to subclass it.


@dataclass(kw_only=True)
class Weights_Base:
    """Weights of the Hamiltonian terms.

    Children should also be dataclasses.
    """


@dataclass
class Quantum_Numbers_Base:
    """Quantum numbers conserved in the dense basis but not the sparse one.

    Determines the symmetry group representation of a Hamiltonian block.
    """

    # TODO: Should we allow creation from terable or not?
    def to_npint32(self):
        """Return quantum numbers as a numpy array of int32."""
        return np.array(list(asdict(self).values()), dtype=np.int32)


@dataclass(kw_only=True)
class Hamiltonian_Config_Base:
    r"""Information necessary to construct a Hamiltonian."""

    CALCULATIONS_FOLDER_NAME: typing.ClassVar[str] = "calculations"  # TODO: Change to instance var to allow for simple cross-validation of basis_maps
    FIGURES_FOLDER_NAME: typing.ClassVar[str] = "figures"

    weights: Weights_Base
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


@dataclass(kw_only=True, config={'validate_assignment': True, "extra": "forbid"})
class Eigenpair_Config:
    """Information necessary to diagonalize a Hamiltonian.

    num_eigenpairs: the number of desired eigenpairs.
    num_recalculated_vecs_per_iteration: The implementation uses the Implicitly Restarted Arnoldi Method.
        Increasing this number should speed the diagonalization up, but increases memory requirements.
    num_threads: Setting to None indicates to copy the attribute from hamiltonian config when put into Config.
    """

    # TODO: There must be a better way to copy the num_threads.
    num_eigenpairs: int
    num_recalculated_vecs_per_iteration: int = 4
    num_threads: int | None = None


@dataclass(kw_only=True)
class Spectrum_Config_Base:
    """Information necessary, in addition to a Hamiltonian_Config instance, to calculate an operator spectrum.

    num_threads: Setting to None indicates to copy the attribute from hamiltonian config when put into Config.
    """

    # TODO: There must be a better way to copy the num_threads.
    num_threads: int | None = None


@dataclass(kw_only=True)
class Combined_Config_Base:
    """Combined specialized configs.

    Provides methods that require attributes from multiple specializations.
    If either of eigenpair or spectrum has num_threads equal to None, it is set to hamiltonian.num_threads.
    """

    hamiltonian: Hamiltonian_Config_Base
    eigenpair: Eigenpair_Config
    spectrum: Spectrum_Config_Base | None = None
    # TODO: maybe eigenpair should contain hamiltonian and spectrum should contain both?

    def __post_init__(self):
        if self.eigenpair.num_threads is None:
            self.eigenpair.num_threads = self.hamiltonian.num_threads
        if self.spectrum is not None and self.spectrum.num_threads is None:
            self.spectrum.num_threads = self.hamiltonian.num_threads

    @classmethod
    def load(cls, path: pathlib.Path) -> typing.Self:
        """Load from a JSON file."""
        with open(path, mode="r", encoding="utf-8") as fp:
            d = json.load(fp)
        return cls(**d)

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

    @abc.abstractmethod
    def get_spectrum_path(self, operator_name_suffix: str = "") -> pathlib.Path:
        """Return path the spectrum should be saved to.

        Raises ValueError if self.spectrum is None.
        """

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

        Raises ValueError if self.spectrum is None.
        """
