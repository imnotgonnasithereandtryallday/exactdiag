from itertools import product
import logging
import collections
import copy
import pathlib
import typing

import numpy as np
from pydantic import field_validator
from pydantic.dataclasses import dataclass, Field

from exactdiag.general.lanczos_diagonalization import (
    get_lowest_eigenpairs,
    get_spectrum,
    # get_offdiagonal_spectral_function_spectrum,
    Eigenvalues,
    Eigenvectors,
)
from exactdiag.tJ_spin_half_ladder import position_correlations
from exactdiag.tJ_spin_half_ladder import configs
from exactdiag.plotting import choose_plot


_logger = logging.getLogger(__name__)


class IntSymmetryQs(collections.namedtuple):
    """Symmetry quantum numbers as named tuple."""

    leg: int
    rung: typing.Literal[0, 1]


def get_eigenpairs(
    config: configs.Eigenpair_Config,
) -> tuple[Eigenvalues, Eigenvectors]:
    """Return eigevalues and eigenvectors.

    `eigenvectors[i, j]` is the `i`-th component of the `j`-th eigenvector which corresponds to `eigenvalues[j]`.
    """
    if config.hamiltonian.symmetry_qs is None:
        return get_all_k_eigenpairs(config)
    return get_lowest_eigenpairs(config)


def get_all_k_eigenpairs(config: configs.Eigenpair_Config) -> dict[IntSymmetryQs, tuple[Eigenvalues, Eigenvectors]]:
    """Return a map from all unique `symmetry_qs` to corresponding eigenvalues and eigenvectors.

    The values are as described in `get_eigenpairs`.
    """
    mut_config = copy.deepcopy(config)
    mkx = config.hamiltonian.num_rungs // 2 + 1
    mky = 2
    collected = {}
    for kx in range(mkx):
        for ky in range(mky):
            mut_config.hamiltonian.symmetry_qs = {"leg": kx, "rung": ky}
            eigvals, eigvecs = get_lowest_eigenpairs(mut_config)
            collected[(kx, ky)] = eigvals, eigvecs
    return collected


@dataclass(config={"arbitrary_types_allowed": True})
class Spectrum:
    # TODO: to general
    # TODO: return from more functions // make staticmethod?
    ws: np.ndarray
    spectrum: np.ndarray
    config: configs.Full_Spectrum_Config | configs.Full_Position_Correlation_Config
    info: dict = Field(default_factory=dict)

    @field_validator("ws", "spectrum", mode="before")
    @classmethod
    def _validate_ndarray(cls, value):
        return np.array(value)

    def get_figure_path(self, operator_name_suffix: str = "") -> pathlib.Path:
        data_path = self.config.get_spectrum_path(operator_name_suffix)
        rel_path = data_path.parents[1].relative_to(self.config.hamiltonian.get_calc_folder())
        figure_path = self.config.hamiltonian.get_figures_folder() / rel_path / f"{data_path.stem}.pdf"
        return figure_path


def get_excitation_spectrum(config: configs.Full_Spectrum_Config, limited_qs: bool = True) -> Spectrum:
    """Return the spectrum specified in the config."""
    config = copy.deepcopy(config)

    base_name = config.spectrum.name
    info = {}
    if base_name in {"current_rung", "current_leg"}:
        ws, spectrum = get_spectrum(config=config)
    elif base_name == "Szq":
        ws, spectrum, info = get_Szq_spectra(config=config, limited_qs=limited_qs)
    elif base_name == "spectral_function":
        ws, spectrum, info = get_spectral_function_spectra(config=config, limited_qs=limited_qs)
    elif base_name in {"offdiag_spec_func", "offdiagonal_spectral_function"}:
        # shortened as the spectrum name was too long
        ws, spectrum, info = get_offdiagonal_spectral_function_spectra(config=config, limited_qs=limited_qs)
    else:
        raise ValueError(f"{base_name} is not supported")
    return Spectrum(ws, spectrum, config, info)


def get_position_correlations(config: configs.Full_Position_Correlation_Config):
    base_name = config.correlatons.name
    if base_name in {"hole_correlations", "Sz_correlations"}:
        ws, spectrum = position_correlations.get_hole_spin_projection_correlations(config)

    elif base_name == "singlet-singlet":
        ws, spectrum = position_correlations.get_singlet_singlet_correlations(config)

    info = {"fixed_distances": config.correlatons.fixed_distances}
    return Spectrum(ws, spectrum, config, info)


def get_Szq_spectra(config: configs.Full_Spectrum_Config, limited_qs: bool):
    config = copy.deepcopy(config)
    mkx, qxs, qys = get_mkx_qxs_qys(config, limited_qs=False)  # We conditionally limit qs ourselves later.
    qs_list = product(qxs, qys)
    spectra = np.empty((len(qxs), len(qys), config.spectrum.omega_steps))
    ws = np.empty((
        len(qxs),
        len(qys),
        config.spectrum.omega_steps,
    ))  # TODO: also change to have only one dimension for qx,qy
    for i, qx in enumerate(qxs):
        for j, qy in enumerate(qys):
            if limited_qs and qx > mkx // 2:
                qx = mkx - qx
            config.spectrum.operator_symmetry_qs = {"leg": qx, "rung": qy}
            ws[i, j, :], spectra[i, j, :] = get_spectrum(config)
    return ws, spectra, {"qs_list": qs_list}


def get_spectral_function_spectra(config: configs.Full_Spectrum_Config, limited_qs: bool):
    config = copy.deepcopy(config)
    mkx, qxs, qys = get_mkx_qxs_qys(config=config, limited_qs=limited_qs)
    plus_minus_conditions = [
        ("plus", config.hamiltonian.num_holes > 0),
        ("minus", config.hamiltonian.num_holes < config.hamiltonian.num_nodes),
    ]
    base_name = config.spectrum.name
    qs_list = list(product(qxs, qys))  # Why do i get empty plots without the list?
    shape = [sum([i[1] for i in plus_minus_conditions]), len(qs_list), config.spectrum.omega_steps]
    spectra = np.empty(shape, dtype=float)
    ws = np.empty(shape, dtype=float)
    for i, (qx, qy) in enumerate(qs_list):
        if limited_qs and qx > mkx // 2 + 1:
            continue  # TODO: this is inconsistent with get_Szq_spectra
        qs = np.array([qx, qy], dtype=int)  # np.int32)
        for o, (suffix, condition) in enumerate(plus_minus_conditions):
            if not condition:
                continue
            if suffix == "minus":  # TODO: Fragile to change of suffix order.
                # qs in calculation refer to how the state's momentum changes,
                # qs in params file refer to the removed electron's momentum,
                # which is the standard for index of annihilation operator
                qs = -qs
            config.spectrum.operator_symmetry_qs = {"leg": qs[0], "rung": qs[1] % 2}
            config.spectrum.name = f"{base_name}_{suffix}"
            ws[o, i, :], spectra[o, i, :] = get_spectrum(config)
    return ws, spectra, {"qs_list": qs_list}


def get_offdiagonal_spectral_function_spectra(config: configs.Full_Spectrum_Config, limited_qs: bool):
    raise NotImplementedError()
    # config = copy.deepcopy(config)
    # mkx, qxs, qys = get_mkx_qxs_qys(config=config, limited_qs=limited_qs)
    # shape = [len(qxs)*len(qys), config.spectrum.omega_steps]
    # spectra = np.empty(shape, dtype=np.cdouble)
    # ws = np.empty(shape)
    # qs_list = [i for i in product(qxs, qys)]  # why do i get empty plots with just = product(qxs, qys)?
    # config.spectrum.name = 'offdiag_spec_func'
    # # N-2 is the number of spins in config file
    # for i,(qx,qy) in enumerate(qs_list):
    #     config.spectrum.operator_symmetry_qs.leg = qx
    #     config.spectrum.operator_symmetry_qs.leg = qy
    #     input_processor_Nless2, input_processor_N = get_offdiagonal_kwargs(input_processor)
    #     combined_spectrum_kwargs = input_processor_Nless2.spectrum_params
    #     ws[i,:], spectra[i,:] = get_offdiagonal_spectral_function_spectrum(input_processor_Nless2.eigenpair_params,\
    #                                         input_processor_Nless2.spectrum_params, \
    #                                         input_processor_N.eigenpair_params, input_processor_N.spectrum_params, \
    #                                         combined_spectrum_kwargs, **kwargs)
    # return ws, spectra, {'qs_list': qs_list}


def get_mkx_qxs_qys(config: configs.Limited_Spectrum_Config, limited_qs: bool):
    mkx = config.hamiltonian.num_rungs
    input_qs = config.spectrum.operator_symmetry_qs
    if input_qs is None:
        if limited_qs:
            qxs = range(mkx // 2 + 1)
        else:
            qxs = range(mkx)
        qys = range(2)
    else:
        input_qs = input_qs.to_npint32()
        qxs = [input_qs[0]]
        qys = [input_qs[1]]
    return mkx, qxs, qys


def plot_excitation_spectrum(config: configs.Full_Spectrum_Config, show: bool = False, **kwargs) -> None:
    spectrum = get_excitation_spectrum(config=config, **kwargs)
    choose_plot(name=config.spectrum.name, spectrum=spectrum, show=show)


def plot_position_correlation(config: configs.Full_Position_Correlation_Config, show: bool = False) -> None:
    spectrum = get_position_correlations(config=config)
    choose_plot(name=config.spectrum.name, spectrum=spectrum, show=show)
