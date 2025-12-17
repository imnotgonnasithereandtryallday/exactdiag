import logging
import pathlib
from os import makedirs
from functools import partial

from jaxtyping import Array, Complex, Float
import numpy as np
from scipy.sparse.linalg._eigen.arpack.arpack import _SymmetricArpackParams, _UnsymmetricArpackParams  # noqa: PLC2701 Note: renamed in scipy v1.8.0?

from exactdiag.utils import load_calculate_save
from exactdiag.general.cython_lanczos_diagonalization import lanczos_dynamical_response
import exactdiag.general.configs as gc


_logger = logging.getLogger(__name__)

type Eigenvalues = Float[Array, "num_eigenpairs"]  # noqa: F821 - this is not a forward type declaration
type Eigenvectors = Complex[Array, "basis_size, num_eigenpairs"]  # noqa: F821 - this is not a forward type declaration


def get_lowest_eigenpairs(config: gc.Eigenpair_Config) -> tuple[Eigenvalues, Eigenvectors]:
    """Return the eigenpairs by loading or calculating and saving."""
    _logger.debug(f"Getting lowest eigenpairs with {config}.")
    eigvals, eigvecs = load_calculate_save(
        *config.get_eigenpair_paths(),
        load_fun=load_eigenpairs,
        calc_fun=lambda: calculate_eigenpairs(config),
        save_fun=save_eigenpairs,
    )
    _logger.info(
        f"Got lowest eigenpairs with k={config.hamiltonian.symmetry_qs.to_npint32()!s}"
        f", {eigvecs.shape=}\n{eigvals=}"
    )
    return eigvals, eigvecs


def load_eigenpairs(vals_path: pathlib.Path, vecs_path: pathlib.Path) -> tuple[Eigenvalues, Eigenvectors]:
    eigvals = np.load(vals_path, allow_pickle=False)
    eigvecs = np.load(vecs_path, allow_pickle=False)
    _logger.debug(f"loaded eigenpairs from {vals_path!s}, {vecs_path!s}")
    return eigvals, eigvecs


def save_eigenpairs(eigenpairs, vals_path: pathlib.Path, vecs_path: pathlib.Path) -> None:
    makedirs(vals_path.parent, exist_ok=True)
    makedirs(vecs_path.parent, exist_ok=True)
    eigvals, eigvecs = eigenpairs
    np.save(vals_path, eigvals)
    np.save(vecs_path, eigvecs)
    _logger.debug(f"saved eigenpairs to {vals_path!s}, {vecs_path!s}")


def calculate_eigenpairs(config: gc.Eigenpair_Config):  # TODO: type retval
    # eigvecs[:, i] is the normalized eigenvector corresponding to the eigenvalue eigvals[i]
    hamiltonian = config.hamiltonian.setup()()
    num_rows = hamiltonian.shape[0]
    if num_rows == 0:
        _logger.warning("hamiltonian has 0 rows, skipping eigen problem")
        return [], []
    num_eigenpairs = config.eigenpair.num_eigenpairs
    num_added_vecs = config.eigenpair.num_recalculated_vecs_per_iteration
    fully_diagonalizable = num_rows < 2 * (num_eigenpairs + num_added_vecs) or num_rows < 250
    if fully_diagonalizable:
        _logger.info("full exact diagonalization")
        hamiltonian = hamiltonian.toarray()
        eigvals, eigvecs = np.linalg.eigh(hamiltonian)
        i = 0
    else:
        _logger.info("starting diagonalization")
        (eigvals, eigvecs), i = diagonalize_IRLM(
            num_eigenpairs, hamiltonian, num_added_vecs=num_added_vecs, is_symmetric=False
        )
    eigvals = np.real(eigvals)  # to suppress conversion warnings
    order = eigvals.argsort()[:num_eigenpairs]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    _logger.info(f"diagonalization done in {i} iterations")
    return eigvals, eigvecs


def diagonalize_IRLM(num_eigenpairs, operator, is_symmetric=False, num_added_vecs=5, initial_guess=None):
    """Diagonalize a hermitian operator using the implicitly restarted Lanczos method.

    Ordering is not always perfect, some eigenpairs may even be skipped (very rarely in my experience).
    """
    # TODO: typing
    num_columns = operator.shape[1]
    maxiter = 2**30
    tol = 2e-7
    if is_symmetric:
        params = _SymmetricArpackParams(
            num_columns,
            num_eigenpairs,
            "d",
            operator.dot,
            ncv=num_eigenpairs + num_added_vecs,
            v0=initial_guess,
            which="SA",
            maxiter=maxiter,
            tol=tol,
        )
    else:
        params = _UnsymmetricArpackParams(
            num_columns,
            num_eigenpairs,
            "D",
            operator.dot,
            ncv=num_eigenpairs + num_added_vecs,
            v0=initial_guess,
            which="SR",
            maxiter=maxiter,
            tol=tol,
        )
    i = 0
    notify_step = 5000
    while not params.converged:
        params.iterate()
        i += 1
        if i % notify_step == 0:
            _logger.info(f"iteration {i} still not converged, try restarting with increased num_added_vecs.")
    return params.extract(True), i


def get_spectrum(config: gc.Full_Spectrum_Config_Base, suffix: str = ""):
    # suffix - operator_name_suffix for figure file name.
    # TODO: typing.
    def calculate(**kwargs):
        eigvals, eigvecs = get_lowest_eigenpairs(config)
        get_operator, final_kwargs, get_excited_H = config.setup_excitation_operator()
        return calculate_spectrum(
            ws=config.spectrum.get_omegas(),
            broadening=config.spectrum.broadening,
            excited_state_hamiltonian=get_excited_H(),
            excitation_operator=get_operator(),
            eigvecs=eigvecs,
            eigvals=eigvals,
            num_lanczos_vecs=config.spectrum.num_lanczos_vecs,
            num_threads=config.spectrum.num_threads,
        )

    path = config.get_spectrum_path(suffix)
    ws, spectrum = load_calculate_save(path, load_fun=load_spectrum, calc_fun=calculate, save_fun=save_spectrum)
    return ws, spectrum


def load_spectrum(path: pathlib.Path | str):  # FIXME: type spectrum?
    npz_file = np.load(path, allow_pickle=True)
    ws, spectrum = npz_file.values()
    return ws, spectrum


def save_spectrum(ws_spectrum, path: pathlib.Path | str) -> None:  # FIXME: type spectrum?
    path = pathlib.Path(path)
    makedirs(path.parent, exist_ok=True)
    np.savez(path, *ws_spectrum)


def calculate_spectrum(
    ws, broadening, num_lanczos_vecs, excited_state_hamiltonian, excitation_operator, eigvecs, eigvals, num_threads
):
    gs_energy = eigvals[0]
    num_degenerate_states = 0
    spectrum = np.zeros(len(ws))
    zs = ws + gs_energy + 1j * broadening
    for j in range(len(eigvals)):  # TODO: should this be assumed from the inputs?
        # average over degenerate ground states
        if abs(gs_energy - eigvals[j]) > 1e-6:
            break
        num_degenerate_states += 1
        gs = eigvecs[:, j]
        spectrum += lanczos_dynamical_response(
            gs, excitation_operator, excited_state_hamiltonian, zs, num_lanczos_vecs, num_threads
        )
    spectrum /= num_degenerate_states
    return ws, spectrum


# def get_offdiagonal_spectral_function_spectrum(
#     config_Nless2, config_N
# ):
#     operator_N_kwargs = spectrum_N_kwargs["operator_params"]
#     operator_Nless2_kwargs = spectrum_Nless2_kwargs["operator_params"]

#     # N here is the number of spins
#     def calculate(**kwargs):
#         eigvals_N, eigvecs_N = get_lowest_eigenpairs(**eigenpair_N_kwargs)
#         get_operator_N, final_kwargs_N, get_H_N_1 = operator_N_kwargs["setup_func"](**operator_N_kwargs)
#         eigvals_Nless2, eigvecs_Nless2 = get_lowest_eigenpairs(**eigenpair_Nless2_kwargs)
#         get_operator_Nless2, final_kwargs_Nless2, repeated_get_H_N_1 = operator_Nless2_kwargs["setup_func"](
#             **operator_Nless2_kwargs
#         )

#         if any([
#             np.any(value != final_kwargs_Nless2[key])
#             for key, value in final_kwargs_N.items()
#             if not isinstance(value, partial) and key != "calc_folder"
#         ]):
#             # partial functions are hard to compare, their comparison is skipped
#             # calc_folder differs, but is not used
#             raise ValueError("the final states of the two operators are not the same")

#         # how should the states be combined when the gs are degenerate?
#         if np.abs(eigvals_N[0] - eigvals_N[1]) < 1e-6 or np.abs(eigvals_Nless2[0] - eigvals_Nless2[1]) < 1e-6:
#             raise ValueError("denerate ground states are not implemented")

#         operator_N = get_operator_N()
#         operator_Nless2 = get_operator_Nless2()
#         gs_N = eigvecs_N[:, 0]
#         gs_Nless2 = eigvecs_Nless2[:, 0]
#         tmp_state_Nless1 = operator_N.dot(gs_N) + operator_Nless2.dot(gs_Nless2)
#         real_state_Nless1 = np.zeros((len(tmp_state_Nless1), 1), dtype=np.cdouble)
#         real_state_Nless1[:, 0] = tmp_state_Nless1
#         imag_state_Nless1 = np.zeros((len(tmp_state_Nless1), 1), dtype=np.cdouble)
#         imag_state_Nless1[:, 0] = 1j * operator_N.dot(gs_N) + operator_Nless2.dot(gs_Nless2)

#         reference_energy = [0.5 * (eigvals_N[0] + eigvals_Nless2[0])]
#         excited_state_hamiltonian = get_H_N_1()
#         dummy_operator = Dummy_Operator()
#         real_kwargs = {**spectrum_combined_kwargs}
#         real_kwargs["file_name"] = "re_combined_" + spectrum_Nless2_kwargs["file_name"]
#         imag_kwargs = {**spectrum_combined_kwargs}
#         imag_kwargs["file_name"] = "im_combined_" + spectrum_Nless2_kwargs["file_name"]
#         _, real_combined_spectral_functions = get_spectrum(
#             excited_state_hamiltonian=excited_state_hamiltonian,
#             excitation_operator=dummy_operator,
#             eigvecs=real_state_Nless1,
#             eigvals=reference_energy,
#             **real_kwargs,
#         )
#         _, imag_combined_spectral_functions = get_spectrum(
#             excited_state_hamiltonian=excited_state_hamiltonian,
#             excitation_operator=dummy_operator,
#             eigvecs=imag_state_Nless1,
#             eigvals=reference_energy,
#             **imag_kwargs,
#         )

#         ws = spectrum_combined_kwargs["ws"]
#         diagonal_N_kwargs = {**spectrum_N_kwargs}
#         shift_N = reference_energy[0] - eigvals_N[0]
#         diagonal_N_kwargs["ws"] = ws + np.around(shift_N, decimals=1)

#         def get_ws_part(ws):
#             return f"_ws{ws[0]:g},{ws[-1]:g},{len(ws):g}_"

#         file_name = diagonal_N_kwargs["file_name"].split(get_ws_part(ws))  # TODO: this whole block is ugly
#         if len(file_name) != 2:
#             raise RuntimeError("dont be stupid with naming things, please.")
#         file_name = get_ws_part(diagonal_N_kwargs["ws"]).join(file_name)
#         diagonal_N_kwargs["file_name"] = file_name

#         ws_N, diagonal_spectral_function_minus_N = get_spectrum(
#             excited_state_hamiltonian=excited_state_hamiltonian,
#             excitation_operator=operator_N,
#             eigvecs=eigvecs_N,
#             eigvals=eigvals_N,
#             **diagonal_N_kwargs,
#         )
#         diagonal_spectral_function_minus_N_interp = np.interp(ws, ws_N - shift_N, diagonal_spectral_function_minus_N)

#         diagonal_Nless2_kwargs = {**spectrum_Nless2_kwargs}
#         shift_Nless2 = reference_energy[0] - eigvals_Nless2[0]
#         diagonal_Nless2_kwargs["ws"] = ws + np.around(shift_Nless2, decimals=1)

#         file_name = diagonal_Nless2_kwargs["file_name"].split(get_ws_part(ws))  # TODO: this whole block is ugly
#         if len(file_name) != 2:
#             raise RuntimeError("dont be stupid with naming things, please.")
#         file_name = get_ws_part(diagonal_Nless2_kwargs["ws"]).join(file_name)
#         diagonal_Nless2_kwargs["file_name"] = file_name

#         ws_Nless2, diagonal_spectral_function_plus_Nless2 = get_spectrum(
#             excited_state_hamiltonian=excited_state_hamiltonian,
#             excitation_operator=operator_Nless2,
#             eigvecs=eigvecs_Nless2,
#             eigvals=eigvals_Nless2,
#             **diagonal_Nless2_kwargs,
#         )
#         diagonal_spectral_function_plus_Nless2_interp = np.interp(
#             ws, ws_Nless2 - shift_Nless2, diagonal_spectral_function_plus_Nless2
#         )

#         off_diagonal_spectral_function = np.empty(len(ws), dtype=np.cdouble)
#         off_diagonal_spectral_function[:] = 0.5 * (
#             real_combined_spectral_functions
#             - diagonal_spectral_function_minus_N_interp
#             - diagonal_spectral_function_plus_Nless2_interp
#         )
#         off_diagonal_spectral_function -= 0.5j * (
#             imag_combined_spectral_functions
#             - diagonal_spectral_function_minus_N_interp
#             - diagonal_spectral_function_plus_Nless2_interp
#         )
#         return ws, off_diagonal_spectral_function

#     def load(*args, **kwargs):
#         # Numpy saves ws also as complex. Explicit cast to real to suppress warnings.
#         ws, spectrum = load_spectrum(*args, **kwargs)
#         ws = np.real(ws)
#         return ws, spectrum

#     return load_calculate_save(load_fun=load, calc_fun=calculate, save_fun=save_spectrum, **spectrum_combined_kwargs)


class Dummy_Operator:  # TODO: Move next to other matrices?
    def dot(self, vector):
        return vector
