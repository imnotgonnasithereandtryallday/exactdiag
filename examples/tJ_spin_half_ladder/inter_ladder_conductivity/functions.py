from itertools import product
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from exactdiag.tJ_spin_half_ladder import api
from exactdiag.tJ_spin_half_ladder import configs


def get_conductivity_contributions(config: configs.Config):
    """Return inter-ladder conductivity terms.

    Calculated from convolution of single-ladder spectral functions.
    """
    if np.any(config.hamiltonian.symmetry_qs.to_npint32() != 0):
        raise ValueError("The symmetry of q -> -q that comes with k=[0,0] is assumed.")
    if config.spectrum.omega_min != -config.spectrum.omega_max:
        raise ValueError("The currently implemented convolution requires symmetric bounds.")

    ws, apms = _get_spectral_functions(config)
    # interladder conductivity from spectral functions - assumes neighbouring ladders are uncorrelated
    # convolution of spectral functions for combinations c_q^\dagger c_q
    # suffers from boundary effects
    hopping_amplitudes = [0.04, 0.05, 0.04, -0.02]
    qxm, qym = apms.shape[:2]
    s2 = np.zeros(apms.shape[-1])  # t_{perp 2}
    s3 = np.zeros(apms.shape[-1])  # t_{perp 3}
    s5 = np.zeros(apms.shape[-1])  # t_{perp 5}
    s7 = np.zeros(apms.shape[-1])  # t_{perp 7}

    two_pi_over_a = 2 * np.pi / config.hamiltonian.num_rungs
    a_conv = np.zeros((qxm, 2, 2, len(ws)))
    sigma_2357 = np.zeros(apms.shape[-1])
    for qx in range(qxm):
        for i, j in product(range(2), range(2)):
            conv2 = np.interp(ws[::-1], -ws[::-1], np.convolve(apms[qx, j, 1, :], apms[qx, i, 0, :], mode="same"))
            a_conv[qx, i, j, :] = np.convolve(apms[qx, i, 0, :], apms[qx, j, 1, :], mode="same") - conv2
        degeneracy = 1 + (qx not in {0, qxm})
        f1 = np.abs(1 + np.exp(1j * qx * two_pi_over_a)) ** 2 / 2  # extra *2 for H.c. terms included
        f2 = np.abs(np.exp(-2j * qx * two_pi_over_a) + np.exp(1j * qx * two_pi_over_a)) ** 2 / 2
        conv_sum = np.sum(a_conv[qx, :, :], axis=(0, 1))  # a_conv00 + a_conv01 + a_conv10 + a_conv11
        s2 += conv_sum * f1 * degeneracy
        s3 += conv_sum * f1 * degeneracy
        s5 += conv_sum * f2 * degeneracy
        s7 += conv_sum * f2 * degeneracy
        u1 = (1 + np.exp(1j * qx * two_pi_over_a)) / 2
        u2 = (np.exp(-2j * qx * two_pi_over_a) + np.exp(1j * qx * two_pi_over_a)) / 2
        us = np.array([u1, u1, u2, u2])
        signs = [[1, 1, 1, 1], [1, -1, 1, -1], [1, -1, -1, -1], [1, 1, -1, 1]]
        indeces = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for sign, inds in zip(signs, indeces):
            sign = np.array(sign)
            sigma_2357 += (
                np.abs(np.sum(hopping_amplitudes * sign * us)) ** 2 * a_conv[qx, inds[0], inds[1], :] * degeneracy * 2
            )
    # print([(np.amin(s/ws),np.argmin(s/ws),ws[np.argmin(s/ws)]) for s in [s2,s3,s5,s7]])
    sigma_2357 /= ws + 1e-8
    return ws, sigma_2357, hopping_amplitudes


def _get_spectral_functions(config: configs.Config):
    if config.hamiltonian.num_holes == 0:
        # We assume the presence of both creation and annihilation spectral functions.
        raise ValueError("The number of holes must be nonzero.")
    if config.spectrum.name != "spectral_function":
        raise ValueError("Spectrum name must be 'spectral_function'.")

    ws, spectra, info = api.get_spectral_function_spectra(config=config, limited_qs=True)
    single_ws = ws[*[0] * (ws.ndim - 1), :]
    if np.any(ws != single_ws):
        raise RuntimeError("Expected all spectral functions to be calculated for the same energies.")
    # Reorder the spectra to match existing format.
    xmax = config.hamiltonian.num_rungs // 2 + 1
    wsteps = config.spectrum.omega_steps
    apms = np.zeros((xmax, 2, 2, wsteps))
    for i, (qx, qy) in enumerate(info["qs_list"]):
        for pm in range(2):
            apms[qx, qy, pm, :] = spectra[pm, i]
    return single_ws, apms


def plot_interladder_conductivity(config, ws, sigma_2357, hopping_amplitudes, folder=None):
    """Plot the contributions to the conductivity separately."""
    xlim = [0.0, 1.5]  # [ws[0],ws[-1]]
    ylim = [0, None]

    # I think the conductivity must depend on the "distance traveled" during the hopping.
    # a = ap = 3.93e-10
    # b = 1.8e-10
    # ell = 6.69e-10
    # q = 1.6e-19
    # hbar = 1.05e-34
    # distance_traveled = [b,ap+b,b,ap+b]
    # t2s = [t**2*y**2 for t,y in zip(ts,distance_traveled)]
    # constant_factor = np.pi*q**2/hbar/(nn-hn)/a/ell/(ap+b)*2*1e-2
    # 2 from spin summation, omega in eV, 1e-2 for cm-1, t in eV, delta fucntion in 1/eV
    plt.plot(ws, sigma_2357)
    plt.ylabel(r"$\sigma(\omega)$ [a.u.]")
    plt.xlabel("frequency [eV]")
    plt.xlim(xlim)
    plt.ylim(ylim)

    hpar = [getattr(config.hamiltonian.weights, name) for name in ["tl", "tr", "jl", "jr"]]
    plt.title(rf"$t_\parallel={hpar[0]},\,t_\perp={hpar[1]},\,J_\parallel={hpar[2]},\,J_\perp={hpar[3]}$")
    tss = ", ".join([f"{t:g}" for t in hopping_amplitudes])
    if folder is not None:
        folder = pathlib.Path(folder)
        fig_path = folder / f"interladder_sigma_sum{tss}_hpar{hpar}.pdf"
        plt.savefig(fig_path, bbox_inches="tight")
    plt.show()


def multiple_plots(config, varied_param_list, varied_param_index, varied_param_label, lines_per_plot, folder=None):
    """Plot multiple spectra that differ in one varied hamiltonian weight."""
    hpar = [getattr(config.hamiltonian.weights, name) for name in ["tl", "tr", "jl", "jr"]]
    titles = [rf"t_\parallel={hpar[0]}", rf"t_\perp={hpar[1]}", rf"J_\parallel={hpar[2]}", rf"J_\perp={hpar[3]}"]
    title = (
        f"${varied_param_label}$ dependence, $"
        + r",\,".join([s for i, s in enumerate(titles) if i != varied_param_index])
        + "$"
    )

    xlim = [0.0, 1.5]
    found_varied_param_list = []
    ymax = 0
    for i, param in enumerate(varied_param_list):
        hpar[varied_param_index] = param
        try:
            ws, sigma_2357, ts = get_conductivity_contributions(config)
            found_varied_param_list.append(param)
            ymax = max(ymax, np.amax(sigma_2357) * 1.03)
        except FileNotFoundError:
            print("missing file for: ", hpar)
    ylim = [0, ymax]
    part = 1
    for i, param in enumerate(found_varied_param_list):
        hpar[varied_param_index] = param
        ws, sigma_2357, ts = get_conductivity_contributions(config)
        plt.plot(ws, sigma_2357, label=rf"${varied_param_label}={param}$")
        if (i + 1) % lines_per_plot == 0 or i == len(varied_param_list) - 1:
            plt.ylabel(r"$\sigma(\omega)$ [a.u.]")
            plt.xlabel("frequency [eV]")
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.title(title)
            plt.legend()
            if folder is not None:
                folder = pathlib.Path(folder)
                fig_path = folder / f"interladder_sigma_varied{varied_param_index}_dep_part{part}_hpar{hpar}.pdf"
                plt.savefig(fig_path, bbox_inches="tight")
            part += 1
            plt.show()
