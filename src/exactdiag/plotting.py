import logging

import matplotlib
from matplotlib import cm
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np


_logger = logging.getLogger(__name__)
# from .tJ_spin_half_ladder.naming import get_figure_name
# from .hubbard.naming_conventions import get_figure_name as get_hubbard_figure_name # FIXME: Find a consistent solution


# This module is retarded. A spectrum should know how to plot itself.


def choose_plot(name, **kwargs):
    if name == "Szq":
        plot_Szq(**kwargs)
    elif name in {"current_rung", "current_leg"}:
        rung_leg = name.split("_")[1]
        plot_current_rung_leg(rung_leg=rung_leg, **kwargs)
    elif name == "spectral_function":
        plot_all_spectral_functions(**kwargs)
    elif name == "offdiagonal_spectral_function":
        plot_all_offdiagonal_spectral_functions(**kwargs)
    elif name == "hole_correlations":
        plot_hole_correlations(**kwargs)
    elif name == "Sz_correlations":
        plot_Sz_position_correlations(**kwargs)
    elif name == "singlet-singlet":
        plot_singlet_correlations(**kwargs)

    elif name in {"Sq_plus", "Sq_prime_plus"}:
        # TODO: How to separate hubbard and ladder?
        if "eps_range" in kwargs.keys():
            plot_Sq_plus_range(**kwargs)
        else:
            plot_Sq_plus(**kwargs)

    else:
        print(f"Plot {name} not supported.")


def plot_Szq(spectrum, show: bool = False):  # todo type spectrum
    # ws: n+1 dimensional numpy array. Only the last axis is currently used.
    #     The last axis must be the same for the whole matrix.
    # spectrum: 3D numpy array. spectrum[qx,qy,w] gives the susceptibility for (qx,qy) at energy ws[...,w]
    if np.any(spectrum.config.hamiltonian.symmetry_qs.to_npint32() != (0, 0)):
        _logger.warning("Plots for hamiltonian symmetry_qs != (0,0) are probably incorrect.")
    last_axis = tuple([0] * (len(spectrum.ws.shape) - 1) + [slice(None)])
    if np.any(spectrum.ws[last_axis] != spectrum.ws):
        # -------- could be solved with plt.contourf instead of imshow?
        raise ValueError(spectrum.ws)
    ws = spectrum.ws[last_axis]
    ext_spectrum = np.empty(np.array(spectrum.spectrum.shape) + [1, 0, 0])
    ext_spectrum[:-1, :, :] = spectrum.spectrum
    ext_spectrum[-1, :, :] = spectrum.spectrum[0, :, :]
    mkx = ext_spectrum.shape[0]  # TODO: these extents assume too much about input qs
    k_size = 0.5 / mkx
    extent = -k_size, 1 + k_size, np.nanmin(ws), np.nanmax(ws)
    aspect = 1.3
    for y in range(spectrum.spectrum.shape[1]):
        plt.subplots()
        ys = ext_spectrum[:, y, :].T
        plt.imshow(ys, aspect=aspect, origin="lower", cmap=cm.gnuplot, extent=extent)
        cbar = plt.colorbar()
        plt.xlabel(r"$k_x \, [2\pi / a]$")  # TODO: adjust depending on plot_info
        plt.ylabel(r"$\omega$ [eV]")
        cbar.set_label(r"$\chi$ [a. u.]")
        plt.savefig(spectrum.get_figure_path(operator_name_suffix=f"qy{y}_map"), bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

        dos = np.sum(ys[:, :-1], axis=1)
        plt.plot(ws, dos)
        plt.xlim([0, ws[-1]])
        plt.ylim([0, np.nanmax(dos)])
        plt.xlabel(r"$\omega$ [eV]")
        plt.ylabel(r"DOS")
        plt.savefig(spectrum.get_figure_path(operator_name_suffix=f"qy{y}_dos"), bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()


def plot_current_rung_leg(spectrum, show=False, **kwargs):
    # we calculate the real part for omega > 0 only,
    # the omega < 0 part is nonzero due to broadening and is used to
    # erase the unphysical peak at omega = 0
    ws = spectrum.ws
    ys = spectrum.spectrum
    antisym_spectrum = ys - np.interp(-ws, ws, ys)
    divided_spectrum = antisym_spectrum / (ws + 1e-8 * np.amax(np.abs(ws)))
    # TODO: the peak should be present if the initial state is degenerate and we for some reason
    #       shouldn't average over the degenerate states!

    plt.subplots()
    plt.plot(ws, divided_spectrum)
    plt.xlim([0, ws[-1]])
    plt.ylim([0, np.nanmax(divided_spectrum)])
    plt.xlabel(r"$\omega$ [eV]")
    ylabel_suffix = "r" if "rung" in spectrum.config.spectrum.name else r"\ell"
    plt.ylabel(rf"$\sigma_{ylabel_suffix}$ [a. u.]")
    qs = (
        spectrum.config.spectrum.operator_symmetry_qs.to_npint32()
        if spectrum.config.spectrum.operator_symmetry_qs is not None
        else (0, 0)  # FIXME: this should be handled elsewhere.
    )
    plt.title(rf"$q_x, q_y = {qs[0]:g}, {qs[1]:g}$")
    plt.savefig(spectrum.get_figure_path(), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_offdiagonal_spectral_functions(
    ws, spectrum, spectrum_info, hamiltonian_kwargs, plot_info, show=False, **kwargs
):
    # ws: Energies of the calculated spectra. ws[i,:] gives the energies of spectrum[i,:] for any index i in plot_info['qs_list'].
    # spectrum: Response spectra. spectrum[i,:] gives the response for any index i in plot_info['qs_list'].
    plt.subplots(figsize=(8, 3))
    xlim = [np.amin(ws), np.amax(ws)]
    for re_im_label, f in [("Im", np.real), ("Re", np.imag)]:
        # imaginary part of \tilde{F} as defined in poilblanc 2003 prl
        # is proportional to real part of offdiagonal spectral function
        # except there can be an arbitrary phase shift, see dissertation!
        for i, (qx, qy) in enumerate(plot_info["qs_list"]):
            label = f" $q_x=\,{qx}$" if qy == 0 or (qx, 0) not in plot_info["qs_list"] else None
            plt.plot(ws[i, :], f(spectrum[i, :]), label=label, c=f"C{qx}", ls=":" if qy else "-")
        plt.xlim(xlim)
        plt.ylim([None, None])

        # add legend entries
        ex = (None,)
        ey = (None,)
        plt.plot(ex, ey, c="w", label=" ")
        plt.plot(ex, ey, c="k", label=r"$k_y=0$", ls="-")
        plt.plot(ex, ey, c="k", label=r"$k_y=\pi$", ls=":")
        plt.legend(loc=(1.05, -0.07))

        # Set ticks on both sides of axes on
        ax = plt.gca()
        ax.tick_params(axis="x", which="both", bottom=True, top=True, labelbottom=True, labeltop=True)
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))

        plt.xlabel(r"$\omega$ [eV]")
        plt.ylabel(f"{re_im_label}" + r"$\tilde{F}_k$")
        plt.savefig(spectrum.get_figure_path(operator_name_suffix="offdiag_spec_func_all"), bbox_inches="tight")
        if show:
            plt.tight_layout()
            plt.show()  # TODO: how to make the window not block further computation AND survive after the program finishes?
        else:
            plt.close()


def plot_all_spectral_functions(spectrum, show=False, **kwargs):
    # ws: Energies of the calculated spectra. ws[i,pm:] gives the energies of spectrum[i,pm,:] for any index i in plot_info['qs_list'].
    #     pm in [0,1] stands for creation or annihilation part of the spectral function.
    # spectrum: Response spectra. spectrum[i,pm,:] gives the response for any index i in plot_info['qs_list'].
    plt.subplots(figsize=(8, 3))
    for i, (qx, qy) in enumerate(spectrum.info["qs_list"]):
        for plus_minus in range(2):
            label = (
                f" $q_x=\,{qx}$" if plus_minus == 1 and (qy == 0 or (qx, 0) not in spectrum.info["qs_list"]) else None
            )
            plt.plot(
                (-1) ** plus_minus * spectrum.ws[plus_minus, i, :],
                spectrum.spectrum[plus_minus, i, :],
                label=label,
                c=f"C{qx}",
                ls=":" if qy else "-",
            )
    ws_max = np.amax(spectrum.ws)
    ws_min = np.amin(spectrum.ws)
    xlim = [min(ws_min, -ws_max), max(-ws_min, ws_max)]
    plt.xlim(xlim)
    plt.ylim([0, None])

    # add legend entries
    ex = (None,)
    ey = (None,)
    plt.plot(ex, ey, c="w", label=" ")
    plt.plot(ex, ey, c="k", label=r"$k_y=0$", ls="-")
    plt.plot(ex, ey, c="k", label=r"$k_y=\pi$", ls=":")
    plt.legend(loc=(1.05, -0.07))

    # Set ticks on both sides of axes on
    ax = plt.gca()
    ax.tick_params(axis="x", which="both", bottom=True, top=True, labelbottom=True, labeltop=True)
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))

    plt.xlabel(r"$\omega$ [eV]")
    plt.ylabel(r"$A_k$")
    plt.savefig(spectrum.get_figure_path(), bbox_inches="tight")
    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close()


def plot_hole_correlations(spectrum, show=False, periodc=(True, False)):
    correlations = spectrum.spectrum / np.amax(spectrum.spectrum)
    x = spectrum.ws[:, 0]
    y = spectrum.ws[:, 1]
    x_len = np.amax(x) + 1
    y_len = np.amax(y) + 1
    x_shift = x_len // 2
    y_shift = (y_len - 1) // 2
    fixed = [[0,0]] + [shift.to_npint32() for shift in spectrum.info["fixed_distances"]]
    fixed = [[(x + x_shift) % x_len, (y + y_shift) % y_len] for (x, y) in fixed]
    x = (x + x_shift) % x_len
    y = (y + y_shift) % y_len
    label_shift = (-1 + 2 * y) * 0.32
    fac = 0.7
    fac2 = fac * np.pi * 5e4
    area = fac2 * correlations**2
    fig, axes = plt.subplots(figsize=(1.5 * x_len, 1.5 * y_len))
    for i, (xv, yv) in enumerate(zip(x, y)):
        if [xv, yv] in fixed:
            area[i] = 0
    for v in set(y):
        plt.plot([0, x_len - 1 + periodc[0]], [v, v], "-", color="b")
    for v in set(x):
        plt.plot([v, v], [0, y_len - 1 + periodc[1]], "-", color="b")
    plt.scatter(x, y, s=area, color="gray")
    for xv, yv, vv, sv in zip(x, y, correlations, label_shift):
        if [xv, yv] not in fixed:
            plt.text(xv - 0.2, yv + sv, "%.3f" % vv)
    plt.scatter(*zip(*fixed), s=1e-2 * fac2, edgecolors="r", facecolors="none")
    plt.ylim(-0.5, 1.5)
    plt.axis("off")
    fig.set_dpi(100)

    plt.savefig(spectrum.get_figure_path(), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_Sz_position_correlations(spectrum, spectrum_info, hamiltonian_kwargs, plot_info, show=False, **kwargs):
    x = [0.5 * i for i in range(len(spectrum))]
    plt.plot(x, spectrum, marker="s")

    plt.xlabel(r"rung distance $d$")
    plt.ylabel(r"$\sum_i\left<S^z_i S^z_{i+d}\right>$")
    plt.xlim(x[0], x[-1])
    plt.savefig(spectrum.get_figure_path(), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_singlet_correlations(spectrum, show=False):
    fixed_distances = spectrum.info["fixed_distances"]
    plt.subplots()
    plt.title(rf"intra-singlet separations: $\alpha=${fixed_distances[0]}, $\beta=${fixed_distances[1]}")
    y = spectrum.ws[:, 1]
    markers = "sopD^"
    for i, v in enumerate(set(y)):
        label = f"leg distance = {v}"
        y_mask = y == v
        x = spectrum.ws[y_mask, 0]
        positive_mask = np.logical_and(spectrum.spectrum[:] >= 0, y_mask)
        positive = spectrum.spectrum[positive_mask]
        pos_x = spectrum.ws[positive_mask, 0]
        c = f"C{i % 9}"
        plt.plot(x, np.abs(spectrum.spectrum[y_mask]), label=label, marker=markers[i], fillstyle="none", c=c)
        plt.plot(pos_x, positive, marker=markers[i], lw=0, c=c)

    plt.legend()
    plt.xlabel(r"inter-singlet rung distance $d$")
    plt.ylabel(r"$\left<\Delta_{0\alpha}^\dagger \Delta_{d\beta}\right>$")

    plt.yscale("log")
    plt.ylim([1e-5, 1])
    plt.xlim([0, np.amax(spectrum.ws[:, 0])])

    plt.savefig(spectrum.get_figure_path(), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_Sq_plus_range(spectrum, spectrum_info, hamiltonian_kwargs, eps_range, show=False, **kwargs):
    aspect = 0.3
    ws = spectrum_info["ws"]
    extent = eps_range[0], eps_range[-1], np.nanmin(ws), np.nanmax(ws)
    plt.subplots()
    plt.imshow(spectrum[:, :].T, aspect=aspect, origin="lower", cmap=cm.gnuplot, extent=extent)
    cbar = plt.colorbar()
    if "prime" in spectrum_info["name"]:
        s_label = "{S^\prime}"
        prime = "prime"
    else:
        s_label = "S"
        prime = ""
    plt.xlabel(r"$\epsilon^\prime$ [eV]")
    plt.ylabel(r"$\omega$ [eV]")
    cbar.set_label(rf"${s_label}^-_0 {s_label}^+_0$ [a. u.]")
    plt.savefig(spectrum.get_figure_path(), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_Sq_plus(ws, spectrum, spectrum_info, hamiltonian_kwargs, plot_info, show=False, **kwargs):
    ext_spectrum = np.empty(np.array(spectrum.shape) + [1, 0, 0])
    ext_spectrum[:-1, :, :] = spectrum
    ext_spectrum[-1, :, :] = spectrum[0, :, :]
    mkx = ext_spectrum.shape[
        0
    ]  # TODO: these extents assume too much about input qs
    k_size = 0.5 / mkx
    extent = -k_size, 1 + k_size, np.nanmin(ws), np.nanmax(ws)
    aspect = 1.3
    plt.subplots()
    if "prime" in spectrum_info["name"]:
        clabel = r"{d^\prime}"
        prime = "prime"
    else:
        clabel = "d"
        prime = ""
    clabel = r"$\chi_{+-}^" + clabel + "$ [a. u.]"
    for qy in range(spectrum.shape[1]):
        for qx in range(spectrum.shape[0]):
            ys = ext_spectrum[qx, qy, :]
            plt.plot(ws[qx, qy], ys, label=f"q=[{qx},{qy}]")
    plt.ylabel(clabel)
    plt.xlabel(r"$\omega$ [eV]")
    plt.xlim([np.amin(ws), np.amax(ws)])
    plt.ylim([0, None])
    plt.legend()
    plt.savefig(spectrum.get_figure_path(operator_name_suffix=f"{prime}+qs"), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    # for y in range(spectrum.shape[1]):
    #     plt.figure()
    #     ys = ext_spectrum[:,y,:].T
    #     plt.imshow(ys,aspect=aspect,origin='lower',cmap=cm.gnuplot,extent=extent)
    #     cbar = plt.colorbar()
    #     plt.xlabel(r'$k_x \, [2\pi / a]$')
    #     plt.ylabel(r'$\omega$ [eV]')
    #     cbar.set_label(clabel)
    #     plt.savefig(spectrum.get_figure_path(operator_name_suffix=f"{prime}+q_qy{y}_map"), bbox_inches="tight")
    #     if show:
    #         plt.show()
    #     else:
    #         plt.close()

    #     dos = np.sum(ys,axis=1)
    #     plt.plot(ws,dos)
    #     plt.xlim([0,ws[-1]])
    #     plt.ylim([0,np.nanmax(dos)])
    #     plt.xlabel(r'$\omega$ [eV]')
    #     plt.ylabel(r'DOS')
    #     plt.savefig(spectrum.get_figure_path(operator_name_suffix=f"{prime}+q_qy{y}_dos"), bbox_inches="tight")
    #     if show:
    #         plt.show()
    #     else:
    #         plt.close()
