"""This module provides functions for manual validation by visual comparison
of calculated spectra to figures published in
[P. Barabasová, “C-axis optical response of t-J ladders”, dissertation,
Masaryk University, Faculty of Science, Brno, 2020. https://is.muni.cz/th/vzccx/].
Each function shows one of the published figures on the right and our recreation on the left.
Functions do not recalculate the results if already saved and save their resuls if they
must do the calculation.
"""  # noqa: D205, D404

import pathlib

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

import exactdiag.tJ_spin_half_ladder.configs as lc
from exactdiag.tJ_spin_half_ladder.api import get_Szq_spectra

# TODO: Clean up duplication. Reuse plot() for fig 5.11.

REALISTIC_WEIGHTS = lc.Weights(
            tl=0.45, tr=0.36, jl=0.15, jr=0.12,)
ISOTROPIC_WEIGHTS = lc.Weights(
            tl=1, tr=1, jl=0.3, jr=0.3,)


def compare_paulina_fast():
    """Generate figures for comparison using only the small systems."""
    compare_paulina_fig4_3()
    compare_paulina_fig4_4()
    compare_paulina_fig4_5()
    compare_paulina_fig4_6()
    compare_paulina_fig4_7()


def compare_paulina_fig4_3():
    config = _get_config(num_rungs=10, num_holes=0, spectrum_name='Szq', weights=REALISTIC_WEIGHTS, state_symmetry_qs=(0,0), operator_symmetry_qs=None, omega_max=0.7,)
    ws, spectra, _ = get_Szq_spectra(config=config, limited_qs=True)
    plot(ws, spectra, pathlib.Path(__file__).parent/"paulina_figures/Fig4.3_10rungs_0holes_susceptibility.PNG")

def compare_paulina_fig4_4():
    config = _get_config(num_rungs=11, num_holes=0, spectrum_name='Szq', weights=REALISTIC_WEIGHTS, state_symmetry_qs=(0,0), operator_symmetry_qs=None, omega_max=0.7,)
    ws, spectra, _ = get_Szq_spectra(config=config, limited_qs=True)
    plot(ws, spectra, pathlib.Path(__file__).parent/"paulina_figures/Fig4.4_11rungs_0holes_susceptibility.PNG")

def compare_paulina_fig4_5():
    config = _get_config(num_rungs=12, num_holes=0, spectrum_name='Szq', weights=REALISTIC_WEIGHTS, state_symmetry_qs=(0,0), operator_symmetry_qs=None, omega_max=0.7,)
    ws, spectra, _ = get_Szq_spectra(config=config, limited_qs=True)
    plot(ws, spectra, pathlib.Path(__file__).parent/"paulina_figures/Fig4.5_12rungs_0holes_susceptibility.PNG")

def compare_paulina_fig4_6():
    fig, ax = plt.subplots()
    colors = ['k', 'r', 'b', 'g', 'y', 'm']
    for i, num_rungs in enumerate(range(7, 13)):
        config = _get_config(num_rungs=num_rungs, num_holes=0, spectrum_name='Szq', weights=REALISTIC_WEIGHTS, state_symmetry_qs=(0,0), operator_symmetry_qs=(num_rungs//2, 1), omega_max=0.25,)
        ws, spectrum, _ = get_Szq_spectra(config=config, limited_qs=True)
        ls = '-' if num_rungs % 2 == 0 else ":"
        ax.plot(ws[0,0,:], spectrum[0,0,:], label=f'{num_rungs} rungs', ls=ls, c=colors[i])

    ax.legend()
    ax.set_xlabel(r'$\omega$ [eV]')
    ax.set_ylabel(r'$\chi$ [a. u.]')
    ax.set_xlim(0, 0.25)
    ax.set_ylim(0, None)
    x_pos = 0.5
    caption_y_size = 0.1
    plt.subplots_adjust(right=x_pos, left=0.04)
    ax_image = fig.add_axes([x_pos, -caption_y_size, 1-x_pos, 1+caption_y_size])
    ax_image.axis('off') 
    image_path = pathlib.Path(__file__).parent/"paulina_figures/Fig4.6_7-12rungs_0holes_susceptibility_comparison.PNG"
    arr_img = plt.imread(image_path)
    ax_image.imshow(arr_img)
    fig.set_size_inches(18, 6)
    fig.show()


def compare_paulina_fig4_7():
    fig, ax = plt.subplots()
    for num_rungs, c in zip(range(11, 13), ['r', 'g']):
        xs = []
        ys = []
        for qx in range(num_rungs//2+1):
            config = _get_config(num_rungs=num_rungs, num_holes=0, spectrum_name='Szq', weights=REALISTIC_WEIGHTS, state_symmetry_qs=(0,0), operator_symmetry_qs=(qx, 1), omega_max=0.7,)
            ws, spectrum, _ = get_Szq_spectra(config=config, limited_qs=True)
            imax = np.argmax(spectrum[0, 0])
            xs.append(1-qx/num_rungs)
            ys.append(ws[0, 0, imax])
        ax.plot(xs, ys, label=f'{num_rungs} rungs', marker='o', c=c)
    ax.legend()
    ax.set_xlabel(r'$k_x \, [2\pi / a]$')
    ax.set_xlabel(r'$\omega$ [eV]')
    ax.set_xlim(0.5, 1)
    ax.set_ylim(0.05, 0.3)
    x_pos = 0.5
    caption_y_size = 0.1
    plt.subplots_adjust(right=x_pos, left=0.04)
    ax_image = fig.add_axes([x_pos, -caption_y_size, 1-x_pos, 1+caption_y_size])
    ax_image.axis('off') 
    image_path = pathlib.Path(__file__).parent/"paulina_figures/Fig4.7_susceptibility_maximum.PNG"
    arr_img = plt.imread(image_path)
    ax_image.imshow(arr_img)
    fig.set_size_inches(18, 6)
    fig.show()


def compare_paulina_fig5_1():
    """Note: This takes over an hour to calculate on my 14 cores."""
    # TODO: Implement.
    # config0 = _get_config(num_rungs=10, num_holes=0, spectrum_name='Szq', weights=ISOTROPIC_WEIGHTS, state_symmetry_qs=(0,0), operator_symmetry_qs=None, omega_max=1,)
    # ws0, spectra0, _ = get_Szq_spectra(config=config0, limited_qs=True)
    # config4 = _get_config(num_rungs=10, num_holes=4, spectrum_name='Szq', weights=ISOTROPIC_WEIGHTS, state_symmetry_qs=(0,0), operator_symmetry_qs=None, omega_max=1,)
    # ws4, spectra4, _ = get_Szq_spectra(config=config4, limited_qs=True)
    #plot(ws, spectra, pathlib.Path(__file__).parent/"paulina_figures/Fig4.3_10rungs_0holes_susceptibility.PNG")
    raise NotImplementedError()


def compare_paulina_fig5_11():
    """Note: This takes almost three hours to calculate on my 14 cores."""
    config = _get_config(num_rungs=12, num_holes=2, spectrum_name='Szq', weights=REALISTIC_WEIGHTS, state_symmetry_qs=(0,0), operator_symmetry_qs=None, omega_max=0.7,)
    ws, spectra, _ = get_Szq_spectra(config=config, limited_qs=True)
    plot(ws, spectra, pathlib.Path(__file__).parent/"paulina_figures/Fig5.11_12rungs_2holes_susceptibility.PNG")


def plot(ws, spectra, image_path):    
    if np.any(ws != ws[0,0,:]):
        raise ValueError("Spectra for all qx,qy need to have the same omegas.")
    ws = ws[0,0,:]
    extended_spectrum = np.empty(np.array(spectra.shape)+[1,0,0])
    extended_spectrum[:-1,:,:] = spectra
    extended_spectrum[-1,:,:] = spectra[0,:,:]
    mkx = extended_spectrum.shape[0] # TODO: these extents assume too much about input qs
    k_size = 0.5/mkx
    extent = -k_size, 1+k_size, np.amin(ws), np.amax(ws)
    aspect = 1.3
    fig, axs = plt.subplots(nrows=2, ncols=2, width_ratios=[1, 0.7])
    for iy in range(spectra.shape[1]):
        ys = extended_spectrum[:,iy,:]
        im = axs[-1-iy,0].imshow(ys.T, aspect=aspect, origin='lower', cmap=cm.gnuplot, extent=extent, interpolation="none")
        cbar = fig.colorbar(im)
        axs[-1-iy,0].set_xlabel(r'$k_x \, [2\pi / a]$')
        axs[-1-iy,0].set_ylabel(r'$\omega$ [eV]')
        cbar.set_label( r'$\chi$ [a. u.]')

        dos = np.sum(ys[:-1], axis=0)
        axs[-1-iy,1].plot(dos, ws)
        axs[-1-iy,1].set_ylim([0,ws[-1]])
        axs[-1-iy,1].set_xlim([0,np.nanmax(dos)])
        axs[-1-iy,1].set_ylabel(r'$\omega$ [eV]')
        axs[-1-iy,1].set_xlabel(r'DOS')

    fig.set_size_inches(18, 6)
    x_pos = 0.5
    caption_y_size = 0.3
    plt.subplots_adjust(right=x_pos, left=0)
    ax_image = fig.add_axes([x_pos, -caption_y_size, 1-x_pos, 1+caption_y_size])
    ax_image.axis('off') 
    arr_img = plt.imread(image_path)
    ax_image.imshow(arr_img)
    fig.show()


def _get_config(num_rungs, num_holes, spectrum_name, weights, state_symmetry_qs, operator_symmetry_qs,
                omega_max):
    num_threads = 14
    state_symmetry_qs = lc.Quantum_Numbers(leg=state_symmetry_qs[0], rung=state_symmetry_qs[1])
    hamiltonian = lc.Hamiltonian_Config(
        num_rungs=num_rungs, num_holes=num_holes, num_threads=num_threads,
        weights=weights, symmetry_qs=state_symmetry_qs,
    )
    eigenpair = lc.Eigenpair_Part(
        num_eigenpairs=5, num_threads=num_threads,
    )
    if operator_symmetry_qs is not None:
        operator_symmetry_qs = {"leg": operator_symmetry_qs[0], "rung": operator_symmetry_qs[1]}
    spectrum = lc.Spectrum_Part(
        name=spectrum_name, omega_max=omega_max, operator_symmetry_qs=operator_symmetry_qs, broadening=5e-2, num_threads=num_threads
    )
    return lc.Full_Spectrum_Config(hamiltonian=hamiltonian, eigenpair=eigenpair, spectrum=spectrum)
