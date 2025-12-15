import copy
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from typing import Literal

from jaxtyping import Array, Float
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from exactdiag.general.sparse_matrices import Sparse_Matrix
from exactdiag.tJ_spin_half_ladder import api, configs

# ruff: noqa - # FIXME: This file is a hot mess.


def mutate_hamiltonian_config(config: configs.Hamiltonian_Config, j_to_t=None, num_holes=None, ks=None) -> None:
    if j_to_t is not None:
        config.weights.jl = j_to_t
        config.weights.jr = j_to_t
    if num_holes is not None:
        config.num_holes = num_holes
    if ks is not None:
        config.symmetry_qs.leg = ks[0]
        config.symmetry_qs.rung = ks[1]


def calculate_values(excited_state, eigenstate_for_projection, hamiltonain_matrices, hpar):
    projected_state = eigenstate_for_projection * np.vdot(eigenstate_for_projection, excited_state)
    projected_norm = np.linalg.norm(projected_state)
    projected_state /= projected_norm
    tl_state = hpar[0] * hamiltonain_matrices[0].dot(projected_state)
    tr_state = hpar[1] * hamiltonain_matrices[1].dot(projected_state)
    jl_state = hpar[2] * hamiltonain_matrices[2].dot(projected_state)
    jr_state = hpar[3] * hamiltonain_matrices[3].dot(projected_state)
    tl_contr = np.vdot(projected_state, tl_state).astype(float)
    tr_contr = np.vdot(projected_state, tr_state).astype(float)
    jl_contr = np.vdot(projected_state, jl_state).astype(float)
    jr_contr = np.vdot(projected_state, jr_state).astype(float)
    t_contr = np.vdot(projected_state, tl_state + tr_state).astype(float)
    j_contr = np.vdot(projected_state, jl_state + jr_state).astype(float)
    l_contr = np.vdot(projected_state, tl_state + jl_state).astype(float)
    r_contr = np.vdot(projected_state, tr_state + jr_state).astype(float)
    return projected_norm, tl_contr, tr_contr, jl_contr, jr_contr, t_contr, j_contr, l_contr, r_contr


def load_hamiltonian_matrices(
    config: configs.Hamiltonian_Config,
) -> tuple[Sparse_Matrix, Sparse_Matrix, Sparse_Matrix, Sparse_Matrix]:
    tl_matrix = Sparse_Matrix.from_name(name="tl", config=config)
    tr_matrix = Sparse_Matrix.from_name(name="tr", config=config)
    jl_matrix = Sparse_Matrix.from_name(name="jl", config=config)
    jr_matrix = Sparse_Matrix.from_name(name="jr", config=config)
    return tl_matrix, tr_matrix, jl_matrix, jr_matrix


def load_spec_func_matrix(config: configs.Config, pm: Literal["plus", "minus"]) -> Sparse_Matrix:
    # originally 'plus' if pm else 'minus'
    if config.spectrum.operator_symmetry_qs is None:
        raise ValueError("config.spectrum.operator_symmetry_qs must not be None.")
    matrix_name = f"spectral_function_{pm}_{config.spectrum.operator_symmetry_qs.to_npint32()}"
    return Sparse_Matrix.from_name(name=matrix_name, num_threads=config.spectrum.num_threads, config=config.hamiltonian)


def get_shift(dj: float, qx: int, qy: int, mkx: int) -> float:
    return -0.4 * dj + 0.8 * dj * (0.5 + qx + 0.25 * qy - 0.125) / mkx


def plot_observable(
    x, yin, mkx, mky, num_excited, ylim=None, is_energy=False, ylabel="", converge=False, important_qs=None
):
    markers = ["o", "s", "*", "v"]
    cs = [f"C{qx}" for qx in range(mkx)]
    plt.gcf().set_size_inches([15, 4])
    y = np.array(yin)
    dj = np.amin(x[1:] - x[:-1])
    if is_energy:
        minima = np.amin(y, axis=(2, 3, 4))
        centers = (minima[:, 0] - minima[:, 1]) / 2
        y[:, 0, :, :, :] *= -1
        y += centers[:, None, None, None, None]
    for pm in range(2):
        for qy in range(mky):
            for qx in range(mkx):
                shift = get_shift(dj, qx, qy, mkx)
                mfc = None if pm else "none"
                for exc_i in range(num_excited):
                    if important_qs is not None:
                        mask = important_qs[:, pm, qx, qy, exc_i]
                    else:
                        mask = np.ones(len(x), dtype=np.bool)
                    plt.plot(
                        x[mask] + shift,
                        y[mask, pm, qx, qy, exc_i],
                        marker=markers[exc_i],
                        c=cs[qx],
                        lw=0,
                        mfc=mfc,
                        ms=6,
                    )

    if ylim is not None:
        plt.ylim(ylim)
    # plot vertical lines
    y0, y2 = plt.gca().get_ylim()
    y1, y3 = y0, y2
    if converge and (converge == "bot" or "bot" in converge):
        y1 = 0.95 * y0
    if converge and (converge == "top" or "top" in converge):
        y3 = 0.95 * y2
    #     y1 = np.amin(y)
    for i, v in enumerate(x):
        for qx in range(mkx):
            shift = v + (get_shift(dj, qx, 1, mkx) + get_shift(dj, qx, 0, mkx)) / 2
            #             plt.axvline(shift,c=cs[qx],ls='dotted')
            if converge and (converge == "mid" or "mid" in converge):
                gap = 0.07
                plt.vlines(shift, y1, -gap, colors=cs[qx], ls=":")
                plt.vlines(shift, gap, y3, colors=cs[qx], ls=":")
                plt.plot([shift, v], [-gap, 0], c=cs[qx], ls=":")
                plt.plot([v, shift], [0, gap], c=cs[qx], ls=":")
            else:
                plt.vlines(shift, y1, y3, colors=cs[qx], ls=":")
            if converge and (converge == "bot" or "bot" in converge):
                plt.plot([shift, v], [y1, y0], c=cs[qx], ls=":")
            if converge and (converge == "top" or "top" in converge):
                plt.plot([shift, v], [y3, y2], c=cs[qx], ls=":")

    # plot empty data for legend entries
    for exc_i in range(num_excited):
        plt.plot([], c="k", marker=markers[exc_i], lw=0, label=f"{exc_i}-th state")
    handles, labels = plt.gca().get_legend_handles_labels()
    for qx in range(mkx):
        lines = []
        for exc_i in range(num_excited):
            p, *_ = plt.plot([], marker=markers[exc_i], c=cs[qx], lw=0)
            lines.append(p)
        handles.append(tuple(lines))
        labels.append(f"   qx={qx}")
    plt.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.05),
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=2, xpad=0.7)},
    )
    plt.title("for each color -- left:qy=0, right: qy=1")
    plt.xticks(x)
    plt.xlim(x[0] + get_shift(dj, -1, 0, mkx), x[-1] + get_shift(dj, mkx, mky, mkx))
    plt.xlabel("$J$ [$t$]")
    plt.ylabel(ylabel)
    return plt.gcf()


def select_q_combinations_forming_gap(j_to_ts, energies, gap_proximity_factor):
    """Returns identification of states and pairs of states with energies around the gap."""
    important_qs = np.zeros(energies.shape, dtype=np.bool)
    *_, mkx, mky, num_excited = energies.shape
    gap_all_combinations = np.empty((len(j_to_ts), mkx, mky, mky, num_excited, num_excited))
    for qy in range(mky):
        for qpy in range(mky):
            for exc1 in range(num_excited):
                for exc2 in range(num_excited):
                    gap_all_combinations[:, :, qy, qpy, exc1, exc2] = (
                        energies[:, 0, :, qy, exc1] + energies[:, 1, :, qpy, exc2]
                    )
    gaps = np.amin(gap_all_combinations[:, :, :, :, 0, 0], axis=(1, 2, 3))
    pairs = np.zeros([len(j_to_ts), mkx, mky, mky, num_excited, num_excited], dtype=np.bool)
    for qy in range(mky):
        for qpy in range(mky):
            for exc1 in range(num_excited):
                for exc2 in range(num_excited):
                    is_low = gap_all_combinations[:, :, qy, qpy, exc1, exc2] < gap_proximity_factor * gaps[:, None]
                    important_qs[:, 0, :, qy, exc1] = np.logical_or(important_qs[:, 0, :, qy, exc1], is_low)
                    important_qs[:, 1, :, qpy, exc2] = np.logical_or(important_qs[:, 1, :, qpy, exc2], is_low)
                pairs[:, :, qy, qpy, exc1, exc2] = is_low
    return important_qs, pairs

def plot_pair_observable(
    x, y, pairs, mkx, mky, num_excited, ylim=None, is_multiplied=True, ylabel="", converge=False
):
    markers = ["o", "s", "*", "v"]
    cs = [f"C{qx%10}" for qx in range(mkx)]
    dj = 0.1  # (j_to_ts[1]-j_to_ts[0])
    plt.gcf().set_size_inches([15, 4])
    for ix, v in enumerate(x):
        for qx in range(mkx):
            for iq, (qy, qpy) in enumerate(product(range(2), range(2))):
                shift = get_shift(dj, iq, 0.5, 4)
                for ie, (exc1, exc2) in enumerate(product(range(num_excited), range(num_excited))):
                    if pairs[ix, qx, qy, qpy, exc1, exc2]:
                        if is_multiplied:
                            plt.plot(
                                [v + shift],
                                (y[ix, 0, qx, qy, exc1] * y[ix, 1, qx, qpy, exc2]),
                                marker=markers[(exc1 + exc2)%len(markers)],
                                c=cs[qx],
                                lw=0,
                                ms=6,
                            )
                        else:
                            plt.plot(
                                [v + shift],
                                (y[ix, 0, qx, qy, exc1] + y[ix, 1, qx, qpy, exc2]),
                                marker=markers[(exc1 + exc2)%len(markers)],
                                c=cs[qx],
                                lw=0,
                                ms=6,
                            )

    if ylim is not None:
        plt.ylim(ylim)
    # plot vertical lines
    y0, y2 = plt.gca().get_ylim()
    y1, y3 = y0, y2
    if converge and (converge == "bot" or "bot" in converge):
        y1 = 0.95 * y0
    if converge and (converge == "top" or "top" in converge):
        y3 = 0.95 * y2
    #     y1 = np.amin(y)
    for i, v in enumerate(x):
        for iq in range(4):
            shift = v + get_shift(dj, iq, 0.5, 4)
            #             plt.axvline(shift,c=cs[qx],ls='dotted')
            if converge and (converge == "mid" or "mid" in converge):
                gap = 0.07
                plt.vlines(shift, y1, -gap, colors="k", ls=":")
                plt.vlines(shift, gap, y3, colors="k", ls=":")
                plt.plot([shift, v], [-gap, 0], c="k", ls=":")
                plt.plot([v, shift], [0, gap], c="k", ls=":")
            else:
                plt.vlines(shift, y1, y3, colors="k", ls=":")
            if converge and (converge == "bot" or "bot" in converge):
                plt.plot([shift, v], [y1, y0], c="k", ls=":")
            if converge and (converge == "top" or "top" in converge):
                plt.plot([shift, v], [y3, y2], c="k", ls=":")

    # plot empty data for legend entries
    for exc_i in range(num_excited):
        plt.plot([], c="k", marker=markers[exc_i], lw=0, label=rf"$i+i^\prime={exc_i}$")
    handles, labels = plt.gca().get_legend_handles_labels()
    for qx in range(mkx):
        lines = []
        for exc_i in range(num_excited):
            p, *_ = plt.plot([], marker=markers[exc_i], c=cs[qx], lw=0)
            lines.append(p)
        handles.append(tuple(lines))
        labels.append(f"   qx={qx}")
    plt.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.05),
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=2, xpad=0.7)},
    )
    plt.title(
        r"dotted lines from left to right: [$q_y,\,q^\prime_y$] corresponding to $A(q_y),\,A^\dagger(q_y)$" + "\n"
        r"$[q_y,\,q^\prime_y]=[0,0],[0,1],[1,0],[1,1]$"
    )
    plt.xticks(x)
    plt.xlim(x[0] + get_shift(dj, -1, 0, mkx), x[-1] + get_shift(dj, mkx, mky, mkx))
    plt.xlabel("$J$ [$t$]")
    plt.ylabel(ylabel)
    return plt.gcf()


def print_selected(num_excited, ix, qx, qy, qpy, y, pairs, is_multiplied=False):
    s = []
    for exc1, exc2 in product(range(num_excited), range(num_excited)):
        if pairs[ix, qx, qy, qpy, exc1, exc2]:
            if is_multiplied:
                v = y[ix, 0, qx, qy, exc1] * y[ix, 1, qx, qpy, exc2]
            else:
                v = y[ix, 0, qx, qy, exc1] + y[ix, 1, qx, qpy, exc2]
            s.append(f"[{qx},{qy},{qpy}|{exc1},{exc2}]: {v:g}")
    print(", ".join(s))


_AXES = "j 2 num_rungs_half_plus_one num_legs num_excited_states"


@dataclass
class Excited_State_Properties:
    """Collection of properties for multiple excited states.

    The second axis corresponds to application of ["plus", "minus"]
    spectral function. `num_legs` is assumed to be 2.

    The contained values are:
        0) The J/t values used.
        1) Norm of the projection of the excited state to the initial one.
        2-10) Difference of the `<f|O|f>` and `<i|O|i>` matrix elements, where
        2) `O` is the `whole Hamiltonian term.
        3) `O` is the `tl` Hamiltonian term.
        4) `O` is the `tr` Hamiltonian term.
        5) `O` is the `jl` Hamiltonian term.
        6) `O` is the `jr` Hamiltonian term.
        7) `O` are the `tl+tr` Hamiltonian terms.
        8) `O` are the `jl+jr` Hamiltonian terms.
        9) `O` are the `tl+jl` Hamiltonian terms.
        10) `O` are the `tr+jr` Hamiltonian terms.
    """

    j_to_ts: Float[Array, "j"]  # noqa: F821 - "j" is not a forward declaration
    projected_norms: Float[Array, "{_AXES}"]
    energies: Float[Array, "{_AXES}"]
    tl_conts: Float[Array, "{_AXES}"]
    tr_conts: Float[Array, "{_AXES}"]
    jl_conts: Float[Array, "{_AXES}"]
    jr_conts: Float[Array, "{_AXES}"]
    t_conts: Float[Array, "{_AXES}"]
    j_conts: Float[Array, "{_AXES}"]
    l_conts: Float[Array, "{_AXES}"]
    r_conts: Float[Array, "{_AXES}"]


def calculate_excited_state_properties(
    config: configs.Config,
    j_to_ts: list[float],
    num_excited_states: int,
    verbose: bool,
) -> Excited_State_Properties:
    """Return properties of the excited state compared to the initial one.

    Calculates the excited state by applying a spectral function.
    We assume that the gs has k=[0,0], reducing the number of unique spectral functions.
    Compares various properties of the excited and the initial states.
    """

    def qualified_print(*args):
        if verbose:
            print(*args)

    config = copy.deepcopy(config)
    excited_config = copy.deepcopy(config)
    # for gs with k=[0,0], we do not need to calculate all momenta due to symmetry.
    num_kx, num_ky = config.hamiltonian.num_rungs//2 + 1, 2
    shape = [len(j_to_ts), 2, num_kx, num_ky, num_excited_states]
    tl_conts = np.empty(shape)
    tr_conts = np.empty(shape)
    jl_conts = np.empty(shape)
    jr_conts = np.empty(shape)
    t_conts = np.empty(shape)
    j_conts = np.empty(shape)
    l_conts = np.empty(shape)
    r_conts = np.empty(shape)
    energies = np.empty(shape)
    projected_norms = np.empty(shape)

    qualified_print(datetime.now(), "start")
    for ij, j_to_t in enumerate(j_to_ts):
        mutate_hamiltonian_config(config.hamiltonian, j_to_t=j_to_t)
        hpar = [1, 1, j_to_t, j_to_t]
        eigvals, eigvecs = api.get_eigenpairs(config)
        qualified_print(datetime.now(), f"eigenvalues for {hpar} got.")
        gs = eigvecs[:, 0]
        gs_en = eigvals[0]
        h_matrices = load_hamiltonian_matrices(config.hamiltonian)
        _, gs_tl, gs_tr, gs_jl, gs_jr, gs_t, gs_j, gs_l, gs_r = calculate_values(gs, gs, h_matrices, hpar)

        for pm, pms in enumerate(["minus", "plus"]):
            # "plus" means calculating a spectral function that adds an electron.
            for qx in range(num_kx):
                for qy in range(num_ky):
                    config.spectrum.operator_symmetry_qs = copy.deepcopy(
                        config.hamiltonian.symmetry_qs
                    )  # Ugly hack to avoid having None there.
                    config.spectrum.operator_symmetry_qs.leg = qx if pms == "plus" else -qx
                    config.spectrum.operator_symmetry_qs.rung = qy
                    exc_op = load_spec_func_matrix(config, pms)
                    exc_state = exc_op.dot(gs)
                    new_num_holes = config.hamiltonian.num_holes + (-1 if pms == "plus" else 1)
                    mutate_hamiltonian_config(
                        excited_config.hamiltonian,
                        j_to_t=j_to_t,
                        num_holes=new_num_holes,
                        ks=[config.spectrum.operator_symmetry_qs.leg, qy],
                    )
                    eigvals, eigvecs = api.get_eigenpairs(excited_config)
                    h_matrices = load_hamiltonian_matrices(excited_config.hamiltonian)
                    for exc_i in range(num_excited_states):
                        i = (ij, pm, qx, qy, exc_i)
                        energies[i] = eigvals[exc_i] - gs_en
                        eigenstate_for_projection = eigvecs[:, exc_i]
                        (
                            projected_norms[i],
                            tl_conts[i],
                            tr_conts[i],
                            jl_conts[i],
                            jr_conts[i],
                            t_conts[i],
                            j_conts[i],
                            l_conts[i],
                            r_conts[i],
                        ) = calculate_values(exc_state, eigenstate_for_projection, h_matrices, hpar)
                        tl_conts[i] -= gs_tl
                        tr_conts[i] -= gs_tr
                        jl_conts[i] -= gs_jl
                        jr_conts[i] -= gs_jr
                        t_conts[i] -= gs_t
                        j_conts[i] -= gs_j
                        l_conts[i] -= gs_l
                        r_conts[i] -= gs_r
                    qualified_print(datetime.now(), j_to_t, pm, qx, qy, "done")
    qualified_print(datetime.now(), "all done")
    return Excited_State_Properties(
        j_to_ts=np.array(j_to_ts),
        projected_norms=projected_norms,
        energies=energies,
        tl_conts=tl_conts,
        tr_conts=tr_conts,
        jl_conts=jl_conts,
        jr_conts=jr_conts,
        t_conts=t_conts,
        j_conts=j_conts,
        l_conts=l_conts,
        r_conts=r_conts,
    )
