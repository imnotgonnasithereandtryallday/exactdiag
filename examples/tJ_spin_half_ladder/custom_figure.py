import copy
import json
import logging.config
import pathlib
import sys

import matplotlib.pyplot as plt

from exactdiag.logging_utils import setup_logging
from exactdiag.tJ_spin_half_ladder import api
from exactdiag.tJ_spin_half_ladder import configs


if __name__ == "__main__":
    args = sys.argv
    this_file_path = pathlib.Path(args[0])
    params_fname = args[1] if len(args) >= 2 else this_file_path.with_suffix(".json")

    if len(args) >= 3:
        with open(args[2], mode="r", encoding="utf-8") as fp:
            logging.config.dictConfig(json.load(fp))
    else:
        setup_logging()

    # Custom figure -- retarded spectral function of a combined systems of N and N-2 spins
    # (2Stot in [0,1], equal for both systems).
    # Assumes that the GS has the same kx,ky in both systems.
    # The parametes file defines the system with the higher number of holes.
    # see [poilblanc scalapino capponi prl 2003 fig. 2]

    # Note: The Hamiltonian size logged on config creation includes all four terms.
    #       We do not need the whole Hamiltonian in RAM at any point,
    #       but swapping matrices in RAM during diagonalization makes it very slow.
    # ruff: noqa: N816
    config_Nless2 = api.Config.load(params_fname)
    config_N = copy.deepcopy(config_Nless2)
    config_N.hamiltonian.num_holes = config_Nless2.hamiltonian.num_holes - 2
    config_Nless2.spectrum.name = "spectral_function_plus"
    config_N.spectrum.name = "spectral_function_minus"
    # We ignore the operator_symmetry_qs values set in the config here.
    config_Nless2.spectrum.operator_symmetry_qs = configs.Quantum_Numbers(leg=0, rung=0)
    config_N.spectrum.operator_symmetry_qs = configs.Quantum_Numbers(leg=0, rung=0)

    eigval_Nless2, _ = api.get_eigenpairs(config_Nless2)
    eigval_N, _ = api.get_eigenpairs(config_N)

    reference_energy = 0.5 * (eigval_N[0] + eigval_Nless2[0])
    energy_shift_N = eigval_N[0] - reference_energy
    energy_shift_Nless2 = eigval_Nless2[0] - reference_energy
    mkx = config_N.hamiltonian.num_rungs // 2 + 1
    mky = 2
    for qx in range(mkx):
        for qy in range(mky):
            ax = plt.subplot(mkx, mky, 1 + qy + 2 * qx)
            config_Nless2.spectrum.operator_symmetry_qs.leg = qx
            config_Nless2.spectrum.operator_symmetry_qs.rung = qy
            config_N.spectrum.operator_symmetry_qs.leg = -qx
            config_N.spectrum.operator_symmetry_qs.rung = (-qy) % 2
            ws_Nless2, spectrum_Nless2 = api.get_spectrum(config_Nless2)
            ws_N, spectrum_N = api.get_spectrum(config_N)
            ax.set_xlim([-4, 2])
            ax.set_ylim([0, 10])
            ax.plot(ws_Nless2 + energy_shift_Nless2, spectrum_Nless2, label="$A^+_{N-2}$")
            ax.plot(-ws_N - energy_shift_N, spectrum_N, label="$A^-_N$")
            ax.plot([], lw=0, label=f"q={qx},{qy}")
            ax.legend()
    plt.show()
