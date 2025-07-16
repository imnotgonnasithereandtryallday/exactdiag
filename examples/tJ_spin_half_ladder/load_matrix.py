from dataclasses import asdict
import json
import logging.config
import pathlib
import sys

import numpy as np

from exactdiag.logging_utils import setup_logging
from exactdiag.tJ_spin_half_ladder import api
from exactdiag.tJ_spin_half_ladder import configs
from exactdiag.general.sparse_matrices import Sparse_Matrix, Added_Sparse_Matrices


def run_example(config_file: pathlib.Path | str = None):
    if config_file is None:
        config_file = pathlib.Path(__file__).with_suffix(".json")
    config = configs.Eigenpair_Config.load(config_file)

    eigvals, eigvecs = api.get_eigenpairs(config)
    ground_state = eigvecs[:, 0]
    weights = asdict(config.hamiltonian.weights)
    matrices = Added_Sparse_Matrices(
        *zip(*[
            [Sparse_Matrix.from_name(name=name, config=config.hamiltonian), weight] for name, weight in weights.items()
        ])
    )
    trial_vec = matrices.dot(ground_state)
    print(f"|E_0 * gs - H @ gs| = {np.linalg.norm(eigvals[0] * ground_state - trial_vec):g}")


if __name__ == "__main__":
    args = sys.argv
    params_fname = args[1] if len(args) >= 2 else None

    if len(args) >= 3:
        with open(args[2], mode="r", encoding="utf-8") as fp:
            logging.config.dictConfig(json.load(fp))
    else:
        setup_logging()

    run_example(params_fname)
