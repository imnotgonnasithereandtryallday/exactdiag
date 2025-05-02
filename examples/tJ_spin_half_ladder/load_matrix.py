from dataclasses import asdict
import json
import logging.config
import pathlib
import sys

import numpy as np

from exactdiag.logging_utils import setup_logging
from exactdiag.tJ_spin_half_ladder import api
from exactdiag.general.sparse_matrices import Sparse_Matrix, Added_Sparse_Matrices


if __name__ == "__main__":
    args = sys.argv
    this_file_path = pathlib.Path(args[0])
    params_fname = args[1] if len(args) >= 2 else this_file_path.with_suffix(".json")
    config = api.Config.load(params_fname)

    if len(args) >= 3:
        with open(args[2], mode="r", encoding="utf-8") as fp:
            logging.config.dictConfig(json.load(fp))
    else:
        setup_logging()

    eigvals, eigvecs = api.get_eigenpairs(config)
    ground_state = eigvecs[:, 0]
    num_threads = 12
    weights = asdict(config.hamiltonian.weights)
    matrices = Added_Sparse_Matrices(
        *zip(*[
            [Sparse_Matrix.from_name(name=name, config=config.hamiltonian), weight] for name, weight in weights.items()
        ])
    )
    trial_vec = matrices.dot(ground_state)
    print(f"|E_0 * gs - H @ gs| = {np.linalg.norm(eigvals[0] * ground_state - trial_vec):g}")
