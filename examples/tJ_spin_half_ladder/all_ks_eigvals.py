import json
import logging.config
import pathlib
import sys

from exactdiag.logging_utils import setup_logging
from exactdiag.tJ_spin_half_ladder import api


def run_example(config_file: pathlib.Path | str = None):
    if config_file is None:
        config_file = pathlib.Path(__file__).with_suffix(".json")
    config = api.Config.load(config_file)

    k_to_pairs = api.get_all_k_eigenpairs(config)
    print("(kx,ky), [eigenvalues], eigenvector length")
    for k, (vals, vecs) in k_to_pairs.items():
        print(k, vals, vecs.shape[0])


if __name__ == "__main__":
    args = sys.argv
    params_fname = args[1] if len(args) >= 2 else None

    if len(args) >= 3:
        with open(args[2], mode="r", encoding="utf-8") as fp:
            logging.config.dictConfig(json.load(fp))
    else:
        setup_logging()

    run_example(params_fname)
