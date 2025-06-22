import json
import logging.config
import pathlib
import sys

from exactdiag.logging_utils import setup_logging
from exactdiag.tJ_spin_half_ladder import api


if __name__ == "__main__":
    args = sys.argv
    this_file_path = pathlib.Path(args[0])
    params_fname = args[1] if len(args) >= 2 else this_file_path.with_suffix(".json")
    config = api.Combined_Position_Config.load(params_fname)

    if len(args) >= 3:
        with open(args[2], mode="r", encoding="utf-8") as fp:
            logging.config.dictConfig(json.load(fp))
    else:
        setup_logging()

    api.plot_position_correlation(config=config, show=True)
