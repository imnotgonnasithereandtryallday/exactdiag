import json
import logging.config
import pathlib
import sys

from exactdiag.logging_utils import setup_logging
from exactdiag.tJ_spin_half_ladder import api


def run_example(config_file: pathlib.Path | str = None):
    if config_file is None:
        config_file = pathlib.Path(__file__).with_suffix(".json")
    config = api.Full_Position_Correlation_Config.load(config_file)
    api.plot_position_correlation(config=config, show=True)


if __name__ == "__main__":
    args = sys.argv
    params_fname = args[1] if len(args) >= 2 else None

    if len(args) >= 3:
        with open(args[2], mode="r", encoding="utf-8") as fp:
            logging.config.dictConfig(json.load(fp))
    else:
        setup_logging()

    run_example(params_fname)
