import json
import logging.config
import pathlib
import sys

from exactdiag.logging_utils import setup_logging
from exactdiag.tJ_spin_half_ladder import api


def run_example(config_file: pathlib.Path | str = None):
    if config_file is None:
        config_file = pathlib.Path(__file__).with_suffix(".json")
    config = api.Full_Spectrum_Config.load(config_file)

    omega_mins = {
        "current_rung": 0,  # Note: this may not plot what you would expect and logs a warning.
        "current_leg": -0.5,  # Calculating negative values allows us to suppres an un-physical peak at 0 energy.
        "Szq": 0,  # Note: this logs a warning.
        "spectral_function": -2,
    }
    for name, omega_min in omega_mins.items():
        config.spectrum.name = name
        config.spectrum.omega_min = omega_min
        api.plot_excitation_spectrum(config=config, show=True, limited_qs=False)


if __name__ == "__main__":
    args = sys.argv
    params_fname = args[1] if len(args) >= 2 else None

    if len(args) >= 3:
        with open(args[2], mode="r", encoding="utf-8") as fp:
            logging.config.dictConfig(json.load(fp))
    else:
        setup_logging()

    run_example(params_fname)
