import glob
import importlib.util
import pathlib
import re
import sys

import matplotlib.pyplot as plt


def test_running_examples():
    """All examples should complete successfully."""
    # NOTE: skips .ipynb files.
    files = glob.glob("examples/tJ_spin_half_ladder/*.py")
    expected_num_files = 7
    assert len(files) == expected_num_files

    for i, file in enumerate(files):
        name = _str_to_variable_name(pathlib.Path(file).stem)
        m = _import_from_path(f"m{i}_{name}", file)
        with plt.ion():
            m.run_example()


def _import_from_path(module_name, file_path):
    # Taken from
    # https://docs.python.org/3/library/importlib.html#importing-programmatically
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _str_to_variable_name(string: str):
    """Return sanitized string that can be used as a variable name."""
    return re.sub(r"\W+|^(?=\d)", "_", string)
