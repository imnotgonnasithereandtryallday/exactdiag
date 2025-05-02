# ruff: noqa
"""To cythonize and compile files, run
python setup.py build_ext

To build a wheel, run
python setup.py bdist_wheel

Note: It caches a lot of things. Deleteting `build/`, `dist/` and `exactdiag.egg-info` folder might help.
"""
# TODO: This use of setup.py is now deprecated.


from pathlib import Path
import argparse
import glob

import numpy as np
from setuptools import setup, Extension
from Cython.Compiler.Version import version
from Cython.Build import cythonize
import Cython.Compiler.Options

PACKAGE_NAME = "exactdiag"

def _get_cythonized_extension_modules(enable_profiling):
    print(f"cython version={version}; profiling={enable_profiling}")

    general_subname = "general"
    package_path = Path("src") / PACKAGE_NAME
    general_path = package_path / general_subname
    general_fullname = f"{PACKAGE_NAME}.{general_subname}"
    group_subname = "group"
    group_path = general_path / group_subname
    group_fullname = f"{general_fullname}.{group_subname}"
    ladder_subname = "tJ_spin_half_ladder"
    ladder_path = package_path / ladder_subname
    ladder_fullname = f"{PACKAGE_NAME}.{ladder_subname}"
    hubbard_subname = "hubbard"
    hubbard_path = package_path / hubbard_subname
    hubbard_fullname = f"{PACKAGE_NAME}.{hubbard_subname}"
    benchmark_subname = "diagonalization_benchmark"
    benchmark_path = package_path / benchmark_subname
    benchmark_fullname = f"{PACKAGE_NAME}.{benchmark_subname}"

    macros = []
    if enable_profiling:
        macros.append(("CYTHON_TRACE_NOGIL", 1))

    # os.environ["CC"] = "clang++" # select c++ compiler # FIXME: add as argument when calling setup.
    shared_ext_settings = {
        "language": "c++",
        "extra_compile_args": ["/openmp", "/O2", "/std:c++20"],  # ['/openmp:experimental', '/O2', '/fp:fast', '/GL',  \
        # '/QIntel-jcc-erratum'],# '/favor:AMD64'], # FIXME: these should probably be a cmd-line arguments.
        "define_macros": macros,
        "extra_link_args": ["/openmp", "-fopenmp"],
    }
    names = [
        (general_fullname, general_path,
        [
            "types_and_constants",
            "cython_sparse_dot",
            "cpp_classes",
        ]),
        (group_fullname, group_path, ["symmetry_generator"]),
        # TODO: This intermission means bad folder structure.
        #       Does the order here actually matter?
        (general_fullname, general_path,
        [
            "symmetry",
            "basis_indexing",
            "symmetry_utils",
            "column_functions",
            "matrix_rules_utils",
            "cython_sparse_matrices",
            "cython_lanczos_diagonalization",
        ]),
        (ladder_fullname, ladder_path,
        [
            "matrix_element_functions",
            "symmetry",
            "matrix_setup",
        ]),
    ]
    extensions = []
    for full_name, full_path, module_names in names:
        for module_name in module_names:
            extensions.append(Extension(
                f"{full_name}.{module_name}",
                [f"{full_path / f'{module_name}.pyx'}"],
                **shared_ext_settings,
            ))

    Cython.Compiler.Options.docstring = False
    Cython.Compiler.Options.error_on_uninitialized = True
    directives = {
        "language_level": "3",  # We assume Python 3 code
        "boundscheck": False,  # Do not check array access
        "wraparound": False,  # a[-1] does not work
        "embedsignature": False,  # Do not save typing / docstring
        "always_allow_keywords": False,  # Faster calling conventions
        "initializedcheck": False,  # We assume memory views are initialized
        "cdivision": True,
        "nonecheck": False,
        "binding": False,
    }
    if enable_profiling:
        directives["profile"] = True
        directives["linetrace"] = True
        directives["binding"] = True

    ext_modules = cythonize(
        extensions,
        compiler_directives=directives,
        emit_linenums=enable_profiling,
        force=True,
        build_dir="build/"
    )  # show_all_warnings=True)

    # cythonize also copies the header files, which causes problems with
    # pragma once not guarding properly because there are two identical files at different locations.
    # I do not know how to stop cythonize from copying them.
    for dirpath, _, filenames in Path("build/src").walk():
        for name in filenames:
            if name[-2:] == ".h":
                (dirpath / name).unlink()
    # Still, there is a problem if I have multiple headers with the same name in different folders.
    # TODO: Can we get a better build backend?
    return ext_modules


setup(
    name=PACKAGE_NAME,
    ext_modules=_get_cythonized_extension_modules(enable_profiling=False),
    include_dirs=[np.get_include()],
    package_dir={PACKAGE_NAME: f"src/{PACKAGE_NAME}"},
    package_data={PACKAGE_NAME: ["logging_config.json"]},
    install_requires=[
        "numpy",
    ],
    zip_safe=False,
)
