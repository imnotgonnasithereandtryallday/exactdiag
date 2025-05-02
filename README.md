# Introduction
A python 3.13 package for [exact diagonalization](https://en.wikipedia.org/wiki/Exact_diagonalization).
The package facilitates:
1. Creation a basis spanning a [representation](https://en.wikipedia.org/wiki/Irreducible_representation#Overview)-invariant subspace.
2. Construction sparse matrices of operators on these subspaces.
3. Finding eigenpairs using the [Lanczos algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm)
4. Applying various operators to the state of our system and calculation the spectral decpomposition of said operators.

The main draw-back of ED is how ram-hungry it is, since the number of states grows exponentially with system size.


# Installation
To build wheel (check `build-system.requires` in [pyproject.toml](pyproject.toml) for required packages), run

`python setup.py bdist_wheel`

This creates a `exactdiag-*.whl` file (where `*` stands for version- and platform-specfic string) in the `dist/` folder.
Note: this reqiures a c++20 compiler and a linker. Also there are currently some hard-coded compiler flags in [setup.py](setup.py), 
which may not be supported by your compiler.

To install the python package and all dependencies, run

`pip install dist/exactdiag-*.whl`

To test the build (requires an additional dependency `pytest`), run

`python -m pytest tests`


# Usage
See the [`examples/`](examples/) folder.

