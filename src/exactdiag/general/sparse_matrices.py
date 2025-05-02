from json import load as json_load
import pathlib
import logging
import typing
from dataclasses import asdict

import numpy as np

from exactdiag.general.cython_sparse_dot import sparse_matrix_dot, block_diagonal_sparse_matrix_dot
from exactdiag.general.cython_sparse_matrices import calc_sparse_matrix, Signature
from exactdiag.general.types_and_constants import MATRIX_INDEX_TYPE, VALUE_INDEX_TYPE, VALUE_TYPE, SYMMETRY_OPTIONS
from exactdiag.general import configs
from exactdiag.utils import load_calculate_save


_logger = logging.getLogger(__name__)
FILE_NAMES = {
    "info_name": "matrix_info.json",
    "values_name": "values.npy",
    "minor_inds_name": "minor_inds.npy",
    "value_inds_name": "value_inds.npy",
    "major_pointers_name": "major_pointers.npy",
}


# TODO: Add some inheritance to the classes.
class Sparse_Matrix:
    def __init__(self, signature: Signature | dict, dtype=VALUE_INDEX_TYPE, num_threads=12):
        if isinstance(signature, dict):
            signature = Signature(**signature)

        folder = pathlib.Path(signature.folder_name)
        _logger.debug(f"Reading {folder!s}.")
        with open(folder / signature.info_name, mode="r", encoding="utf-8") as file:
            loaded_signature = json_load(file)
        loaded_signature = asdict(Signature(**loaded_signature))  # Takes care of trivial conversions.
        input_keys = asdict(signature).keys()
        conflicting_items = []
        for loaded_key, loaded_value in loaded_signature.items():
            if loaded_key not in input_keys or loaded_value != signature[loaded_key]:
                conflicting_items.append((loaded_key, loaded_value, signature[loaded_key]))
        if len(conflicting_items) > 0:
            message = (
                f"{signature['matrix_name']} matrix is already saved with different signature."
                + f"conflicts (key, saved value, requested value): {conflicting_items}."
                + f"continuing with {conflicting_items}."
            )
            _logger.warning(message)
            signature = Signature(**loaded_signature)
        self.name = signature.matrix_name
        vals = folder / signature.values_name
        value_inds = folder / signature.value_inds_name
        major_pointers = folder / signature.major_pointers_name
        minor_inds = folder / signature.minor_inds_name
        shape = signature.shape
        self.symmetry_strings = signature.symmetry_strings
        self.signature = signature
        self.num_threads = num_threads

        try:
            self.vals = np.memmap(filename=vals, dtype=VALUE_TYPE)
            self.value_inds = np.memmap(filename=value_inds, dtype=dtype)
            self.major_pointers = np.memmap(filename=major_pointers, dtype=MATRIX_INDEX_TYPE)
            self.minor_inds = np.memmap(filename=minor_inds, dtype=MATRIX_INDEX_TYPE)
            # TODO: should these be kept open?
            self.names = [vals, value_inds, major_pointers, minor_inds]
        except ValueError:  # cannot mmap an empty but existing file
            self.vals = []
            self.value_inds = []
            self.major_pointers = []
            self.minor_inds = []

        if shape is None:
            self.shape = [len(self.major_pointers) - 1] * 2
        else:
            self.shape = shape

        # TODO: Some of the attributes should be properties to avoid having duplicate data.
        self.is_row_major = signature.major == "row"
        self.is_sorted = "sorted" in self.symmetry_strings
        self.is_hermitian_triangle = "triangle_only" in self.symmetry_strings and "hermitian" in self.symmetry_strings

    def dot(self, vec, weight=1, matvec=None):
        if matvec is None:
            first_set_to_zero = True
            matvec = np.empty(self.shape[0], dtype=VALUE_TYPE)
        else:
            first_set_to_zero = False
        if len(self.minor_inds) == 0 or abs(weight) < 1e-6:
            return matvec
        sparse_matrix_dot(
            vec,
            self.vals,
            self.value_inds,
            self.minor_inds,
            self.major_pointers,
            self.is_row_major,
            self.is_sorted,
            self.is_hermitian_triangle,
            weight,
            self.num_threads,
            matvec,
            first_set_to_zero,
        )
        return matvec

    def shift_dot(self, vin, shift):
        return self.dot(vin) - shift * vin

    def toarray(self):
        """Convert to dense matrix.

        Do not use for large matrices.
        """
        mat = np.zeros(self.shape, dtype=VALUE_TYPE)
        for i in range(self.shape[1]):
            s1 = self.major_pointers[i]
            s2 = self.major_pointers[i + 1]
            for j in range(s1, s2):
                u = self.minor_inds[j]
                mat[u, i] += self.vals[self.value_inds[j]]
                if self.is_hermitian_triangle and u != i:
                    mat[i, u] += np.conj(self.vals[self.value_inds[j]])
            s1 = s2
        return mat

    @classmethod
    def from_name(
        cls, name: str, config: "configs.Hamiltonian_Config_Base", num_threads: int | None = None
    ) -> typing.Self:
        """Return a matrix from a folder specified by config.

        If `num_threads` is None, sets the number of threads the matrix should use to the value in the config.
        """
        signature_file = config.get_calc_folder() / f"{name}" / FILE_NAMES["info_name"]
        with open(signature_file, mode="r", encoding="utf-8") as file:
            signature = json_load(file)
        num_threads = config.num_threads if num_threads is None else num_threads
        return cls(signature, num_threads=num_threads)


class Block_Diagonal_Sparse_Matrix(Sparse_Matrix):
    def __init__(self, signature, block_indices, dtype=VALUE_INDEX_TYPE, num_threads=12):
        super.__init__(signature, dtype, num_threads)
        self.block_indices = block_indices
        if not self.is_row_major:
            raise ValueError("block diagonal dot not implemented for column-major matrix")
        if len(self.block_indices) < 2:
            # if we can't take advantage of the blocks, fall back onto the generically threaded version
            self.dot = super().dot

    def dot(self, vec, weight=1, matvec=None):
        if len(self.block_indices) > 1:
            # if we can't take advantage of the blocks, fall back onto the generically threaded version
            return super().dot(vec, weight, matvec)

        if matvec is None:
            first_set_to_zero = True
            matvec = np.empty(self.shape[0], dtype=VALUE_TYPE)
        else:
            first_set_to_zero = False
        if len(self.minor_inds) == 0 or abs(weight) < 1e-6:
            return matvec
        block_diagonal_sparse_matrix_dot(
            vec,
            self.block_indices,
            self.vals,
            self.value_inds,
            self.minor_inds,
            self.major_pointers,
            self.is_row_major,
            self.is_sorted,
            self.is_hermitian_triangle,
            weight,
            self.num_threads,
            matvec,
            first_set_to_zero,
        )
        return matvec


class Shuffled_Block_Diagonal_Sparse_Matrix(Sparse_Matrix):
    # FIXME: At least add a note on what is the idea.
    pass


class Diagonal_Matrix:
    def __init__(self, vals, dtype):
        try:
            self.vals = np.memmap(filename=vals, dtype=dtype)
        # should these be kept open?
        except ValueError:  # cannot mmap an empty but existing file
            self.vals = []
        self.shape = [np.len(self.vals)] * 2  # numpy array?
        self.nonzero = any(self.vals != 0)
        self.dtype = dtype

    def dot(self, vec, weight=1, matvec=None):
        if matvec is None:
            matvec = np.zeros(self.shape[0], dtype=VALUE_TYPE)
        if not self.nonzero or abs(weight) < 1e-6:
            return matvec
        matvec += weight * self.vals * vec
        return matvec

    def toarray(self):
        return np.diag(self.vals, dtype=self.dtype)


class Added_Sparse_Matrices:
    def __init__(self, mats, weights=None):
        self.refs = mats
        if any([mat.shape != mats[0].shape for mat in mats[1:]]):
            raise ValueError("incompatible shapes")
        self.shape = mats[0].shape
        self.num_matrices = len(mats)
        self.weights = np.empty(self.num_matrices, dtype=VALUE_TYPE)
        if weights is None:
            self.weights[:] = [1] * self.num_matrices
        elif len(weights) == self.num_matrices:
            self.weights[:] = weights
        else:
            raise ValueError("wrong weights supplied")
        self.num_threads = np.amax([ref.num_threads for ref in self.refs])

    def add_matrix(self, matrix, weight):
        if self.shape != matrix.shape:
            raise ValueError(f"Nonmatching shapes {self.shape} and {matrix.shape}.")
        self.weights.append(weight)
        self.num_matrices += 1
        self.refs.append(matrix)

    def dot(self, vec):
        matvec = None
        for i in range(self.num_matrices):
            matvec = self.refs[i].dot(vec, self.weights[i], matvec)
        return matvec

    def toarray(self):
        mat = np.zeros(self.shape, dtype=VALUE_TYPE)
        for i in range(self.num_matrices):
            mat += self.weights[i] * self.refs[i].toarray()
        return mat


def get_sparse_matrices(set_of_matrix_kwargs, weights):
    matrices = [get_sparse_matrix(**matrix_kwargs) for matrix_kwargs in set_of_matrix_kwargs]
    weight_list = [weights[mat.name] for mat in matrices]
    added_matrices = Added_Sparse_Matrices(matrices, weights=weight_list)

    matrix_names = [matrix_kwargs["signature"]["matrix_name"] for matrix_kwargs in set_of_matrix_kwargs]
    _logger.info(f"Matrices got {matrix_names}")
    return added_matrices


def get_sparse_matrix(column_func, max_values_per_column, shape, signature, num_threads, **kwargs):
    load_fun = lambda *args, **kwargs: Sparse_Matrix(signature, num_threads=num_threads)
    num_columns = shape[1]

    def calc_fun(*args, **kwargs):
        calc_sparse_matrix(column_func, num_columns, max_values_per_column, num_threads, signature)
        return Sparse_Matrix(signature, num_threads=num_threads)

    def save_fun(*args, **kwargs):
        # the matrix and signature are memory-mapped in calc_sparse_matrix
        pass

    path = pathlib.Path(signature["folder_name"]) / signature["info_name"]
    matrix = load_calculate_save(
        path,
        load_fun=load_fun,
        calc_fun=calc_fun,
        save_fun=save_fun,
    )
    return matrix


def symmetry_int_from_strings(*strings):
    symmetry = 0
    for s in strings:
        for j, value in enumerate(SYMMETRY_OPTIONS):
            if s == value:
                symmetry += 2**j
    return symmetry


def symmetry_strings_from_int(i):
    strings = []
    for j, value in enumerate(SYMMETRY_OPTIONS):
        if (i % 2 * j) // j == 1:
            strings.append(value)
    return strings
