from libcpp cimport bool   
from libc.stdlib cimport malloc,free

cimport numpy as np     
cimport cython                  

from exactdiag.general.types_and_constants cimport MATRIX_INDEX_TYPE_t, VALUE_INDEX_TYPE_t, VALUE_TYPE_t, MAX_NUM_UNIQUIE_VALUES
from exactdiag.general.matrix_rules_utils cimport Wrapped_Column_Func
from exactdiag.general.column_functions cimport I_Lambda_Column

from json import dump
from tempfile import TemporaryFile
import logging
import typing
import pathlib
from dataclasses import asdict
import os

from cython.parallel import prange
from pydantic.dataclasses import dataclass
import numpy as np

from exactdiag.general.types_and_constants import  MATRIX_INDEX_TYPE, VALUE_INDEX_TYPE, VALUE_TYPE, SYMMETRY_OPTIONS

cdef extern from "complex.h":
    double abs(VALUE_TYPE_t) nogil


_logger = logging.getLogger(__name__)

@dataclass(kw_only=True, config={"validate_assignment": True, "extra": "forbid"})
class Signature:  # TODO: clear up typing, use already when creating the signatures
    """JSON-serializable metadata about a matrix."""

    matrix_name: str
    shape: tuple[int, int]
    symmetry_strings: tuple[str, ...]
    max_values_per_column: int
    major: typing.Literal['row', 'column']
    values_name: str
    minor_inds_name: str
    value_inds_name: str
    major_pointers_name: str
    folder_name: str
    info_name: str
    initial_system_info: dict[str, typing.Any]
    final_system_info: dict[str, typing.Any] | None = None

    def __getitem__(self, key):
        # TODO: I do not like this. remove and fix sparse_matrices.{get_sparse_matrices, get_sparse_matrix}
        return getattr(self, key)

    def __post_init__(self):
        if 'triangle_only' in self.symmetry_strings and 'hermitian' not in self.symmetry_strings:
            raise ValueError(f'symmetry option triangle_only requires the hermitian option')


cdef class _Tmp_To_Perm_Transcriber:
    def __init__(self, long long num_columns, int max_values_per_column):
        self.max_values_per_column = max(1,max_values_per_column)
        self.tmp_values = np.memmap(filename=TemporaryFile(),dtype=VALUE_TYPE,mode='w+',shape=MAX_NUM_UNIQUIE_VALUES)          
        self.tmp_value_inds = np.memmap(filename=TemporaryFile(),dtype=VALUE_INDEX_TYPE,mode='w+',shape=num_columns*self.max_values_per_column)
        self.tmp_row_inds = np.memmap(filename=TemporaryFile(),dtype=MATRIX_INDEX_TYPE,mode='w+',shape=num_columns*self.max_values_per_column)
        # column_pointers' length is known in advance
        # it is created as temporary file so that it is automatically deleted if the calculation is aborted
        self.tmp_column_pointers = np.memmap(filename=TemporaryFile(),dtype=MATRIX_INDEX_TYPE,mode='w+',shape=num_columns+1)
        self.tmp_column_pointers[0] = 0
        self.num_unique_values = 0
        self.num_nonzero_elements = 0
        self.last_filled_column_index = 0
        self.num_columns = num_columns

    cdef void organize_chunk_results(self, long chunk_size, const MATRIX_INDEX_TYPE_t* chunk_inds, \
                                    const VALUE_TYPE_t* chunk_values, const int* chunk_num_results) noexcept nogil:
        cdef MATRIX_INDEX_TYPE_t num_remaining_colums = self.num_columns - self.last_filled_column_index
        cdef MATRIX_INDEX_TYPE_t index_shift
        cdef long num_columns_in_chunk = min(chunk_size, num_remaining_colums)
        cdef long i
        cdef int j, num_nonzero_in_column
        cdef bool not_found
        cdef VALUE_TYPE_t value
        
        for i in range(num_columns_in_chunk):
            self.tmp_column_pointers[self.last_filled_column_index+1] = self.tmp_column_pointers[self.last_filled_column_index] + chunk_num_results[i] 
            self.last_filled_column_index += 1
            index_shift = i*self.max_values_per_column
            num_nonzero_in_column = chunk_num_results[i]
            
            if num_nonzero_in_column > self.max_values_per_column:
                raise IndexError(f'max_values_per_column={self.max_values_per_column} set too low, {num_nonzero_in_column} found')

            for j in range(num_nonzero_in_column):
                self.tmp_row_inds[self.num_nonzero_elements] = chunk_inds[index_shift+j] 
                value = chunk_values[index_shift+j]
                not_found = True
                for tmp_value_index in range(self.num_unique_values):
                    if abs(value - self.tmp_values[tmp_value_index]) < 1e-6:
                        self.tmp_value_inds[self.num_nonzero_elements] = tmp_value_index
                        not_found = False
                        break
                if not_found:
                    if self.num_unique_values == MAX_NUM_UNIQUIE_VALUES:
                        raise IndexError('too many unique values')
                    self.tmp_values[self.num_unique_values] = value
                    self.tmp_value_inds[self.num_nonzero_elements] = self.num_unique_values
                    self.num_unique_values += 1
                self.num_nonzero_elements += 1


    def rewrite_to_perm(self, matrix_signature: Signature):
        # Note: We do not allow signature as dict here. 
        #       This forces users to be sure the signature is valid before starting the calculation.
        folder = pathlib.Path(matrix_signature.folder_name)
        _logger.debug(f"Saving {folder}")

        os.makedirs(folder, exist_ok=True)
        with open(folder/matrix_signature.info_name, mode='w', encoding='utf-8') as fp:
            dump(asdict(matrix_signature), fp, indent=4)

        cdef MATRIX_INDEX_TYPE_t num_rows, num_columns
        cdef bool is_row_major = matrix_signature.major == 'row'
        num_rows, num_columns = matrix_signature.shape
        if self.num_unique_values == 0:
            # memmap cannot save empty file
            self.num_unique_values = 1
            self.tmp_values[0] = 0
            self.tmp_value_inds[0] = 0
            self.num_nonzero_elements = 1
            self.tmp_column_pointers[num_columns] = 1


        # the matrix is calculated column-major, but we might want to save it as row-major # TODO: move to its own functions, not even called here?
        cdef bool is_hermitian = 'hermitian' in matrix_signature.symmetry_strings
        cdef MATRIX_INDEX_TYPE_t column_index, row_index, row_pointer, element_index, start, end
        cdef MATRIX_INDEX_TYPE_t major_length = num_rows if is_row_major else num_columns
        cdef MATRIX_INDEX_TYPE_t[:] minor_inds = np.memmap(filename=folder/matrix_signature.minor_inds_name, dtype=MATRIX_INDEX_TYPE, mode='w+', shape=self.num_nonzero_elements)
        cdef MATRIX_INDEX_TYPE_t[:] major_pointers = np.memmap(filename=folder/matrix_signature.major_pointers_name, dtype=MATRIX_INDEX_TYPE, mode='w+', shape=major_length+1)
        cdef VALUE_INDEX_TYPE_t[:] value_inds = np.memmap(filename=folder/matrix_signature.value_inds_name, dtype=VALUE_INDEX_TYPE, mode='w+', shape=self.num_nonzero_elements)
        cdef VALUE_TYPE_t[:] values = np.memmap(filename=folder/matrix_signature.values_name, dtype=VALUE_TYPE, mode='w+', shape=self.num_unique_values)
        values[:] = self.tmp_values[:self.num_unique_values]

        if not is_row_major:
            # i have no idea why, but this rewriting to trim the file is very fast
            value_inds[:] = self.tmp_value_inds[:self.num_nonzero_elements]
            minor_inds[:] = self.tmp_row_inds[:self.num_nonzero_elements]
            major_pointers[:] = self.tmp_column_pointers[:]
            return

        if is_hermitian:
            minor_inds[:] = self.tmp_row_inds[:self.num_nonzero_elements]
            major_pointers[:] = self.tmp_column_pointers[:]
            value_inds[:] = self.tmp_value_inds[:self.num_nonzero_elements]
            for element_index in range(len(values)):
                values[element_index] = np.conj(values[element_index])
            return

        # first pass - get nonzero element counts for each row
        major_pointers[:] = 0
        for column_index in range(num_columns):
            for element_index in range(self.tmp_column_pointers[column_index],self.tmp_column_pointers[column_index+1]):
                major_pointers[self.tmp_row_inds[element_index]+1] += 1
        for row_index in range(num_rows):
            major_pointers[row_index+1] += major_pointers[row_index]
        # second pass - add column and value indices
        minor_inds[:] = -1
        for column_index in range(num_columns):
            for row_pointer in range(self.tmp_column_pointers[column_index],self.tmp_column_pointers[column_index+1]):
                row_index = self.tmp_row_inds[row_pointer]
                for element_index in range(major_pointers[row_index],major_pointers[row_index+1]):
                    if minor_inds[element_index] == -1:
                        minor_inds[element_index] = column_index
                        value_inds[element_index] = self.tmp_value_inds[element_index]
                        break
        # third pass - sort minor and value indices in each row
        for row_index in range(num_rows):
            start = major_pointers[row_index]
            end = major_pointers[row_index+1]
            tmp_inds = np.array(minor_inds[start:end],dtype=MATRIX_INDEX_TYPE)
            sorted_indices = np.argsort(tmp_inds)
            tmp_vals = np.array(value_inds[start:end],dtype=VALUE_INDEX_TYPE)
            for element_index in range(start,end):
                # memoryview slice can only be assigned a scalar? 
                minor_inds[element_index] = tmp_inds[sorted_indices[element_index-start]]
                value_inds[element_index] = tmp_vals[sorted_indices[element_index-start]]

cpdef void calc_sparse_matrix(Wrapped_Column_Func py_column_func, const MATRIX_INDEX_TYPE_t num_columns, 
                                    const int max_values_per_column, const int in_num_threads, matrix_signature: Signature | dict):
    """Calculate a matrix.
    
    The matrix is saved on disk.
    """
    if isinstance(matrix_signature, dict):
        matrix_signature = Signature(**matrix_signature)
    _logger.info(f"Started creating matrix {matrix_signature.matrix_name}.")

    # The calculation is split into chunks, results in each chunk have gaps
    # (exact number of nonzero elements per column is unknown).
    # After each chunk, the the gaps are removed. Finally, the results are saved into permanent files
    cdef long min_chunk_per_thread = long(1e3)
    cdef int num_threads = min(in_num_threads, max(1, num_columns // min_chunk_per_thread))
    cdef long chunk_size = min(long(5e5),num_columns)
    cdef int chunk_repeat_per_thread = min(10, max(1, chunk_size // (min_chunk_per_thread*num_threads)))
    cdef long prange_chunksize = max(chunk_size // (chunk_repeat_per_thread*num_threads), min_chunk_per_thread) # TODO: is this too complicated?
    cdef long num_chunks = num_columns // chunk_size  if (num_columns % chunk_size) == 0 else num_columns // chunk_size + 1
    cdef MATRIX_INDEX_TYPE_t chunk_index, results_start, results_end
    cdef MATRIX_INDEX_TYPE_t chunk_array_size = chunk_size*max_values_per_column
    cdef MATRIX_INDEX_TYPE_t start_column_index, end_column_index, column_index
    cdef MATRIX_INDEX_TYPE_t* chunk_inds = <MATRIX_INDEX_TYPE_t*> malloc(chunk_array_size * sizeof(MATRIX_INDEX_TYPE_t))
    cdef VALUE_TYPE_t* chunk_values =  <VALUE_TYPE_t*> malloc(chunk_array_size * sizeof(VALUE_TYPE_t))
    cdef int* chunk_num_results = <int*> malloc(chunk_size * sizeof(int))
    cdef _Tmp_To_Perm_Transcriber transformer
    cdef I_Lambda_Column* column_func = py_column_func.cpp_shared_ptr.get()

    for column_index in range(chunk_size):
        chunk_num_results[column_index] = 0
    for column_index in range(chunk_array_size):
        chunk_inds[column_index] = 0
    for column_index in range(chunk_array_size):
        chunk_values[column_index] = 0

    transformer = _Tmp_To_Perm_Transcriber(num_columns, max_values_per_column)
    if max_values_per_column == 0:
        # skip trivially empty matrix
        num_chunks = 0
    # move with parallel here, followed by master construct to keep the threads open? OMP 5.0 not supported by cython yet #------------------
    # do the threads open and close with each chunk?
    with nogil:
        for chunk_index in range(num_chunks):
            start_column_index = chunk_index * chunk_size
            end_column_index = min(num_columns, start_column_index + chunk_size)
            for column_index in prange(start_column_index, end_column_index, schedule='dynamic', chunksize=prange_chunksize, num_threads=num_threads):
                results_start = max_values_per_column * (column_index - start_column_index)
                column_func[0](column_index,&chunk_values[results_start], &chunk_inds[results_start],
                                &chunk_num_results[column_index - start_column_index])
            transformer.organize_chunk_results(chunk_size, chunk_inds, chunk_values, chunk_num_results)
    _logger.debug(f"Rewriting matrix {matrix_signature.matrix_name} to perm.")
    transformer.rewrite_to_perm(matrix_signature)
    free(chunk_num_results)
    free(chunk_inds)
    free(chunk_values)
    _logger.info(f"Finished creating matrix {matrix_signature.matrix_name}.")
