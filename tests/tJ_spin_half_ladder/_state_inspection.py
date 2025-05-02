from itertools import product
import pathlib

import numpy as np

from exactdiag.general.cpp_classes import Py_State_Amplitude
from exactdiag.tJ_spin_half_ladder.configs import Hamiltonian_Config
from tests.tJ_spin_half_ladder.test_basis_size import Precalculated

class Inspector:
    def __init__(self, config: Hamiltonian_Config):
        self.config = config
        self.state_translator, self.basis_map, self.symmetries = config.get_translators()

    @classmethod
    def from_precalculated(cls, path: str | pathlib.Path):
        config = Precalculated.load(path).to_hamiltonian_config()
        return cls(config)
    
    def get_state_from_sparse(self, sparse_index):
        return self.state_translator.sparse_index_to_state(sparse_index)

    def get_state_from_dense(self, dense_index):
        return self.get_state_from_sparse(self.basis_map.get_sparse(dense_index))

    def print_states_amplitudes(self, state: list[int]):
        k_ranges = [range(i) for i in self.symmetries.get_shift_periodicities()]
        state = np.array(state, dtype=np.int16) # TODO: import pos_int
        state_amplitude = Py_State_Amplitude(state, 1)
        ks = np.empty(self.symmetries.get_num_shifts(), dtype=np.int32)
        for ks[:] in product(*k_ranges):
            self.symmetries.translate_by_symmetry(state, ks, 1, 1, state_amplitude)
            print(ks, state_amplitude.state, state_amplitude.amplitude)

    def get_states_amplitudes(self, state: list[int]):
        k_ranges = [range(i) for i in self.symmetries.get_shift_periodicities()]
        state = np.array(state, dtype=np.int16) # TODO: import pos_int
        state_amplitude = Py_State_Amplitude(state, 1)
        ks = np.empty(self.symmetries.get_num_shifts(), dtype=np.int32)
        dic = {}
        for ks[:] in product(*k_ranges):
            self.symmetries.translate_by_symmetry(state, ks, 1, 1, state_amplitude)
            dic[tuple(ks)] = (state_amplitude.state, state_amplitude.amplitude)
        return dic