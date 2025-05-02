from libcpp.memory cimport shared_ptr

from exactdiag.general.symmetry cimport I_State_Index_Amplitude_Translator
from exactdiag.general.group.symmetry_generator cimport I_Symmetry_Generator
from exactdiag.general.basis_indexing cimport Basis_Index_Map
from exactdiag.general.column_functions cimport I_Lambda_Column

cdef class Wrapped_Column_Func:
    cdef shared_ptr[I_Lambda_Column] cpp_shared_ptr;