cimport numpy as np

from typing import Protocol

from scipy.linalg import norm as scipy_norm
from jaxtyping import Array, Num, Complex, Float
from cython.parallel import prange
import numpy as np

# TODO: Rename so that it is next to the non-cython file.

class HermiteanMatrix(Protocol):
    def dot(self, vector: Num[Array, "n"]) -> Num[Array, "n"]: ...

class Matrix(Protocol):
    def dot(self, vector: Num[Array, "m"]) -> Num[Array, "n"]: ...

def lanczos_dynamical_response(
    const double complex[:] initial_state: Complex[Array, "m"], 
    excitation_operator: Matrix, 
    excited_state_hamiltonian: HermiteanMatrix, 
    const double complex[:] zs: Complex[Array, 'w'],
    int num_lanczos_vecs, 
    int num_threads,
    ) -> Float[Array, 'w']: 
    """Calculate the spectrum of an operator applied to a state."""
    # follows notation of Dagotto 1994 Rev. Mod. Phys., vol. 66 no. 3, pg. 776
    cdef long i, length = len(zs), chunksize = max(1,length // (num_threads*5))
    cdef double[:] ans, bns, spectrum
    excited_state = excitation_operator.dot(initial_state)   
    norm = scipy_norm(excited_state)
    spectrum = np.zeros(len(zs), dtype=np.double)
    if norm < 1e-8:
        return spectrum
    excited_state = excited_state / norm
    ans, bns = _lanczos_get_anbn(excited_state, num_lanczos_vecs, excited_state_hamiltonian)  
    cdef double factor = -1.0 / np.pi * norm**2
    for i in prange(length,nogil=True,schedule='dynamic',chunksize=chunksize,num_threads=num_threads):
        spectrum[i] = factor * _spectral_value_at_z(ans, bns, zs[i])
    return spectrum

cpdef double _spectral_value_at_z(double[:] ans, double[:] bns, double  complex z) nogil:
    cdef int length = len(ans)
    cdef int i
    cdef double complex det = z - ans[length-1]
    for i in range(1,length):
        det = z - ans[length-1-i] - bns[length-i]**2 / det
    return (1.0/det).imag

def _lanczos_get_anbn(v0, num_lanczos_vecs, hamiltonian):
    vn = v0 / scipy_norm(v0)
    bn = 0
    # an, bn are real if used with hermitean hamiltonian.
    ans = np.zeros(num_lanczos_vecs, dtype=np.double)
    bns = np.zeros(num_lanczos_vecs+1, dtype=np.double)
    vnm = np.zeros(len(vn), dtype=np.cdouble)
    vnp = np.zeros(len(vn), dtype=np.cdouble)
    num_nonzero_elements = 0
    for i in range(num_lanczos_vecs):
        vnp[:], ans[i], bns[i+1] = _lanczos_vec_iteration(vn, vnm, bns[i], hamiltonian)
        num_nonzero_elements += 1      
        if bns[i+1] < 1e-8: # new vector is zero
           break
        vnm[:] = vn  # TODO: could juggle pointers instead of copying values
        vn[:] = vnp
    return ans[:num_nonzero_elements], bns[:num_nonzero_elements]

def _lanczos_vec_iteration(vn, vnm, bn, hamiltonian):
    # gets v_n, v_{n-1} and b_n
    # returns a_n and b_{n+1}
    # all in/out vectors are normalized -> bn is not squared in vnp = ... -bn*vnm
    hvn = hamiltonian.dot(vn)
    an = np.real(np.vdot(vn, hvn)) # explicitly stated real to suppress warning
    vnp = hvn - an*vn - bn*vnm 
    bnp = scipy_norm(vnp)  
    vnp /= bnp
    return vnp,an,bnp
