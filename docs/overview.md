
# Overview
In this section, we briefly touch on some of the terminology used throughout the package.


## Basis states and symmetries
We define a basis as a ordered set of states complete with respect to a Hamiltonian.
We start by assuming there is a cheap way of mapping between basis state's ordinal and 
a short ordered list of small integers. If it is possible to order the states in
a way that block-diagonalizes the Hamiltonian, we should do so.

We call the list and its ordinal the `state` and `sparse index`, respectively.
When applicable, we split the sparse index into several components (e.g. hole positions vs spin configuration on the remaining positions).

Given a [representation](https://en.wikipedia.org/wiki/Group_representation) of a symmetry group 
(that commutes with the symmetries conserved in the space spanned by the sparse basis) of the Hamiltonian, 
we define another basis on the representation-invariant subspace, called the `dense basis`.
We refer to the ordinals of these basis states as `dense index`.

In our construction, each spase basis state has non-zero projection into at most one dense basis state
<!-- (Would this s till be the case with a non-commutative group?) -->.
Therefore, we can choose the state with the `lowest sparse index` to represent the dense basis state.
When expressing a dense basis state as a linear combination of sparse states, 
the lowest sparse state has an amplitude of one (up to normalization), the other amplitudes are generally complex.

We consider only commutative groups, because I did not have the time for non-commutative groups.

The relations between sparse index, its components, and the corresponding state, and their lowest variants with amplitude are delegated to the `State_Index_Amplitude_Translator`.

For more details, see [Appendix B 1, 2 of my dissertation](https://is.muni.cz/th/zy2au/Adamus_thesis.pdf).


## Sparse matrix format
Operators with only short-range interactions tend to be sparse, i.e., most of
the matrix elements are zero, and the number of nonzero elements in each row scales
linearly with system size. Further, there are only very few unique values the matrix elements can take.

Thus, the chosen format for a matrix of `M` columns with `D` non-zero elements is:
- A list (of whatever length is necessary) of unique (complex) values of the matrix
elements. We call this simply `values`.
- A list of length `D` containing the `value-indices` corresponding to the values of
the matrix elements.
- A list of length `D` containing the row-indices of the matrix elements. We call this list `minor indices`.
- A list of length `M+1` where `i`-th and `i+1`-th elements give pointers to
(or the indices of) the beginning and the end of the `i`-th column in the two above
lists, respectively. We call this list `major pointers`.

In particular, the first and the last elements in the last list are `0` and `D`, respectively;
and the number of nonzero matrix elements in column `i` is given by the difference
of the list elements `i+1` and `i`.
We have described the column-major format, but row-major is also supported, 
as well as triangular form for Hermitian matrices.

The matrices are memory-mapped onto disk. Use SSDs if possible.


## Operator creation
There are some operators defined. Creation of custom operators from python is currently not possible.
