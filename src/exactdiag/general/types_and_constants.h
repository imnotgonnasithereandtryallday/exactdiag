#pragma once
#include <array>
#include <complex>
#include <limits>
#include <string>

// NOTE: make sure these match those in the .pyx file

// state stuff
typedef short state_int;
typedef short pos_int; // used also for relative quantitites -- cannot be unsigned. # TODO: introduce Option type

// FIXME: unify, to enum -- state hierarchy then gets this enum instead of vector of values
const state_int HOLE_VALUE = 0;
const state_int SPIN_DOWN_VALUE = 1;
const state_int SPIN_UP_VALUE = 2;

const state_int OCCUPIED_VALUE = 1;
const state_int EMPTY_VALUE = 0;


// matrix stuff
typedef long long MATRIX_INDEX_TYPE; // FIXME: cannot be unsigned because basis indexing uses -1 on invalid input
typedef unsigned VALUE_INDEX_TYPE;
typedef std::complex<double> VALUE_TYPE;
const VALUE_INDEX_TYPE MAX_NUM_UNIQUIE_VALUES = std::numeric_limits<VALUE_INDEX_TYPE>::max();
const std::array<std::string, 2> SYMMETRY_OPTIONS {"hermitian", "sorted"};
