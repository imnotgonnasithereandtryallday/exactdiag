#pragma once
// defined in .h because cython generates .cpp files
#include <algorithm>
#include <vector>
#include <numbers>
#include <iostream>
#include <cstdlib>
#include <cstdarg>
#include <complex>
#include <concepts>
#include "../types_and_constants.h"
#include "../cpp_classes.h"
#include "../symmetry_utils.h"
#include "./symmetry_generator.h"
#include "./commutation_counter.h"
#include "./ndarray.h"
using std::vector;
using std::malloc; 
using std::free;
using std::complex;
using std::unique_ptr;
using std::make_unique;


// TODO: import VALUE_TYPE_t ?
//       pos_int type often specified too explicitly?
using VALUE_TYPE_t = complex<double>;



inline bool are_states_equal(const state_int* state1, const state_int* state2, size_t length) {
    return std::memcmp(state1, state2, length*sizeof(state_int)) == 0;
}


class Permutations: public I_Symmetry_Generator {
    /* Describes a group set{M_i^n, n in range(basis_length), i in range(num_shifts)}
       with elements that correspond to a subset of basis permutations.
       The order of application is fixed: i=0 is applied first, i=num_shifts-1 last.
       I. e., operations only shuffle state_ints and have period divisible by basis_length.
       The elements actually permute the basis and add an amplitude factor, therefore,
       even if the permutation of M_i^n == M_j^m for any j != i; n,m != 0, we keep both elements. 
       The operations also describe transformations of a state:
       transformed_state[P(k)+s] = state[k+s] for k in range(basis_length), s in range(internal_stride),
             and P in the subset of permutations. 
             Note: basis_length is also the number of 'distinct' nodes with respect to this permutation.
    */
    private:
    vector<int> shift_periodicities;
    size_t num_shifts;
    int basis_length;
    pos_int internal_stride;
    vector<pos_int> flattened_shifts_to_index;
    ndarray<int> index_to_shifts; // shape [basis_length, num_shift]
                                  // index_to_shifts[i,j] gives j-th element of the shift that transforms pos_int 0 to j.
    ndarray<pos_int> index_maps; // shape [num_shifts, basis_length, basis_length], 
                                 // index_maps[i,j,k] gives the basis index = M_i^j k
                                 // i. e., transformed basis index k.
    vector<float> quantum_numbers; // shape [num_shifts] // TODO: should be float or int?
    State_Hierarchy state_hierarchy;
    Commutation_Counter_Factory commutation_counter_factory;

    public:
    Permutations(size_t num_shifts, pos_int basis_length, pos_int internal_stride, 
                const ndarray<pos_int>& shifts_to_index, const ndarray<pos_int>& index_maps, 
                const vector<float>& quantum_numbers,
                const State_Hierarchy& state_hierarchy, const Commutation_Counter_Factory& commutation_counter_factory) 
        : shift_periodicities(num_shifts, basis_length)
        , num_shifts(num_shifts)
        , basis_length(basis_length)
        , internal_stride(internal_stride) // TODO: should this be used in more methods?
        , flattened_shifts_to_index(shifts_to_index.data)
        , index_to_shifts(vector<size_t>{static_cast<size_t>(basis_length), num_shifts})
        , index_maps(vector<size_t> {num_shifts, static_cast<size_t>(basis_length), static_cast<size_t>(basis_length)})
        , quantum_numbers(quantum_numbers)
        , state_hierarchy(state_hierarchy)
        , commutation_counter_factory(commutation_counter_factory)
        {
        /* shifts_to_index : pos_int array of shape [basis_length]*num_shifts // TODO: this should be possible to be reduced to shape=shift_periodicities if known in advance
                     shifts_to_index[i,j,...,k] gives the basis index correspoinding to shifts = [i,j,...,k].
                     basis index = M_0^i M_1^j ... M_(num_shifts-1)^k 0,
                     i. e., transformed basis index 0.
           index_maps: shape [num_shifts, basis_length], 
                       (note, one dimension less than self.index_maps)
                       index_maps[i,k] gives the basis index = M_i k
                       i. e., transformed basis index k.
        */
        // TODO: check that at least the shapes of the arrays match what is expected!
        // TODO: is this shift_periodicities initialization generally correct??

        for (pos_int i=0; i < basis_length; ++i) {
            vector<size_t> indices (shifts_to_index.argwhere_first(i));
            for (size_t j=0; j < num_shifts; ++j) {
                this->index_to_shifts[vector<size_t>{static_cast<size_t>(i), j}] = static_cast<pos_int>(indices[j]);
            }
        }
        for (size_t iop=0; iop < num_shifts; ++iop) {
            // std::memcpy(&this->index_maps[iop, 0, 0], index_maps.data.data() + iop*index_maps.shape[1], index_maps.shape[1] * sizeof(pos_int));
            for (size_t i=0; i < basis_length; ++i) { // TODO: check if the above works now -- with vector indexing?
                    this->index_maps[vector<size_t>{iop, 0, i}] = i;
            }
            
            for (size_t ipow=1; ipow < basis_length; ++ipow) {
                for (size_t i=0; i < basis_length; ++i) {
                    size_t j = this->index_maps[vector<size_t>{iop, ipow-1, i}];
                    this->index_maps[vector<size_t>{iop, ipow, i}] = index_maps[vector<size_t>{iop, j}];
                }
            }
            
            for (size_t ipow=1; ipow < basis_length; ++ipow) {
                if (this->index_maps[vector<size_t>{iop, ipow, 0}] == 0) {
                    shift_periodicities[iop] = ipow;
                    break;
                }
            }
        }
    }

    unique_ptr<I_Symmetry_Generator> clone() const override {
        return unique_ptr<I_Symmetry_Generator>(new Permutations(*this));
    }        
    
    const vector<int>& get_shift_periodicities() const override {
        return this->shift_periodicities;
    }
    size_t get_num_shifts() const override {
        return this->num_shifts;
    }
    int get_basis_length() const override {
        return this->basis_length;
    }

    void get_unit_shifts_from_index(int index, int* shifts) const override {
        // Return shifts such that index = M_0^i M_1^j ... M_(num_shifts-1)^k 0,
        // i. e., transform from 0 to index.
        // shifts may not be unique for a given index.
        // shifts are assumed to be pre-allocated to at least self.num_shifts elements.
        size_t i = static_cast<size_t>(python_mod(index, this->basis_length));
        memcpy(shifts, &this->index_to_shifts[std::vector<size_t>{i,0}], this->num_shifts * sizeof(int));
    }

    pos_int get_index_from_unit_shifts(const int* shifts) const override {
        // Return index that 0 is transformed into by prod_i M_i^shifts[i].
        int flattened_shift = 0;
        // int periodicity = 1;
        for (int i=0; i < this->num_shifts; ++i) {
            // flattened_shift *= this->shift_periodicities[i];  // this will be after the shape of shifts is reduced
            flattened_shift *= this->basis_length;
            flattened_shift += python_mod(shifts[i], this->shift_periodicities[i]);
        }
        return this->flattened_shifts_to_index[flattened_shift];
    }

    int get_symmetry_states_from_state(const state_int* state, bool check_lowest, int num_repeat, int external_stride, 
                                            State_Amplitude* states_amplitudes) const override { 
                                            // TODO: check if the algorithm can be improved.
        // Operations should be ordered from the slowest to the fastest
        // so that the slowest operation has to generate the fewest states.
        // Consider overriding.
        // states_amplitudes is assumed to be pre-allocated to at least this->max_num_states.
        // Returns n, only the first n elements of states_amplitudes are defined.
        int num_found = 1;
        State_Amplitude state_amplitude = State_Amplitude(this->basis_length, 0);

        states_amplitudes[0] = State_Amplitude(this->basis_length, state, 1);
        for (int iop=0; iop < this->num_shifts; ++iop) {
            vector<int> symmetry_indices = vector<int>(this->num_shifts, 0);
            int num_found_previous = num_found;

            // TODO: The work with previous_amplitudes can probably be optimized better
            vector<VALUE_TYPE_t> previous_amplitudes (num_found_previous);
            for (int istate=0; istate < num_found_previous; ++istate) {
                previous_amplitudes[istate] = states_amplitudes[istate].amplitude;
            }

            for (int ipow=1; ipow < this->shift_periodicities[iop]; ++ipow) { 
                symmetry_indices[iop] = ipow;
                for (int istate=0; istate < num_found_previous; ++istate) { // FIXME: is this necessary? can we use stride of each symmetry instead?
                    this->translate_by_symmetry(states_amplitudes[istate].state, symmetry_indices.data(), num_repeat, external_stride, state_amplitude); 
                    if (check_lowest) {
                        int ind_of_lower_state = this->state_hierarchy.check_lower_state(state, state_amplitude.state, this->basis_length);
                        if (ind_of_lower_state == 2) { 
                            // input state is not the lowest
                            return 0;
                        }
                    }

                    // TODO: I think i should check up to num_found_previous in both this and the following loop.
                    //      Here for early exit. In the following, 'previously found' are probably guaranteed to be hit before any added in this iop loop.
                    //      If check_lowest, i already know whether they are equal.
                    state_amplitude.amplitude *= previous_amplitudes[istate];
                    if (are_states_equal(state_amplitude.state, states_amplitudes[0].state, this->basis_length)
                        && abs(state_amplitude.amplitude - previous_amplitudes[0]) > 1e-8) { // TODO: does this work for arbitrary permutations?
                        // invalid initial state
                        return 0;
                    }
                    bool is_in = false;
                    for (int i=0; i<num_found; ++i) {
                        if (are_states_equal(states_amplitudes[i].state, state_amplitude.state, this->basis_length)) {
                            is_in = true;
                            states_amplitudes[i].amplitude += state_amplitude.amplitude;
                            break;
                        }
                    }
                    if (not is_in) {
                        states_amplitudes[num_found] = state_amplitude;
                        num_found += 1;
                    }
                }
            }
        }
        
        int skipped = 0;
        VALUE_TYPE_t norm = 0;
        for (int i=0; i<num_found; i++) {
            VALUE_TYPE_t amplitude = states_amplitudes[i].amplitude;
            if (abs(amplitude) < 1e-8) {
                skipped += 1;
                continue;
            }
            states_amplitudes[i-skipped] = states_amplitudes[i];
            norm += amplitude*conj(amplitude);
        }
        for (int i=0; i<num_found-skipped; ++i) {    
            states_amplitudes[i].amplitude /= sqrt(norm);
        }
        return num_found-skipped;
    }

    void translate_by_symmetry(const state_int* state, const int* symmetry_indices, int num_repeat, int external_stride,
                                    State_Amplitude& state_amplitude) const override { // TODO: check if the algorithm can be improved.
        // Consider overriding.
        // Populate state_amplitude with prod_i M_i^shifts[i] state.
        int* shifts = (int*) malloc(sizeof(int) * this->num_shifts); // TODO: is this called in a hot loop? stack allocator?
        memcpy(shifts, symmetry_indices, this->num_shifts*sizeof(int)); // TODO: these are fragile to change of type! // FIXME: Seems i dont need shifts at all. use cymmetry indices instead?
        int stride = this->internal_stride * external_stride;
        Commutation_Counter commutation_counter = this->commutation_counter_factory.get(this->basis_length, stride);
        for (int ir=0; ir<num_repeat; ++ir) {
            int offset = ir * this->basis_length * stride;
            for (int i=0; i < this->basis_length; ++i) {
                int shifted_index = this->get_index_by_relative_shift(i, shifts);
                for (int ip=0; ip<stride; ++ip) {
                    state_amplitude.state[offset + shifted_index*stride + ip] = state[offset + i*stride + ip];
                }
                commutation_counter.add(offset + shifted_index*stride, &state[offset + i*stride]);
            }
        }                
        state_amplitude.amplitude = 1 - 2*commutation_counter.commutation_parity;

        complex<double> phase = 0;
        for (int j=0; j < this->num_shifts; ++j) {
            phase += symmetry_indices[j] * this->quantum_numbers[j] / static_cast<double>(this->shift_periodicities[j]);
            // TODO: does this work for non-orthogonal elementary translations? for non-orthogonal periodic boundary?
        } // TODO: do we need to keep quantum_numbers and shift_periodicities separately?
        const complex<double> imag_2pi(0.0, 2.0 * std::numbers::pi);
        state_amplitude.amplitude *= exp(-imag_2pi*phase);
        free(shifts);
    }
};