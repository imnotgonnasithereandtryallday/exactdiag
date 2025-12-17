#pragma once
// defined in .h because cython generates .cpp files
// The _ladder suffix is because the build system has some problem with correctly identifying headers locations.
#include <cstring>
#include <vector>
#include <memory>
#include <numeric>
#include <numbers>
#include "../general/types_and_constants.h"
#include "../general/symmetry.h"
#include "../general/group/symmetry_generator.h"
#include "../general/group/commutation_counter.h"
#include "../general/group/cluster.h"
#include "../general/symmetry_utils.h"
#include <iostream>
#include <iomanip>

using std::vector;
using std::unique_ptr;

class Symmetries: public Permutations {
    private:
    State_Hierarchy state_values_hierarchy;
    State_Hierarchy spin_values_hierarchy;
    Commutation_Counter_Factory commutation_counter_factory;

    public:
    Symmetries(int num_rungs, int num_legs, const vector<float>& quantum_numbers)
        : state_values_hierarchy {HOLE_VALUE, SPIN_DOWN_VALUE, SPIN_UP_VALUE} // TODO: this should be specified elsewhere!
        , spin_values_hierarchy {SPIN_DOWN_VALUE, SPIN_UP_VALUE}
        , commutation_counter_factory {SPIN_DOWN_VALUE, SPIN_UP_VALUE}
        , Permutations(get_rectangular_cluster(
            num_rungs,
            num_legs, 
            quantum_numbers,
            State_Hierarchy{HOLE_VALUE, SPIN_DOWN_VALUE, SPIN_UP_VALUE},
            Commutation_Counter_Factory{SPIN_DOWN_VALUE, SPIN_UP_VALUE}
            // this->state_values_hierarchy, // TODO: can i force this to run after this->state_values_hierarchy is initialized?
            // this->commutation_counter_factory
          ))
    {
        // assumes hamiltonian with one type of rung and one type of leg on a periodic cluster.
        // TODO: generalize.
        // spin degrees are treated with a value        
    }

    Symmetries(const Symmetries& other)
        : state_values_hierarchy(other.state_values_hierarchy)
        , spin_values_hierarchy(other.spin_values_hierarchy)
        , commutation_counter_factory(other.commutation_counter_factory)
        , Permutations(other)
    {}
};

class Symmetries_Single_Spin_Half_2leg_Ladder: public I_Symmetry_Generator { // FIXME: rename. single_spin_half_2leg_ladder should be a folder.
    // A specialized class with fast implementation of get_symmetry_states_from_state.
    // Uses translation symmetry along the legs and mirror symmetry along axis in between the legs.
    // TODO: Generalizable to clusters with point symmetry of the lattice and natural node indexing?
    private:
    pos_int num_nodes;
    vector<int> symmetry_qs;
    vector<int> symmetry_periodicities;
    const State_Hierarchy state_hierarchy {HOLE_VALUE, SPIN_DOWN_VALUE, SPIN_UP_VALUE}; // TODO: should this be specified elsewhere?

    public:
    Symmetries_Single_Spin_Half_2leg_Ladder(pos_int num_rungs, const vector<int>& quantum_numbers)
        : num_nodes(2 * num_rungs) 
        , symmetry_qs(quantum_numbers)
        , symmetry_periodicities{num_rungs, 2}
    {}


    unique_ptr<I_Symmetry_Generator> clone() const override {
        return unique_ptr<I_Symmetry_Generator>(new Symmetries_Single_Spin_Half_2leg_Ladder(*this));
    }

    const vector<int>& get_shift_periodicities() const override {
        return this->symmetry_periodicities;
    }

    size_t get_num_shifts() const override {
        return 2;
    }

    int get_basis_length() const override {
        return this->num_nodes;
    }
    pos_int get_index_by_relative_index(int initial_index, int relative_index) const override {
        // TODO: Profile how much the python_mods cost -- should we require inputs within bounds from the caller?
        initial_index = python_mod(initial_index, this->num_nodes);
        relative_index = python_mod(relative_index, this->num_nodes);
        pos_int leg = initial_index / 2 + relative_index / 2;
        pos_int rung = (initial_index + relative_index) % 2;
        return python_mod(2*leg + rung, this->num_nodes);
    }
    pos_int get_index_by_relative_shift(int initial_index, const int* relative_shift) const override {
        initial_index = python_mod(initial_index, this->num_nodes);
        pos_int leg = initial_index / 2 + relative_shift[0];
        pos_int rung = python_mod(initial_index + (relative_shift[1] % 2), 2);
        return python_mod(2*leg + rung, this->num_nodes);
    }

    void get_unit_shifts_from_index(int index, int* shifts) const override {
        index = python_mod(index, this->num_nodes);
        shifts[0] = index / 2;
        shifts[1] = index % 2;
        // TODO: does something like std::tie(shifts[0], shifts[1]) = std::div(index, 2); work?
    }
    pos_int get_index_from_unit_shifts(const int* shifts) const override {
        return 2 * python_mod(shifts[0], this->num_nodes / 2) + python_mod(shifts[1], 2);
    }

    int get_symmetry_states_from_state(const state_int* state, bool check_lowest, int num_repeat, int stride, State_Amplitude* states_amplitudes) const override {
        // The optimalization of this method is critical for calculation of matrix elements.
        // It is therefore implemented specifically for ladders avoiding the more general translate_by_symmetry of permutations.
        // NOTE: We simply ignore the num_repeat and stride arguments.

        // FIXME: Do not ignore the arguments.
        // TODO: Explain the approach here.
        
        pos_int num_nodes = this->get_basis_length();
        state_int* repeated_state = (state_int*) malloc(2*num_nodes * sizeof(state_int));
        state_int* translated_state;

        bool num_spins_less_one_odd = true;
        pos_int num_paired_rungs = 0;
        for (pos_int node_index=0; node_index<num_nodes; ++node_index) {
            state_int node_value = state[node_index];
            num_paired_rungs += (node_index%2 == 0)
                                && (node_value != HOLE_VALUE)
                                && (state[node_index+1] != HOLE_VALUE);
            num_spins_less_one_odd = num_spins_less_one_odd ^ (node_value != HOLE_VALUE); // ^ is XOR
        }
        VALUE_TYPE_t iTwoPi (0, 2 * std::numbers::pi);
        int mirror_anticommutation_sign = num_paired_rungs % 2 == 0 ? 1 : -1;
        VALUE_TYPE_t mirror_exp = exp(-iTwoPi * static_cast<VALUE_TYPE_t>(this->symmetry_qs[1] / double(this->symmetry_periodicities[1])));
        VALUE_TYPE_t mirror_fac = mirror_exp * static_cast<VALUE_TYPE_t>(mirror_anticommutation_sign);
        VALUE_TYPE_t rung_exp = exp(-iTwoPi * static_cast<VALUE_TYPE_t>(this->symmetry_qs[0] / double(this->symmetry_periodicities[0])));
        VALUE_TYPE_t rung_translated_amplitude = 1;
        bool is_mirror_symmetric = false;

        states_amplitudes[0] = State_Amplitude(num_nodes, state, 1);

        memcpy(repeated_state, state, num_nodes * sizeof(state_int));
        memcpy(repeated_state + num_nodes, state, num_nodes * sizeof(state_int));
        // this repeated pattern is only useable for a single ladder
        state_int* mirror_repeated_state = (state_int*) malloc(2*num_nodes * sizeof(state_int));
        for (pos_int node_index=0; node_index< num_nodes; node_index +=2) {
            mirror_repeated_state[node_index] = state[node_index+1];
            mirror_repeated_state[node_index+1] = state[node_index];
        }
        memcpy(mirror_repeated_state + num_nodes, mirror_repeated_state, num_nodes * sizeof(state_int));

        int num_found = 1;
        int max_num_states = num_nodes;
        Product_Generator product_generator = Product_Generator(this->symmetry_periodicities.data(), this->get_num_shifts(), 1);
        for (int product_index=1; product_index<max_num_states; ++product_index) {
            vector<int> symmetry_indices = product_generator.next();
            VALUE_TYPE_t amplitude;
            if (product_index % this->symmetry_periodicities[1] == 1) {
                if (is_mirror_symmetric) {
                    continue;
                }
                translated_state = &mirror_repeated_state[num_nodes - 2*symmetry_indices[0]];
                amplitude = rung_translated_amplitude * mirror_fac;
            } else {
                translated_state = &repeated_state[num_nodes - 2*symmetry_indices[0]];
                rung_translated_amplitude *= rung_exp;
                if (num_spins_less_one_odd 
                    && ((translated_state[0] == HOLE_VALUE) ^ (translated_state[1] == HOLE_VALUE))) 
                {
                    rung_translated_amplitude *= -1;
                }
                amplitude = rung_translated_amplitude;
            }

            if (check_lowest) {
                int ind_of_lower_state = this->state_hierarchy.check_lower_state(state, translated_state, num_nodes);
                if (ind_of_lower_state == 2) { // input state is not the lowest
                    free(repeated_state);
                    free(mirror_repeated_state);
                    return 0;
                }
            }

            // we take the initial state, find the mirror state
            // then shift the initial state by one rung and find the mirror state again
            // then shift the initial state by two rungs... 
            // (specific for the rung translation and mirror symmetries):
            // the first (if any) repeated state found can be the second state (mirror shifted input state)
            // or any rung shift that divides periodicity. 
            // it will always be equal to either the first (input) state or the second (mirror-shifted) state
            // if the repeated state was after rung shift, we have found all rung-shifted states.
            double eps = 1e-8; // FIXME: This threshold works for VALUE_TYPE_t of complex double, but
                               // we would get above this threshold if we had float instead of double in e.g. the rung_exp denominator.
            if (symmetry_indices[0] < this->symmetry_periodicities[0]/2 + 1) {
                 // TODO: we could compute highest factor on class initialization.
                if (are_states_equal(states_amplitudes[0].state, translated_state, num_nodes)) {
                    if (abs(amplitude - VALUE_TYPE_t(1, 0)) > eps) {
                        // invalid initial state
                        free(repeated_state);
                        free(mirror_repeated_state);
                        return 0;
                    }
                    if (product_index == 1) {
                        // the state is mirror-symmetric, we can skip the mirror symmetries in the product
                        is_mirror_symmetric = true;
                        continue;
                    } else {
                        // all states found
                        break;
                    }
                }
                if ((num_found != 1) && are_states_equal(states_amplitudes[1].state, translated_state, num_nodes)) {
                    if (abs(amplitude - states_amplitudes[1].amplitude) > eps) {
                        // invalid initial state
                        free(repeated_state);
                        free(mirror_repeated_state);
                        return 0;
                    }
                    // all states found
                    break;
                }
            }
            states_amplitudes[num_found] = State_Amplitude(num_nodes, translated_state, amplitude);
            num_found += 1;
        }

        for (int i=0; i<num_found; ++i) {
            states_amplitudes[i].amplitude /= sqrt(num_found);
        }
        free(repeated_state);
        free(mirror_repeated_state);
        return num_found;
    }

    void translate_by_symmetry(const state_int* state, const int* symmetry_indices, int num_repeat, int stride, State_Amplitude& state_amplitude) const override {
        // TODO: explain the approach
        // TODO: note that symmetry is modulo periodicity -- if the initial state is incompatible, that may lead to a different amplitude.

        const pos_int num_rungs = this->num_nodes / 2;
        const int dx = python_mod(symmetry_indices[0], num_rungs);
        const int dy = python_mod(symmetry_indices[1], 2);
        int num_paired_rungs = 0;
        int num_traveled_spins = 0;
        int num_spins_less_one = -1;
        
        for (pos_int i=0; i<this->num_nodes; ++i) {
            state_int node_value = state[i]; 
            num_paired_rungs += (i%2 == 0)
                                && (node_value != HOLE_VALUE)
                                && (state[i+1] != HOLE_VALUE);
            int translated_index = this->get_index_by_relative_shift(i, symmetry_indices);
            state_amplitude.state[translated_index] = node_value;
            if (node_value != HOLE_VALUE) { 
                num_spins_less_one += 1;
                if (i/2 + dx > num_rungs) {
                    num_traveled_spins += 1;
                }
            }
        }

        VALUE_TYPE_t iTwoPi (0, 2 * std::numbers::pi);
        VALUE_TYPE_t anticommutation_sign = (num_spins_less_one*num_traveled_spins + num_paired_rungs*dy) % 2 == 0 ? 1 : -1; 
        VALUE_TYPE_t qr = dx * this->symmetry_qs[0] / double(num_rungs)
                         + 0.5 * dy * this->symmetry_qs[1];
        state_amplitude.amplitude = anticommutation_sign *  exp(-iTwoPi * qr);
    }
};

class State_Index_Amplitude_Translator: public I_State_Index_Amplitude_Translator {
    private:
    pos_int num_holes;
    pos_int num_down_spins;
    int num_spin_states;
    ndarray<MATRIX_INDEX_TYPE> combinatorics_table; // 2D
    std::shared_ptr<I_Symmetry_Generator> symmetries;

    public:
    State_Index_Amplitude_Translator (
        pos_int num_holes, pos_int num_down_spins, int num_spin_states, 
        const ndarray<MATRIX_INDEX_TYPE>& combinatorics_table, const std::shared_ptr<I_Symmetry_Generator>& symmetries
    ) 
        : num_holes(num_holes)
        , num_down_spins(num_down_spins) // TODO: conserved number of spins should also be in the symmetries class?
        , num_spin_states(num_spin_states) // TODO: remove. this is the same as num_sparse_minor_states!
        , combinatorics_table(combinatorics_table)
        , symmetries(symmetries)
    {}
    State_Index_Amplitude_Translator (
        pos_int num_holes, pos_int num_down_spins, int num_spin_states, 
        const vector<vector<MATRIX_INDEX_TYPE>>& combinatorics_table, const std::shared_ptr<I_Symmetry_Generator>& symmetries
    ) 
        : num_holes(num_holes)
        , num_down_spins(num_down_spins)
        , num_spin_states(num_spin_states)
        , symmetries(symmetries)
        , combinatorics_table(vector<size_t>{combinatorics_table.size(), combinatorics_table[0].size()})
    {   
        size_t allocated = 0;
        for (const auto& row : combinatorics_table) {
            memcpy(&this->combinatorics_table[allocated], row.data(), row.size() * sizeof(MATRIX_INDEX_TYPE));
            allocated += row.size();
        }
    }

    std::shared_ptr<I_Symmetry_Generator> get_symmetries() const override {
        return this->symmetries;
    }
    size_t get_num_minor_sparse_states() const {
        return this->num_spin_states;
    }

    vector<state_int> sparse_index_to_state(MATRIX_INDEX_TYPE sparse_index) const {
        int remaining_holes = this->num_holes;
        int remaining_spins = this->num_down_spins;
        MATRIX_INDEX_TYPE remaining_hole_ind = sparse_index / this->num_spin_states;
        MATRIX_INDEX_TYPE remaining_spin_ind = sparse_index - remaining_hole_ind*this->num_spin_states;
        pos_int num_nodes = this->symmetries->get_basis_length();
        vector<state_int> state (num_nodes);

        for (pos_int j=num_nodes-1; j >= 0; --j) { // NOTE: this requires pos_int to be signed, otherwise j >= 0 is always true
            if (remaining_holes > 0) {
                MATRIX_INDEX_TYPE allocation_cost = this->combinatorics_table[vector<size_t>{static_cast<size_t>(remaining_holes), static_cast<size_t>(j)}];
                if (remaining_hole_ind >= allocation_cost) {
                    state[j] = HOLE_VALUE; // TODO: this should be generalized using hierarchy, but symmetry does not expose it currently
                    remaining_holes -= 1;
                    remaining_hole_ind -= allocation_cost;
                    continue;
                }
            }
            if (remaining_spins > 0) {
                MATRIX_INDEX_TYPE allocation_cost = this->combinatorics_table[vector<size_t>{static_cast<size_t>(remaining_spins), static_cast<size_t>(j-remaining_holes)}];
                if (remaining_spin_ind >= allocation_cost) {   
                    state[j] = SPIN_DOWN_VALUE;
                    remaining_spins -= 1;
                    remaining_spin_ind -= allocation_cost;
                    continue;
                }
            }
            state[j] = SPIN_UP_VALUE;
        }
        return state;
    }

    MATRIX_INDEX_TYPE state_to_sparse_index( const state_int* state) const {
        MATRIX_INDEX_TYPE hole_index = 0;
        MATRIX_INDEX_TYPE spin_index = 0;
        int allocated_holes = int(state[0] == HOLE_VALUE);
        int allocated_down_spins = int(state[0] == SPIN_DOWN_VALUE);
        for (pos_int i=1; i < this->symmetries->get_basis_length(); ++i) {
            if (state[i] == HOLE_VALUE) { // TODO: this can probably be done with state hierarchy aswell
                allocated_holes += 1;
                hole_index += this->combinatorics_table[vector<size_t>{static_cast<size_t>(allocated_holes), static_cast<size_t>(i)}];
            } else if (state[i] == SPIN_DOWN_VALUE) {
                allocated_down_spins += 1;
                spin_index += this->combinatorics_table[vector<size_t>{static_cast<size_t>(allocated_down_spins), static_cast<size_t>(i-allocated_holes)}];
            }
        }
        return spin_index + hole_index*this->num_spin_states;
    }

    MATRIX_INDEX_TYPE check_major_index_supports_lowest_states(MATRIX_INDEX_TYPE hole_index) const {
        // Hole index can support lowest states if it can not be lowered by applying symmetries.
        // Since only the holes affect the anticommutation sign, its compatibility is also checked.
        // Returns hole_index if the hole_index can support, -1 otherwise
        // TODO: this could be one of three private methods -- one that chcecks with (anti-)commutations and one that does not.
        pos_int num_nodes = this->symmetries->get_basis_length();
        if (this->num_holes <= 1 or this->num_holes >= num_nodes-1) {
            return (hole_index == 0) ? 0 : -1;
        }
        vector<pos_int> hole_positions (num_holes);
        this->find_hole_positions(hole_index, hole_positions);
        vector<state_int> state (num_nodes, SPIN_UP_VALUE);
        for (pos_int hole_pos : hole_positions) {
            state[hole_pos] = HOLE_VALUE;
        }
        // adding a single flipped spin avoids spin system being accidentally more symmetric
        pos_int first = find_first(state, SPIN_UP_VALUE);
        if (first != -1) {
            state[first] = SPIN_DOWN_VALUE;
        }
        vector<State_Amplitude> states_amplitudes (num_nodes, State_Amplitude());
        pos_int num_states = this->symmetries->get_symmetry_states_from_state(
            state.data(), true, states_amplitudes.data()
        );
        return (num_states > 0) ? hole_index : -1;
    }

    void find_hole_positions(MATRIX_INDEX_TYPE hole_index, vector<pos_int>& hole_positions) const {
        int remaining_holes = this->num_holes;
        MATRIX_INDEX_TYPE temp_hole_index = hole_index;

        for (pos_int j=this->symmetries->get_basis_length()-1; j >= 0; --j) {  // FIXME: this requires pos_int to be signed, otherwise j >= 0 is always true
            MATRIX_INDEX_TYPE allocation_cost = this->combinatorics_table[vector<size_t>{static_cast<size_t>(remaining_holes), static_cast<size_t>(j)}];
            if (temp_hole_index >= allocation_cost) {
                hole_positions[remaining_holes-1] = j;
                remaining_holes -= 1;
                temp_hole_index -= allocation_cost;
                if (remaining_holes == 0) {
                    return;
                }
            }
        }
    }
};
