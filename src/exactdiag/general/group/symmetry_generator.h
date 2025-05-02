#pragma once
// defined in .h because cython generates .cpp files
#include <vector>
#include <numbers>
#include <iostream>
#include <cstdlib>
#include <complex>
#include <concepts>
#include "../types_and_constants.h"
#include "../cpp_classes.h"
#include "../symmetry_utils.h"
using std::vector;
using std::malloc; 
using std::free;
using std::complex;
using std::unique_ptr;
using std::make_unique;
using std::move;

// TODO: complex<double> -> VALUE_TYPE_t ?
//       add const to methods where aplicable
//       add local variable generator to reduce the number of indirections
class I_Symmetry_Generator {
    public:
        virtual ~I_Symmetry_Generator() = default; // if a class contains a virtual method it should contain a virtual destructor as well
        // C++ does not allow virtual copy constructors, but we require one.
        // Is there a better way to indicate this?
        virtual unique_ptr<I_Symmetry_Generator> clone() const = 0; // TODO: Is there a way to do this in general using type of this and copy constructor?

        virtual const vector<int>& get_shift_periodicities() const = 0; // TODO: what should be int ad what pos_int?
        // TODO: can we specify that the lifetime of the periodicities reference is at least the lifetime of the I_Symmetry_Generator instance? Do we want to?
        // TODO: can we hide get_shift_periodicities? is vector too implementation-specific?
        virtual size_t get_num_shifts() const = 0;
        virtual int get_basis_length() const = 0;

        virtual pos_int get_index_by_relative_index(int initial_index, int relative_index) const {
            // Return initial_index transformed by operation that maps 0 to relative_index.
            // TODO: This probably does not have a unique return value in general.
            int* relative_shift = (int*) malloc(sizeof(int) * this->get_num_shifts()); 
            // TODO: is this called in a hot loop? We could make it stack-allocated by templating the number of shifts.
            this->get_unit_shifts_from_index(relative_index, relative_shift);
            pos_int index = this->get_index_by_relative_shift(initial_index, relative_shift);
            free(relative_shift);
            return index;
        }

        virtual pos_int get_index_by_relative_shift(int initial_index, const int* relative_shift) const {
            // Return initial_index transformed by prod_i M_i^relative_shift[i].
            size_t num_shifts = this->get_num_shifts();
            int* shift = (int*) malloc(sizeof(int) * num_shifts); // TODO: is this called in a hot loop?
            this->get_unit_shifts_from_index(initial_index, shift);
            const vector<int>& shift_periodicities = this->get_shift_periodicities();
            for (int i=0; i < num_shifts; ++i) {
                shift[i] = python_mod(shift[i] + relative_shift[i], shift_periodicities[i]);
            }
            pos_int index = this->get_index_from_unit_shifts(shift);
            free(shift);
            return index;
        }
        virtual void get_unit_shifts_from_index(int index, int* shifts) const= 0;
        virtual pos_int get_index_from_unit_shifts(const int* shifts) const = 0;
        // virtual size_t get_flat_shift_index(const int* shifts) const = 0; // ------ is this needed?
        // FIXME: populate instead of get where there is no return value?

        // FIXME: overloaded methods on an interface do not work:
        // if one is overriden in derived class, all of the inherited overloads are hidden 
        // and cannot be called directly from an instace of the derived class.
        int get_symmetry_states_from_state(const state_int* state, bool check_lowest, State_Amplitude* states_amplitudes) const {
            return this->get_symmetry_states_from_state(state, check_lowest, 1, 1, states_amplitudes); 
        }
        virtual int get_symmetry_states_from_state(const state_int* state, bool check_lowest, int num_repeat, int stride, State_Amplitude* states_amplitudes) const = 0;
        
        void translate_by_symmetry(const state_int* state, const int* symmetry_indices, State_Amplitude& state_amplitude) const {
            this->translate_by_symmetry(state, symmetry_indices, 1, 1, state_amplitude);
        }
        virtual void translate_by_symmetry(const state_int* state, const int* symmetry_indices, int num_repeat, int stride, State_Amplitude& state_amplitude) const = 0;
        // # TODO: describe the functions and their arguments!
        //     # ---------------- mam externi stride -- pokud budu mit usporadani:
        //     #   x=0,y=0,s=0
        //     #   x=0,y=0,s=1
        //     #   x=0,y=1,s=0
        //     #   ...
        //     #   x=1,y=0,s=0
        //     #   ...
        //     # chcu mit v translacich externi stride (2 spiny) a internni stride (dimenze y)
        //     # repeat je pak naopak pri poradi s=0,x=0,y=0    
        complex<double> get_q_dot_r(pos_int index_of_node_at_r, const vector<int>& symmetry_qs) const {
            // # symmetry_qs: the integer quantum numbers specifying the symmetries
            // # --------------------- should this be here? uses many properties of symmetries, but technically calculates a result
            // # --------------------- related to the instance of symmetries with different symmetry_qs!

            // # If this starts getting called in hot loops, consider adding self.preallocated_shifts here too.
            int* shifts = (int*) malloc(this->get_num_shifts() * sizeof(int));
            this->get_unit_shifts_from_index(index_of_node_at_r, shifts);
            complex<double> q_dot_r = 0;
            vector<int> periodicities = this->get_shift_periodicities();
            for (int i=0; i<symmetry_qs.size(); i++) {
                q_dot_r += symmetry_qs[i] * shifts[i] / float(periodicities[i]);
            }
            free(shifts);
            return q_dot_r * 2.0 * std::numbers::pi;
        }
};


class Direct_Sum: public I_Symmetry_Generator {
    private:
        vector<int> shift_periodicities;
        size_t num_shifts;
        int basis_length;
        vector<unique_ptr<I_Symmetry_Generator>> generators;
        // We have to use indirection if we want this vector to work with subclasses of I_Symmetry_Generator.
        void assign_non_generators() {
            // to be used after this->generators is set
            this->shift_periodicities = vector<int>();
            this->num_shifts = 0;
            this->basis_length = 1;
            for (const auto& generator: this->generators) {
                this->num_shifts += generator->get_num_shifts();
                this->basis_length *= generator->get_basis_length();
                vector<int> periodicities = generator->get_shift_periodicities();
                this->shift_periodicities.insert(this->shift_periodicities.end(), periodicities.begin(), periodicities.end());
            }
        }
    public:
        Direct_Sum(const Direct_Sum& that): Direct_Sum(that.generators) {}

        Direct_Sum(Direct_Sum&& that) {
            this->generators = std::move(that.generators);
            this->assign_non_generators();
        }

        template <std::derived_from<I_Symmetry_Generator> T>
        Direct_Sum(const vector<T>& generators) {
            this->generators = vector<unique_ptr<I_Symmetry_Generator>>(generators.size());
            for (int i=0; i<generators.size(); i++) {
                this->generators[i] = generators[i].clone();
            }            
            this->assign_non_generators();
        }

        template <std::derived_from<I_Symmetry_Generator> T>
        Direct_Sum(const vector<unique_ptr<T>>& generators) {  
            this->generators = vector<unique_ptr<I_Symmetry_Generator>>(generators.size());
            for (size_t i=0; i<generators.size(); i++) {
                this->generators[i] = generators[i]->clone();
            }
            this->assign_non_generators();
        }

        Direct_Sum& operator=(Direct_Sum&& that) {
            this->num_shifts = that.num_shifts;
            this->basis_length = that.basis_length;
            this->shift_periodicities = std::move(that.shift_periodicities);
            this->generators = std::move(that.generators);
            return *this;
        }
        unique_ptr<I_Symmetry_Generator> clone() const override {
            return unique_ptr<I_Symmetry_Generator>(new Direct_Sum(this->generators));
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

        void get_unit_shifts_from_index(const int index, int* const shifts) const override {
            div_t res = div_t(index, 0);
            size_t shift_index = 0;
            for (size_t i=0; i<this->generators.size(); i++) {
                res = div(res.quot, this->generators[i]->get_basis_length());
                this->generators[i]->get_unit_shifts_from_index(res.rem, shifts+shift_index);
                shift_index += this->generators[i]->get_num_shifts();
            }
        }

        pos_int get_index_from_unit_shifts(const int* const shifts) const override {
            int index = 0;
            int base = 1;
            int shift_index = 0;
            for (int i=0; i<this->generators.size(); i++) {
                index += base * this->generators[i]->get_index_from_unit_shifts(shifts + shift_index);
                shift_index += int(this->generators[i]->get_num_shifts()); // FIXME: mixing int and size_t!
                base *= this->generators[i]->get_basis_length();
            }
            return index;
        }     

        // virtual size_t get_flat_shift_index(const int* const shifts) = 0;

        int get_symmetry_states_from_state(const state_int* const state, const bool check_lowest, const int external_num_repeat, 
                                                const int external_stride, State_Amplitude* const states_amplitudes) const override {
            int num_found_states = this->generators[0]->get_symmetry_states_from_state(state, check_lowest, states_amplitudes);
            int num_repeat;
            int stride = external_stride;
            int next_stride;
            for (int i=0; i< this->generators.size(); i++) {
                next_stride = stride * this->generators[i]->get_basis_length();
                num_repeat = external_num_repeat * this->basis_length / next_stride;
                for (int j=0; j<num_found_states; j++) {
                    num_found_states += this->generators[i]->get_symmetry_states_from_state(states_amplitudes[j].state, check_lowest, 
                                                                                        num_repeat, stride,
                                                                                        states_amplitudes + num_found_states);
                }
                stride = next_stride;
            }
            return num_found_states;
        }

        void translate_by_symmetry(const state_int* const state, const int* const symmetry_indices, const int external_num_repeat, 
                                        const int external_stride, State_Amplitude& state_amplitude) const override {
            int shift_index = 0;
            int num_repeat;
            int stride = external_stride;
            int next_stride;
            complex<double> amplitude = 1;
            memcpy(state_amplitude.state, state, this->basis_length*sizeof(state_int));
            for (int i=0; i< this->generators.size(); i++) {
                next_stride = stride * this->generators[i]->get_basis_length();
                num_repeat = external_num_repeat * this->basis_length / next_stride;
                this->generators[i]->translate_by_symmetry(state_amplitude.state, symmetry_indices + shift_index, num_repeat, stride, state_amplitude);
                amplitude *= state_amplitude.amplitude;
                shift_index += this->generators[i]->get_num_shifts();
                stride = next_stride;
            }
            state_amplitude.amplitude = amplitude;
        }
};
