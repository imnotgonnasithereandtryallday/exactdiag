#pragma once
// defined in .h because cython generates .cpp files
#include <vector>
#include <cstdlib>
#include <complex>
#include "./types_and_constants.h"
#include <string.h>
#include <iterator>
using std::vector;
using std::malloc; 
using std::free;
using std::complex;

class Product_Generator: public std::iterator<std::input_iterator_tag, vector<int>> {
    // Generates elements of a cartesian product of length sets,
    // each defined as {0, ..., tops[i]}
    int iteration;
    int length;
    int* tops;
    vector<int> result;
    void calc_result() {
        int remains = this->iteration;
        int den = 1;
        for (int i = this->length-1; i >= 0; i--) {
            this->result[i] = (remains / den) % this->tops[i];
            remains -= this->result[i]*den;
            den *= this->tops[i];
        }
    }
    public:
        Product_Generator() {
            // seems like cython requires a argument-less constructor
            this->tops = nullptr;
        }

        Product_Generator(const int* tops, int length) {
            this->iteration = 0;
            this->length = length;
            this->tops = (int*) malloc(length * sizeof(int));
            memcpy(this->tops, tops, length * sizeof(int));
            this->result = vector<int>(length);
            calc_result();
        }

        Product_Generator(const int* tops, int length, int skip_first) {
            this->iteration = skip_first;
            this->length = length;
            this->tops = (int*) malloc(length * sizeof(int));
            memcpy(this->tops, tops, length * sizeof(int));
            this->result = vector<int>(length);
            calc_result();
        }

        Product_Generator(const Product_Generator & that) { 
            this->iteration = that.iteration;
            this->length = that.length;
            this->tops = (int*) malloc(this->length * sizeof(int));
            memcpy(this->tops, that.tops, this->length * sizeof(int));
            this->result = vector<int>(that.result);
            calc_result();
        }

        ~Product_Generator() {
            free(this->tops);
        }

        Product_Generator& operator=(const Product_Generator& that){
            if (this == &that) {
                return *this;
            }
            this->iteration = that.iteration;
            this->length = that.length;
            free(this->tops);
            this->tops = (int*) malloc(this->length * sizeof(int));
            memcpy(this->tops, that.tops, this->length * sizeof(int)); // exception safety?
            this->result = vector<int>(that.result);
            return *this;
        }

        vector<int>& operator*() {return this->result;}

        vector<int>& operator++() {
            // does not check for iteration limits
            this->iteration++;
            calc_result();
            return operator*();
        }

        vector<int> operator++(int) {
            vector<int> tmp = operator*(); 
            operator++(); 
            return tmp;
        }

        void begin() {
            this->iteration = 0;
            calc_result();
        }

        vector<int> next() {
            // cython does not support the pre-/post-increment operator
            // the zero is a dummy argument to call post-increment
            return operator++(0);
        }
};


class State_Amplitude {
    public:
        state_int* state;
        complex<double> amplitude;
        pos_int length;

        State_Amplitude() {
            this->state = nullptr;
            this->amplitude = 0;
            this->length = 0;
        }

        State_Amplitude(int length, const complex<double>& amplitude) {
            this->state = (state_int*) malloc(length * sizeof(state_int));
            this->amplitude = amplitude;
            this->length = length;
        }

        State_Amplitude(pos_int length, const state_int* state, const complex<double>& amplitude) {
            this->state = (state_int*) malloc(length * sizeof(state_int));
            memcpy(this->state, state, length * sizeof(state_int));
            this->amplitude = amplitude;
            this->length = length;
        }

        State_Amplitude(const State_Amplitude & state_amplitude) {
            this->amplitude = state_amplitude.amplitude;
            this->length = state_amplitude.length;
            this->state = (state_int*) malloc(this->length * sizeof(state_int));
            memcpy(this->state, state_amplitude.state, this->length * sizeof(state_int));
        }

        State_Amplitude(State_Amplitude && state_amplitude) {
            this->amplitude = state_amplitude.amplitude;
            this->length = state_amplitude.length;
            this->state = state_amplitude.state;
            state_amplitude.state = nullptr;
        }

        ~State_Amplitude() {
            free(this->state);
        }

        State_Amplitude& operator=(const State_Amplitude& that) {
            if (this == &that) {
                return *this;
            }
            this->amplitude = that.amplitude;
            if (this->length != that.length) {
                this->length = that.length;
                free(this->state);
                this->state = (state_int*) malloc(this->length * sizeof(state_int));
            }
            memcpy(this->state, that.state, this->length * sizeof(state_int)); // exception safety? // what happens if that.length is 0 and that.state is nullptr?
            return *this;
        }

        State_Amplitude& operator=(State_Amplitude&& that){
            if (this == &that) {
                return *this;
            }
            this->amplitude = that.amplitude;
            this->length = that.length;
            free(this->state);
            this->state = that.state;
            that.state = nullptr;
            return *this;
        }

};

