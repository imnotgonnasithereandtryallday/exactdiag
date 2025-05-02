#pragma once
#include <algorithm>
#include <format>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
#include <omp.h>
#include <iostream>
#include <iterator>
#include "./types_and_constants.h"
#include "./symmetry.h"


template<class T>
int compare(const void* a, const void* b) {
    // TODO: branchless
    if (((T*)a)[0] < ((T*)b)[0]) {
        return -1;
    }
    if (((T*)a)[0] == ((T*)b)[0]) {
        return 0;
    }
    //if (((T*)a)[0] > ((T*)b)[0]) {
    return 1;
    //}
}

class Basis_Index_Map {
    private:
    size_t dense_to_sparse_len;
    std::vector<MATRIX_INDEX_TYPE> dense_to_sparse;
    std::vector<int>num_sparse_states_in_dense;

    public:
    Basis_Index_Map(
        const std::vector<MATRIX_INDEX_TYPE>& translation_array, 
        const std::vector<int>& num_sparse_in_dense//, 
        //const vector<MATRIX_INDEX_TYPE>[:,:]& states_in_hole_index // FIXME: 2d array, not implemented anyways
    ) 
    : dense_to_sparse_len(translation_array.size())
    , dense_to_sparse(translation_array)
    , num_sparse_states_in_dense(num_sparse_in_dense)
    {
        /* translation_array: Has a length equal to the size of the dense basis.
                              Index of the array corresponds to the dense index of a state, 
                              value corresponds to the sparse index of the state. 

           num_sparse_in_dense: Has a length equal to the size of the dense basis.
                                Index of the array corresponds to the dense index of a state, 
                                value corresponds to the number of sparse basis states that 
                                are represented with nonzero amplitude in the linear combination that 
                                forms the dense state.
                                Saving this information is useful for calculating operators that commute with symmetries

           states_in_hole_index: This was supposed to allow to search only a small part of dense_to_sparse in the get_dense method.
                                 It is not currently used, as it does not seem to speed the search up, but might become
                                 usefull for much larger basis sets.
        */
    }
    Basis_Index_Map(
        std::vector<MATRIX_INDEX_TYPE>&& translation_array, 
        std::vector<int>&& num_sparse_in_dense
    ) 
    : dense_to_sparse_len(translation_array.size())
    , dense_to_sparse(translation_array)
    , num_sparse_states_in_dense(num_sparse_in_dense)
    {}

    Basis_Index_Map (const std::filesystem::path& path) {
        std::ifstream file (path, std::ios::binary);
        if (!file.is_open()) {
            std::string message = std::format("Failed to open {}", path.string());
            std::cerr<<"ERROR: "<<message<<std::endl;
            throw std::runtime_error(message);
        }
        file.exceptions(std::ifstream::failbit | std::ifstream::badbit | std::ifstream::eofbit); 
        try {
            file.read(reinterpret_cast<char*>(&this->dense_to_sparse_len), sizeof(this->dense_to_sparse_len));
            this->dense_to_sparse = std::vector<MATRIX_INDEX_TYPE> (this->dense_to_sparse_len);
            this->num_sparse_states_in_dense = std::vector<int>(this->dense_to_sparse_len);
            file.read(reinterpret_cast<char*>(this->dense_to_sparse.data()), sizeof(MATRIX_INDEX_TYPE) * this->dense_to_sparse_len);
            file.read(reinterpret_cast<char*>(this->num_sparse_states_in_dense.data()), sizeof(int) * this->dense_to_sparse_len);
        } catch (const std::ios_base::failure& e) {
            std::string message = std::format("ERROR: Failed to read {}. {}", path.string(), e.what());
            std::cerr<<message<<std::endl;
            throw e;
        } catch (const std::bad_alloc& e) {
            std::string message = std::format("ERROR: Failed to allocate {} elements when loading {}. {}", this->dense_to_sparse_len, path.string(), e.what());
            std::cerr<<message<<std::endl;
            throw e;
        } catch (const std::length_error& e) {
            std::string message = std::format("ERROR: Failed to allocate {} elements when loading {}. {}", this->dense_to_sparse_len, path.string(), e.what());
            std::cerr<<message<<std::endl;
            throw e;
        }

    }

    Basis_Index_Map (const Basis_Index_Map& other) 
    : dense_to_sparse_len(other.dense_to_sparse_len)
    , dense_to_sparse(other.dense_to_sparse)
    , num_sparse_states_in_dense(other.num_sparse_states_in_dense)
    {}

    Basis_Index_Map (Basis_Index_Map&& other) 
    : dense_to_sparse_len(other.dense_to_sparse_len)
    , dense_to_sparse(std::move(other.dense_to_sparse))
    , num_sparse_states_in_dense(std::move(other.num_sparse_states_in_dense))
    {}
       
    Basis_Index_Map& operator=(const Basis_Index_Map& that) {
        this->dense_to_sparse_len = that.dense_to_sparse_len;
        this->dense_to_sparse = that.dense_to_sparse;
        this->num_sparse_states_in_dense = that.num_sparse_states_in_dense;
        return *this;
    }  

    Basis_Index_Map& operator=(Basis_Index_Map&& that) {
        this->dense_to_sparse_len = that.dense_to_sparse_len;
        this->dense_to_sparse = std::move(that.dense_to_sparse);
        this->num_sparse_states_in_dense = std::move(that.num_sparse_states_in_dense);
        return *this;
    }  

    void save(std::filesystem::path path) const {
        std::filesystem::path folder = path.parent_path();
        try {
            std::filesystem::create_directories(folder);
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << std::format("Failed to create folders {}: {}", folder.string(), e.what()) << std::endl;
            throw e;
        }
        std::ofstream file (path, std::ios::binary);
        file.exceptions(std::ofstream::failbit | std::ofstream::badbit | std::ofstream::eofbit); 
        try {
            file.write(reinterpret_cast<const char*>(&this->dense_to_sparse_len), sizeof(this->dense_to_sparse_len));
            file.write(reinterpret_cast<const char*>(this->dense_to_sparse.data()), sizeof(MATRIX_INDEX_TYPE) * this->dense_to_sparse_len);
            file.write(reinterpret_cast<const char*>(this->num_sparse_states_in_dense.data()), sizeof(int) * this->dense_to_sparse_len);
        } catch (const std::ios_base::failure& e) {
            std::cerr << std::format("Failed to save {}: {}", path.string(), e.what()) << std::endl;
            throw e;
        }
    }

    MATRIX_INDEX_TYPE get_sparse(MATRIX_INDEX_TYPE dense_index) const {
        return this->dense_to_sparse[dense_index];
    }

    MATRIX_INDEX_TYPE get_dense(MATRIX_INDEX_TYPE sparse_index) const {
        // FIXME: return an option.
        if (sparse_index < 0) {
            return -1;
        }
        void* dense_index_ptr = bsearch( // TODO: not guaranteed to be implemented as binary search. CHECK time complexity!
            &sparse_index,
            this->dense_to_sparse.data(),
            this->dense_to_sparse_len,
            sizeof(MATRIX_INDEX_TYPE),
            compare<MATRIX_INDEX_TYPE>
        );
        if (dense_index_ptr == nullptr) {
            return -1;
        }
        return ((MATRIX_INDEX_TYPE*) dense_index_ptr) - this->dense_to_sparse.data();
    }

    int get_num_sparse_in_dense(MATRIX_INDEX_TYPE dense_index) const {
        return this->num_sparse_states_in_dense[dense_index];
    }

    MATRIX_INDEX_TYPE get_num_sparse_in_dense_from_sparse(MATRIX_INDEX_TYPE sparse_index) const {
        MATRIX_INDEX_TYPE dense_index = this->get_dense(sparse_index);
        if (dense_index > -1) {
            return this->num_sparse_states_in_dense[dense_index];
        }
        return 0;
    }

    MATRIX_INDEX_TYPE get_num_states() const {
        return this->dense_to_sparse_len;
    }
};


Basis_Index_Map calc_basis_map(MATRIX_INDEX_TYPE num_major_only_states, const I_State_Index_Amplitude_Translator& state_translator, unsigned num_threads) {
    /*Construct the mapping between the sparse and dense bases.
    
    Works in three steps:
    1) Assuming the state is composed of multiple parts (e.g. holes and spins), the major part is checked
       for compatibility with the dense states.
       # ----- this could be extended to multiple steps for checking all but the most minor part
    2) For the major parts that are compatible, all the possible complete states are checked again, giving us the 
       sparse indices and the number of sparse states forming the linear combination of the dense state.
    3) The results are gathered in a single-threaded part of the code and the map object is returned
    
    num_major_only_states: the number of states in the major part of the sparse basis.
                           What constitues a major part here is given by the 
                           check_major_index_supports_lowest_states and get_lowest_sparse_indices_from_major_index 
                           methods of the state_translator argument.
    state_translator: an instance of a class derived from I_State_Index_Amplitude_Translator that implements methods
                      check_major_index_supports_lowest_states and get_lowest_sparse_indices_from_major_index.
    */
    // TODO: Add logging.
    vector<MATRIX_INDEX_TYPE> eligible_major_indeces (num_major_only_states);
    size_t chunksize = num_major_only_states/num_threads + 1;
    
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunksize)
    for (long long i=0; i < num_major_only_states; ++i) { // omp does not support unsigned type here
        eligible_major_indeces[i] = state_translator.check_major_index_supports_lowest_states(i);
    }
    std::erase(eligible_major_indeces, -1);

    size_t length = eligible_major_indeces.size();
    vector<Indices_Counts> lowest_sparse_indices_in_major_index (length);
    MATRIX_INDEX_TYPE num_dense_states = 0;
    
    chunksize = 1; // The distribution of work can be very un-even and length small.
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunksize) reduction(+:num_dense_states)
    for (long long i=0; i < length; ++i) { // omp does not support unsigned type here
        lowest_sparse_indices_in_major_index[i] = state_translator.get_lowest_sparse_indices_from_major_index(eligible_major_indeces[i]);
        num_dense_states += lowest_sparse_indices_in_major_index[i].indices.size();
    }
    MATRIX_INDEX_TYPE dense_index = 0;
    // TODO: Is there a better way to handle the memory requirements here?
    vector<MATRIX_INDEX_TYPE> translation_array (num_dense_states);
    vector<int> num_sparse_in_dense (num_dense_states);
    for (size_t i=0; i < length; ++i) {
        std::vector<MATRIX_INDEX_TYPE> lowest_sparse_indices = lowest_sparse_indices_in_major_index[i].indices;
        std::vector<int> num_sparse_states = lowest_sparse_indices_in_major_index[i].counts;
        size_t n = lowest_sparse_indices.size();
        if (n == 0) {
            continue;
        }
        memcpy(translation_array.data() + dense_index, lowest_sparse_indices.data(), n * sizeof(MATRIX_INDEX_TYPE));
        memcpy(num_sparse_in_dense.data() + dense_index, num_sparse_states.data(), n * sizeof(int));
        dense_index += n;
    }
    return Basis_Index_Map(std::move(translation_array), std::move(num_sparse_in_dense));
}


Basis_Index_Map get_basis_map(const std::filesystem::path& path, MATRIX_INDEX_TYPE num_major_only_states, 
                              const I_State_Index_Amplitude_Translator& state_translator, unsigned num_threads) {
    if (std::filesystem::exists(path)) {
        return Basis_Index_Map(path);
    }
    Basis_Index_Map map = calc_basis_map(num_major_only_states, state_translator, num_threads);
    map.save(path);
    return map;
}

std::shared_ptr<Basis_Index_Map> get_basis_map(const std::string& path, MATRIX_INDEX_TYPE num_major_only_states, 
                              const I_State_Index_Amplitude_Translator& state_translator, unsigned num_threads) {
    // cython does not support std::filesystem, returns shared_ptr so we avoid cython's need for nullary constructor
    std::filesystem::path the_real_path (path);
    Basis_Index_Map map = get_basis_map(the_real_path, num_major_only_states, state_translator, num_threads);
    return std::shared_ptr<Basis_Index_Map>(new Basis_Index_Map(std::move(map)));
}