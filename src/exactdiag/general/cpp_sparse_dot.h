#pragma once
#include <complex>
#include <cstdlib>
#include <cstring>
using std::complex;


// prefetch does not work?
// gcc
// _builtin_prefetch(&value_inds[j+16],0,0);
// _builtin_prefetch(&vals[value_inds[j+8]],0,0);
// prefetch address can be invalid, there will be no segfault

// clang, MSVC
// _mm_prefetch(prefetch_value_inds+j+4,_MM_HINT_NTA);
// _mm_prefetch(prefetch_vals+value_inds[j+4],_MM_HINT_T1);


void matvec_explicit_elements_pass_column_major(const long long range_start, const long long range_end, 
                        const long long vec_in_length, const long long* const column_pointers, 
                        const complex<double>* const vals, const short* const value_inds, const long long* const row_inds, 
                        const complex<double>* const vec_in, const bool is_sorted, const bool is_triangular, 
                        complex<double>* const __restrict vec_out, const complex<double> weight) {
    //  vec_out initialized elsewhere
    int j;
    long long row_ind, column_ind, column_end = vec_in_length;
    long long end_element;

    if (is_triangular) {
        column_end = range_end;
    }
    if (is_sorted) {
        for (column_ind=0; column_ind<column_end; column_ind++) {
            end_element = column_pointers[column_ind+1];
            for (j=column_pointers[column_ind]; j<end_element; j++) {
                row_ind = row_inds[j];
                if (row_ind >= range_end) {
                    break;
                }
                if (row_ind < range_start) {
                    continue;
                }
                vec_out[row_ind] += vec_in[column_ind] * vals[value_inds[j]] * weight;  
            }                    
        }
    } else {
        for (column_ind=0; column_ind<column_end; column_ind++) {
            end_element = column_pointers[column_ind+1];
            for (j=column_pointers[column_ind]; j<end_element; j++) {
                row_ind = row_inds[j];
                if (row_ind >= range_start && row_ind < range_end) {
                    vec_out[row_ind] += vec_in[column_ind] * vals[value_inds[j]] * weight;  
                }
            }                    
        }
    }
}

complex<double> matvec_explicit_elements_pass_row_major(const long long row_index, const long long* const row_pointers, \
        const complex<double>* const vals, const short* const value_inds, const long long* const column_inds, \
        const complex<double>* const vec_in) {
    
    long long j, start = row_pointers[row_index];
    complex<double> retval = 0;
    for (j=row_pointers[row_index]; j<row_pointers[row_index+1]; j++) {
        //_mm_prefetch((char*) (column_inds + (start+128-row_pointers[461889])*(start+128 < row_pointers[461889]) + row_pointers[461889]),_MM_HINT_T0);
        //_mm_prefetch((char*) (vec_in+column_inds[(start+256-row_pointers[461889])*(start+256 < row_pointers[461889]) + row_pointers[461889]]),_MM_HINT_T1); //column_inds[start] nesmi byt out of bounds
        retval += vec_in[column_inds[j]] * vals[value_inds[j]];  
    }           
    return retval;  
}

complex<double> matvec_hermitian_conjugate_pass_column_major(const long long column_ind, const int num_nonzero_elements, \
        const complex<double>* const vals, const short* const value_inds, const long long* const row_inds, \
        const complex<double>* const vec_in, const bool is_sorted) {
    int j;
    long long row_ind;
    complex<double> herm_conj_term = 0;
    if (num_nonzero_elements == 0) {
        return herm_conj_term;
    }

    if (is_sorted) {
        // row_inds are sorted, i can check for diagonal element first
        row_ind = row_inds[0];
        if (row_ind != column_ind) {   
            herm_conj_term = vec_in[row_ind] * conj(vals[value_inds[0]]);
        }  
        for (j=1; j<num_nonzero_elements; j++) {
            herm_conj_term += vec_in[row_inds[j]] * conj(vals[value_inds[j]]);
        }  
    } else {
        for (j=0; j<num_nonzero_elements; j++) {                  
            row_ind = row_inds[j];
            if (row_ind != column_ind) {                                  
                herm_conj_term += vec_in[row_ind] * conj(vals[value_inds[j]]);   
            }
        }                 
    }   
    return herm_conj_term;  
}


void matvec_hermitian_conjugate_pass_row_major(const long long range_start, const long long range_end, 
                        const long long vec_in_length, const long long* const row_pointers, 
                        const complex<double>* const vals, const short* const value_inds, const long long* const column_inds, 
                        const complex<double>* const vec_in, const bool is_sorted,  
                        complex<double>* const __restrict vec_out, const complex<double> weight) {
    //  vec_out initialized elsewhere
    long long row_ind, column_ind;
    long long end_element, j;

    if (is_sorted) {
        for (row_ind=0; row_ind<range_end; row_ind++) {
            end_element = row_pointers[row_ind+1];
            for (j=row_pointers[row_ind]; j<end_element; j++) {
                column_ind = column_inds[j];
                if (column_ind >= range_end) {
                    break;
                }
                if (column_ind < range_start || column_ind == row_ind) {
                    continue;
                }
                vec_out[column_ind] += vec_in[row_ind] * conj(vals[value_inds[j]]) * weight;  
            }                    
        }
    } else {
        for (row_ind=0; row_ind<range_end; row_ind++) {
            end_element = row_pointers[row_ind+1];
            for (j=row_pointers[row_ind]; j<end_element; j++) {
                column_ind = column_inds[j];
                if (column_ind >= range_start && column_ind < range_end && column_ind != row_ind) {
                    vec_out[column_ind] += vec_in[row_ind] * conj(vals[value_inds[j]]) * weight;  
                }
            }                    
        }
    }
}



void matvec_block_diagonal_row_major(const long long row_start, const long long row_end, const long long* const row_pointers, \
        const complex<double>* const vals, const short* const value_inds, const long long* const column_inds, \
        const complex<double>* const vec_in, const bool is_sorted, const bool is_triangular, complex<double>* const __restrict vec_out) {
    //  vec_out initialized elsewhere
    long long row_ind, j, column_ind, columns_end, columns_start;
    complex<double> value;

    if (is_triangular) {
        if (is_sorted) {
            for (row_ind=row_start; row_ind<row_end; row_ind++) {
                columns_start = row_pointers[row_ind];
                columns_end = row_pointers[row_ind+1];
                if (columns_end-columns_start == 0) {
                    continue;
                }
                for (j=columns_start; j<columns_end-1; j++) {
                    value = vals[value_inds[j]];
                    column_ind = column_inds[j];
                    vec_out[row_ind] += vec_in[column_ind] * value;  
                    vec_out[column_ind] += vec_in[column_ind] * conj(value);  
                }
                column_ind = column_inds[columns_end-1];
                value = vals[value_inds[columns_end-1]];
                vec_out[row_ind] += vec_in[column_ind] * value; 
                if (column_ind != row_ind) {
                    vec_out[row_ind] += vec_in[column_ind] * conj(value);  
                }
            }     
        } else {
            for (row_ind=row_start; row_ind<row_end; row_ind++) {
                for (j=row_pointers[row_ind]; j<row_pointers[row_ind+1]; j++) {
                    value = vals[value_inds[j]];
                    column_ind = column_inds[j];
                    vec_out[row_ind] += vec_in[column_ind] * value;                      
                    if (column_ind != row_ind) {
                        vec_out[column_ind] += vec_in[column_ind] * conj(value);
                    }
                }
            }   
        }
    } else {
        for (row_ind=row_start; row_ind<row_end; row_ind++) {
            for (j=row_pointers[row_ind]; j<row_pointers[row_ind+1]; j++) {
                vec_out[row_ind] += vec_in[column_inds[j]] * vals[value_inds[j]];  
            }
        }        
    }   
}