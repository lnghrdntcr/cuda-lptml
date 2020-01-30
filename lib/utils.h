//
// Created by francesco on 12/23/19.
//

#ifndef LPTML_UTILS_H
#define LPTML_UTILS_H


/*#include <thrust/sort.h>
#include <thrust/functional.h>*/
#include "csv.h"
#include "types.h"
#include <cstdlib>
#include <ctime>

pair_index_type combinations(unsigned_type begin, unsigned_type end) {
    // Res must be of (end - begin) * (end - begin - 1)
    // Commonly referred as "n choose 2", but doubled since values are stored in line
    // to save random memory accesses

    pair_index_type res;

    // Create the first set autonomusly
    // To avoid a branch in the for loop
    for (int i = 1; i < end; ++i) {
        res.push_back(std::make_pair(0, i));
    }

    int prev_begin = 0;
    for (int i = 1; i < end; ++i) {
        for (int j = prev_begin + 1; j < end; ++j) {
            if (i != j) {
                res.push_back(std::make_pair(i, j));
            }
        }
        prev_begin++;
    }
    return res;
}
template <typename T>
T percentile(std::vector<T> data, unsigned_type percentile_value, const size_t DIM) {
    int index = ((float) percentile_value / 100 * DIM) + 1;
    // TODO: Sort data
    std::sort(data.begin(), data.end());
    return (data[index] + data[index]) / 2;
}

std::vector<unsigned_type> range(unsigned_type begin, unsigned_type end, unsigned_type stride = 1) {
    std::vector<unsigned_type> range_vec;

    for (int i = begin; i < end; i += stride) {
        range_vec.push_back(i);
    }

    return range_vec;

}

#define CUDA_CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

template<unsigned_type num_cols = 4>
void read_iris(
        std::vector <std::vector<float_type>> *dataset, // The dataset in "matrix form"
        std::vector<unsigned_type> *labels,            // Labels vector
        unsigned_type *DIM_Y,                           // Number of elements
        std::string path                               // Path to csv
) {

    // Number of columns is DIM_X + 1 to accomodate for the label row
    io::CSVReader<num_cols + 1> csv(path);

    // Attributes
    float_type sepal_length, sepal_width, petal_length, petal_width;

    // Label
    std::string label = "";

    csv.read_header(io::ignore_no_column, "sepal_length", "sepal_width", "petal_length", "petal_width", "species");

    while (csv.read_row(sepal_length, sepal_width, petal_length, petal_width, label)) {

        std::vector<float_type> cur_row;

        cur_row.push_back(as_float_type(sepal_length));
        cur_row.push_back(as_float_type(sepal_width));
        cur_row.push_back(as_float_type(petal_length));
        cur_row.push_back(as_float_type(petal_width));

        // Ugly `if - else if` because switch case does not supports strings
        if (label.compare("setosa") == 0) {
            (*labels).push_back(0);
        } else if (label.compare("versicolor") == 0) {
            (*labels).push_back(1);
        } else if (label.compare("virginica") == 0) {
            (*labels).push_back(2);
        }

        (*dataset).push_back(cur_row);

    }

    (*DIM_Y) = (*dataset).size();

}
template <class T, typename R>
void train_test_split(
        std::vector<T> *x_train,
        std::vector<T> *x_test,
        std::vector<R> *y_train,
        std::vector<R> *y_test,
        std::vector<T> x,
        std::vector<R> y,
        const float_type factor = 0.5,
        const bool DEBUG = false
        ){

    const unsigned_type split_rateo = (float) 1 / factor;

    if(DEBUG){
        std::cout << "split_rateo = " << (unsigned_type) split_rateo << std::endl;
    }

    for (int i = 0; i < x.size(); ++i) {
        if(i % split_rateo == 0){
            (*x_train).push_back(x[i]);
            (*y_train).push_back(y[i]);
        } else {
            (*x_test).push_back(x[i]);
            (*y_test).push_back(y[i]);
        }
    }

}

bool deep_equal(row_type a, row_type b){
    if(a.size() != b.size()) return false;
    for (int i = 0; i < a.size(); ++i) {
        if(a[i] != b[i]) return false;
    }
    return true;
}

matrix_type identity(size_t dimension){
    auto identity_matrix = matrix_type (dimension, row_type(dimension, 0.0));
    for (int i = 0; i < dimension; ++i) {
        identity_matrix[i][i] = 1;
    }

    return identity_matrix;

}

matrix_type cholesky_transformer(matrix_type initial_matrix){

    auto temp_cholesky = matrix_type(initial_matrix.size(), row_type(initial_matrix.size(), 0.0));

    for (int i = 0; i < temp_cholesky.size(); ++i) {
        for (int j = 0; j < (i + 1); ++j) {
            float_type tmp = 0.0;
            for (int k = 0; k < j; ++k) {
                tmp += temp_cholesky[i][k] * temp_cholesky[j][k];
            }
            if(i == j){
                temp_cholesky[i][j] = sqrt(initial_matrix[i][i] - tmp);
            } else {
                temp_cholesky[i][j] = (float_type) 1.0 / temp_cholesky[j][j] * (initial_matrix[i][j] - tmp);
            }
        }
    }

    return temp_cholesky;

}

template <typename T>
std::vector<T> unwrap_constraint(pair_type constraints){
    std::vector<T> unwrapped;
    for(auto constraint: constraints){
        auto c1 = constraint.first;
        auto c2 = constraint.second;

        for(auto el: c1) unwrapped.push_back(el);

        for(auto el: c2) unwrapped.push_back(el);

    }

    return unwrapped;
}

template <typename T>
std::vector<T> unwrap_matrix(matrix_type matrix){
    std::vector<T> unwrapped;

    for(auto row: matrix){
        for (auto el: row){
            unwrapped.push_back(el);
        }
    }

    return unwrapped;

}

row_type cpu_mmult(matrix_type A, row_type x){
    const unsigned_type num_rows = A.size();
    const unsigned_type num_cols = A[0].size();

    row_type ret(num_rows, 0.0);

    for(int i = 0; i < num_rows; ++i){
        for(int j = 0; j < num_cols; ++j){
            ret[i] += A[i][j] * x[j];
        }
    }
    return ret;
}

#endif //LPTML_UTILS_H
