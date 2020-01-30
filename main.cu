#include <iostream>
#include <cstdio>
#include "clion_utils.h"
#include "lib/csv.h"
#include "lib/types.h"
#include "lib/utils.h"
#include "lib/lptml.h"
#include "lib/cuda_wrapper.cuh"
#include "/usr/include/mosek.h"
#define IRIS_PATH "/home/francesco/CLionProjects/lptml/datasets/iris.csv"
#define ITERATIONS 2000

#include "lib/mosek/9.1/tools/platform/linux64x86/h/fusion.h"
#include "lib/mosek/9.1/tools/platform/linux64x86/h/mosek.h"


int main() {

    // Dimensions
    const unsigned_type DIM_X = 4;
    unsigned_type DIM_Y = 0;

    // Upper and lower bounds
    float_type u;
    float_type l;

    matrix_type dataset;
    label_row_type labels;

    read_iris(&dataset, &labels, &DIM_Y, IRIS_PATH);

    auto combin = combinations(0, DIM_Y);

    cuda_get_bounds(&u, &l, combin, dataset);

    // Train test split
    matrix_type x_train;
    matrix_type x_test;

    label_row_type y_train;
    label_row_type y_test;

    train_test_split(&x_train, &x_test, &y_train, &y_test, dataset, labels);

    fit(x_train, y_train, u, l, DIM_Y, DIM_X, combinations(0, x_train.size()));
}