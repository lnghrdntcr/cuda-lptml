#include <iostream>
#include <cstdio>
#include <algorithm>
#include <random>
#include "clion_utils.h"
#include "lib/csv.h"
#include "lib/types.h"
#include "lib/utils.h"
#include "lib/lptml.h"
#include "lib/cuda_wrapper.cuh"
#define IRIS_PATH "/home/francesco/CLionProjects/lptml/datasets/iris.csv"
#define ITERATIONS 200
#define NUM_TESTS 10

int main() {
    // Initialize random seed
    srand(time(0));

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
    std::vector<float_type> accuracies;

    auto shuffled_ds = parallel_shuffle(dataset, labels);
    auto new_ds = shuffled_ds.first;
    auto new_labels = shuffled_ds.second;

    for (int i = 0; i < NUM_TESTS; ++i) {
        std::cout << "Test: " << i + 1 << std::endl;
        train_test_split(&x_train, &x_test, &y_train, &y_test, new_ds, new_labels);
        auto G = fit(x_train, y_train, u, l, DIM_Y, DIM_X, combinations(0, x_train.size()));
        auto x_train_lptml = transpose(cpu_mmult(G, transpose(x_train)));
        auto x_test_lptml = transpose(cpu_mmult(G, transpose(x_test)));

        auto accuracy_lptml = predict(x_train_lptml, y_train, x_test_lptml, y_test);
        auto accuracy_knn = predict(x_train, y_train, x_test, y_test);

        std::cout << "\t\tAccuracy LPTML -> " << accuracy_lptml
                  << "\n\t\tAccuracy knn -> " << accuracy_knn << std::endl;
        accuracies.push_back(accuracy_lptml);

        x_train.clear();
        x_test.clear();
        y_train.clear();
        y_test.clear();
    }
}