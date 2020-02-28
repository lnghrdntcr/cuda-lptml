#include <iostream>
#include <cstdio>
#include <algorithm>
#include <random>
#include <chrono>
#include "clion_utils.h"
#include "lib/csv.h"
#include "lib/types.h"
#include "lib/utils.h"
#include "lib/lptml.h"
#include "lib/cuda_wrapper.cuh"

#define IRIS_PATH "/home/francesco/CLionProjects/lptml_cuda/datasets/iris.csv"
#define SYNTH_PATH "/home/francesco/CLionProjects/lptml_cuda/datasets/synth.csv"
#define WINE_PATH "/home/francesco/CLionProjects/lptml_cuda/datasets/wine.csv"

#define ITERATIONS 200
#define NUM_TESTS 10
using namespace std::chrono;

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

    //read_wine(&dataset, &labels, &DIM_Y, WINE_PATH);
    //read_iris(&dataset, &labels, &DIM_Y, IRIS_PATH);
    read_synth(&dataset, &labels, &DIM_Y, SYNTH_PATH);

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
    auto timings = std::vector<unsigned_type>();

    auto iterations_test = std::vector < unsigned_type > {100, 500, 1000, 1500, 2000};

    for (auto it: iterations_test) {

        for (int i = 0; i < NUM_TESTS; ++i) {
            std::cout << "[" << it << "] -> " << i + 1 <<  std::endl;
            train_test_split(&x_train, &x_test, &y_train, &y_test, new_ds, new_labels);
            auto begin = high_resolution_clock::now();
            auto G = fit(x_train, y_train, u, l, DIM_Y, DIM_X, it, combinations(0, x_train.size()));
            auto end = high_resolution_clock::now();
            timings.push_back(duration_cast<milliseconds>(end - begin).count());
            std::cout << "Finished in " << (float_type) timings[timings.size() - 1] / 1000 << "s"<< std::endl;
            auto x_train_lptml = transpose(cpu_mmult(G, transpose(x_train)));
            auto x_test_lptml = transpose(cpu_mmult(G, transpose(x_test)));

            auto accuracy_lptml = predict(x_train_lptml, y_train, x_test_lptml, y_test);
            auto accuracy_knn = predict(x_train, y_train, x_test, y_test);

            // Skip the test if the identity matrix has been used
            if (!is_identity(G)) {
                std::cout << "\t\tAccuracy LPTML -> " << accuracy_lptml
                          << "\n\t\tAccuracy knn -> " << accuracy_knn << std::endl;
                accuracies.push_back(accuracy_lptml);

                x_train.clear();
                x_test.clear();
                y_train.clear();
                y_test.clear();
            } else {
                i--;
                std::cout << "Repeating -> ";
                // reshuffle
                shuffled_ds = parallel_shuffle(dataset, labels);
                new_ds = shuffled_ds.first;
                new_labels = shuffled_ds.second;
                std::cout << std::endl;
                continue;
            }
            std::cout << std::endl;

            // reshuffle
            shuffled_ds = parallel_shuffle(dataset, labels);
            new_ds = shuffled_ds.first;
            new_labels = shuffled_ds.second;

        }
    }

}