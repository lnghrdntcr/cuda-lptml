//
// Created by francesco on 12/28/19.
//

#ifndef LPTML_LPTML_H
#define LPTML_LPTML_H

#include <thread>
#include "types.h"
#include "utils.h"
#include "cuda_wrapper.cuh"

unsigned_type count_single_constraint(
                                    const pair_type constraint,
                                    const float_type value,
                                    const bool UPPER = true
){
    float_type distance = 0.0;
    return 0;
}

unsigned_type get_dimension(pair_type S,pair_type D) {
    if(D.size() > 0) return D[0].first.size();
    else return S[0].first.size();
}


std::pair<unsigned_type, unsigned_type> count_violated_constraints_SD(
                                                                     pair_type S,
                                                                     pair_type D,
                                                                     matrix_type G,
                                                                     float_type u,
                                                                     float_type l
){

    // TODO: create 2 threads to count violated constraints between S and D
    cuda_count_violated_constraints_SD(S, D, G, u, l);
    return std::make_pair(0, 0);
}

matrix_type learn_metric(
        pair_type S,
        pair_type D,
        float_type u,
        float_type l,
        unsigned_type t,
        matrix_type initial_solution = matrix_type(),
        const bool DEBUG = true
){
    const unsigned_type dimension = get_dimension(D, S);
    const unsigned_type n         =  D.size() + S.size();

    if(DEBUG){
        std::cout << "dimension = " << dimension << " size = " << n << std::endl;
    }

    if(initial_solution.size() == 0) initial_solution = identity(dimension);

    std::pair <unsigned_type,unsigned_type> violated_constraints = count_violated_constraints_SD(S, D, cholesky_transformer(initial_solution), u, l);// TODO: fill
    auto viol_d = violated_constraints.first;
    auto viol_s = violated_constraints.second;
    return initial_solution;
}


void fit(
        const matrix_type x,                          // Dataset
        const label_row_type y,                       // Labels
        const float_type u,                           // Upper bound
        const float_type l,                           // Lower bound
        const size_t DIM_Y,                           // Number of datapoints
        const size_t DIM_X,                           // Number of features
        const pair_index_type all_pairs,              // Combinations of all pairs in the dataset
        matrix_type initial_solution = matrix_type(), // Initial solution
        const bool DEBUG = true
) {
    pair_type similar_pairs_S;
    pair_type dissimilar_pairs_D;
    std::vector<unsigned_type> randomized_indexes = range(0, all_pairs.size());

    for (int i = 0; i < randomized_indexes.size(); ++i) {
        if (y[all_pairs[i].first] == y[all_pairs[i].second]) {
            similar_pairs_S.push_back(make_pair(x[all_pairs[i].first], x[all_pairs[i].second]));
        } else {
            dissimilar_pairs_D.push_back(make_pair(x[all_pairs[i].first], x[all_pairs[i].second]));
        }

    }

    if (DEBUG) {
        std::cout << "Number of constraints: d = " << dissimilar_pairs_D.size() << " s = " << similar_pairs_S.size()
                  << std::endl;
    }

    learn_metric(similar_pairs_S, dissimilar_pairs_D, u, l, 2000, initial_solution);

}

void predict() {

}

#endif //LPTML_LPTML_H
