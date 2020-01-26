//
// Created by francesco on 12/28/19.
//

#ifndef LPTML_LPTML_H
#define LPTML_LPTML_H

#include <thread>
#include "types.h"
#include "utils.h"
#include "cuda_wrapper.cuh"
#include <cmath>
#include <cstdlib>
#include <ctime>

unsigned_type count_single_constraint(
                                    const pair_type constraint,
                                    const float_type value,
                                    const bool UPPER = true
){

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
                                                                     float_type l,
                                                                     unsigned_type iterations = 2000,
                                                                     const bool DEBUG = true
){

    srand(time(0));

    // TODO: create 2 threads to count violated constraints between S and D
    auto constraints = cuda_count_violated_constraints_SD(S, D, G, u, l);
    const unsigned_type n = S.size() + D.size();

    if(DEBUG){
        std::cout << "constraints -> D = " << constraints.first << " S = " << constraints.second << std::endl;
    }

    return constraints;
}

std::vector<float_type> get_rand_vector_r(unsigned_type dimension){

    std::vector<float_type> ret(dimension, 0);

    for (int i = 0; i < dimension; ++i) {
        ret[i] = (float_type) rand() / RAND_MAX;
    }

    // calculate vector norm
    float tmp = 0.0;
    for (int i = 0; i < dimension; ++i) {
        tmp += ret[i] * ret[i];
    }
    tmp = sqrt(tmp);

    for(int i = 0; i < dimension; ++i) {
        ret[i] /= tmp;
    }

    return ret;


}

std::pair<h_prime_type, unsigned_type> subsample (
        const pair_type S,
        const pair_type D,
        const float_type p
        ) {
    srand(time(0));
    unsigned_type s_count = 0;
    unsigned_type d_count = 0;
    h_prime_type H_prime;

    while(s_count == 0){
        for (auto c: S){
            float prob = (float) rand() / RAND_MAX;
            if(prob <= p){
                H_prime["S"].push_back(c);
                s_count++;
            }
        }
    }

    while(d_count == 0){
        for(auto c: D){
            float prob = (float) rand() / RAND_MAX;
            if(prob <= p) {
                H_prime["D"].push_back(c);
                d_count ++;
            }
        }
    }
return std::make_pair(H_prime, s_count + d_count);
}

constraint_type calculate_initial_basis(
        h_prime_type H
        ) {
    return H["S"][0];
}

h_prime_type initial_sort(
        h_prime_type &constraints,
        const matrix_type A,
        const float_type u,
        const float_type l
        ){


    auto G = cholesky_transformer(A);

    int idx_S = 0;
    unsigned_type worst_index_S = 0;

    int idx_D = 0;
    unsigned_type worst_index_D = 0;

    float_type    worst_value_S = 0.0;
    float_type    worst_value_D = 0.0;

    std::thread s([constraints, G, u, &worst_value_S, &worst_index_S, &idx_S](){
        auto cc = constraints.find("S");
        for(auto c: cc->second){

            auto i = cpu_mmult(G, c.first);
            auto j = cpu_mmult(G, c.second);

            float_type violation_tmp = 0.0;
            for(unsigned_type iter = 0; iter < i.size(); ++iter){
                violation_tmp += (i[iter] - j[iter]) * (i[iter] - j[iter]);
            }
            violation_tmp = sqrt(violation_tmp) - u;

            if(violation_tmp < worst_value_S){
                worst_value_S = violation_tmp;
                worst_index_S = idx_S;
                idx_S ++;
            }

        }
    });

    std::thread t([constraints, G, l, &worst_value_D, &worst_index_D, &idx_D](){
        auto cc = constraints.find("D");
        for(auto c: cc->second){

            auto i = cpu_mmult(G, c.first);
            auto j = cpu_mmult(G, c.second);

            float_type violation_tmp = 0.0;
            for(unsigned_type iter = 0; iter < i.size(); ++iter){
                violation_tmp += (i[iter] - j[iter]) * (i[iter] - j[iter]);
            }
            violation_tmp = l - sqrt(violation_tmp);

            if(violation_tmp < worst_value_D){
                worst_value_D = violation_tmp;
                worst_index_D = idx_D;
                idx_D ++;
            }

        }
    });

    t.join();
    s.join();

    if(worst_value_D <= worst_value_S) {
        auto winner = constraints["D"][worst_index_D];
        constraints["D"].erase(constraints["D"].begin() + worst_index_D);
        constraints["D"].insert(constraints["D"].begin(), winner);
    } else {
        auto winner = constraints["S"][worst_index_S];
        constraints["S"].erase(constraints["S"].begin() + worst_index_S);
        constraints["S"].insert(constraints["S"].begin(), winner);
    }

    return constraints;
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
    // Initial solution is also best_A
    if(initial_solution.size() == 0) initial_solution = identity(dimension);

    std::pair <unsigned_type,unsigned_type> violated_constraints = count_violated_constraints_SD(S, D, cholesky_transformer(initial_solution), u, l);
    auto viol_d = violated_constraints.first;
    auto viol_s = violated_constraints.second;

    const unsigned_type initial_violation_count = viol_d + viol_s;

    for(int i = 0; i < t; ++i){
        float exponent = -1 * (rand() % ((int) log2(n)) + 3);
        float p = pow(2.0, exponent);

        unsigned_type min_constraints = pow(dimension, 2);

        auto res = subsample(S, D, p);
        auto R = res.first;
        auto r_count = res.second;

        auto B0 = calculate_initial_basis(R);
        auto rand_vector_r = get_rand_vector_r(dimension);

        // Initial solution is also best_A
        R = initial_sort(R, initial_solution, u, l);

        //TODO: pivot_LP_TYPE

        //if(DEBUG) std::cout << "R_COUNT = " << r_count << std::endl;
    }

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
