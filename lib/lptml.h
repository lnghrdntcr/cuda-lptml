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

h_prime_type calculate_initial_basis(
        h_prime_type H
        ) {
    h_prime_type tmp;
    std::vector<constraint_type> tmp_v;
    tmp_v.push_back(H["S"][0]);
    tmp["S"] = tmp_v;
    return tmp;
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

std::vector<constraint_type> unit_get_permutation(
        std::vector<constraint_type> constraints,
        std::vector<constraint_type> b_constraints
    ) {

    std::vector<constraint_type> c_prime_proj;

    for(int i = 0; i < constraints.size(); ++i){
        constraint_type constraint = constraints[i];
        bool exists_in_both = false;
        for(int j = 0; j < b_constraints.size(); ++j){

            constraint_type b_constraint = b_constraints[j];

            if(deep_equal(constraint.first, b_constraint.first) && deep_equal(constraint.second, b_constraint.second)){
                exists_in_both = true;
            }
        }

        if(!exists_in_both) {
            c_prime_proj.push_back(constraint);
        }
    }
    return c_prime_proj;
}

h_prime_type get_permutation(
        h_prime_type B,
        h_prime_type C,
        const bool DEBUG = false
        ) {

    h_prime_type C_prime;

    std::thread s([&B, &C, &C_prime]() {
        auto s_constraints = (C.find("S") != C.end()) ? C.find("S")->second : std::vector<constraint_type>(0);
        auto s_b_constraints = (B.find("S") != B.end()) ? B.find("S")->second : std::vector<constraint_type>(0);
        auto c_prime_s_proj = unit_get_permutation(s_constraints, s_b_constraints);
        C_prime["S"] = c_prime_s_proj;
    });

    std::thread t([&B, &C, &C_prime]() {
        auto d_constraints = (C.find("D") != C.end()) ? C.find("D")->second : std::vector<constraint_type>(0);
        auto d_b_constraints = (B.find("D") != B.end()) ? B.find("D")->second : std::vector<constraint_type>(0);
        auto c_prime_d_proj = unit_get_permutation(d_constraints, d_b_constraints);
        C_prime["D"] = c_prime_d_proj;
    });

    s.join();
    t.join();

    if(DEBUG) std::cout << "Size: \n\tC_prime[\"S\"] = " << C_prime["S"].size() << "\n\tC_prime[\"D\"] = " << C_prime["S"].size()<< std::endl;

    return C_prime;

}

void maximal_violation(
        h_prime_type constraints,
        int skip,
        matrix_type A,
        float_type u,
        float_type l,
        constraint_type &c_ret,
        float_type &e_ret,
        int &t_ret,
        std::string &key_ret
        ){
    //TODO: this segfaults
    int worst_index_S = 0;
    float worst_value_S = INT_MAX * -1;
    int worst_index_D = 0;
    float worst_value_D = INT_MAX * -1;
    int idx_D = 0;
    int idx_S = 0;
    bool empty_S = false;
    bool empty_D = false;
    matrix_type G;

    if(constraints.size() == 0) return;
    G = cholesky_transformer(A);

    std::thread s([&worst_index_S, &worst_value_S, &constraints, &G, u, &idx_S, skip, &empty_S](){
        auto it_s = constraints.find("S");
        if(it_s == constraints.end()) {
            empty_S = true;
            return;
        }

        auto constraints_S = constraints.find("S")->second;

        for(auto constraint: constraints_S) {
            auto i = cpu_mmult(G, constraint.first);
            auto j = cpu_mmult(G, constraint.second);

            float_type violation_tmp = 0.0;
            for(unsigned_type iter = 0; iter < i.size(); ++iter){
                violation_tmp += (i[iter] - j[iter]) * (i[iter] - j[iter]);
            }
            violation_tmp = sqrt(violation_tmp) - u;

            if(violation_tmp > worst_value_S && idx_S >= skip){
                worst_value_S = violation_tmp;
                worst_index_S = idx_S;
                idx_S ++;
            }
        }
    });

    std::thread d([&worst_index_D, &worst_value_D, &constraints, &G, l, &idx_D, skip, &empty_D](){
        auto it_s = constraints.find("D");
        if(it_s == constraints.end()) {
            empty_D = true;
            return;
        }

        auto constraints_D = constraints.find("D")->second;

        for(auto constraint: constraints_D) {
            auto i = cpu_mmult(G, constraint.first);
            auto j = cpu_mmult(G, constraint.second);
            float_type violation_tmp = 0.0;
            for(unsigned_type iter = 0; iter < i.size(); ++iter){
                violation_tmp += (i[iter] - j[iter]) * (i[iter] - j[iter]);
            }
            violation_tmp = l - sqrt(violation_tmp);

            if(violation_tmp > worst_value_D && idx_D >= skip){
                worst_value_D = violation_tmp;
                worst_index_D = idx_D;
                idx_D ++;
            }
        }
    });

    s.join();
    d.join();

    float_type worst_index;
    float_type worst_value;
    std::string key = "";
    if((empty_D  && !empty_S) || worst_value_S <= worst_value_D) {
        worst_index = worst_index_S;
        worst_value = worst_value_S;
        key = "S";
    } else {
        worst_index = worst_index_D;
        worst_value = worst_value_D;
        key = "D";
    }

    c_ret = constraints[key][worst_index];
    e_ret = worst_value;
    t_ret = worst_index;
    key_ret = key;
}

void pivot_LPType(
        h_prime_type B,
        h_prime_type C,
        float_type u,
        float_type l,
        unsigned_type d,
        unsigned_type last_cost,
        bool use_last_cost,
        matrix_type basis_A,
        const bool DEBUG = true
    ){

    bool calculate_basis_cost = !use_last_cost;
    unsigned_type current_basis_cost = last_cost;

    matrix_type A = identity(d);
    if(basis_A.size() != 0) A = basis_A;

    auto C_perm = get_permutation(B, C);
    auto Buc = h_prime_type(B);
    int i = 0;

    while(true){
        constraint_type c;
        float_type e;
        int t;
        std::string key;

        maximal_violation(C_perm, i, A, u, l, c, e, t, key);

        if( DEBUG ) std::cout << "E = " << e << " t = " << t << " key = " << key << std::endl;
        if(e <= 0) break;

        auto to_unshift = C_perm[key][t];
        C_perm[key].erase(C_perm[key].begin() + t);
        C_perm[key].insert(C_perm[key].begin(), to_unshift);

        i ++;

        auto Bh = h_prime_type(B);
        Bh[key].push_back(c);
        Buc[key].push_back(c);

        // Calculate the cost of the new candidate basis and the cost of the current basis


}


}

matrix_type learn_metric (
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
        pivot_LPType(B0, R, u, l, dimension, 0, false, matrix_type());
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
