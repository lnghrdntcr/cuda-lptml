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
#include <functional>
#include <algorithm>

#include "mosek/9.1/tools/platform/linux64x86/h/fusion.h"
#include "mosek/9.1/tools/platform/linux64x86/h/mosek.h"

using namespace mosek::fusion;
using namespace monty;

unsigned_type get_dimension(
        const pair_type& S,
        const pair_type& D) {
    if(D.size() > 0) return D[0].first.size();
    else return S[0].first.size();
}


std::pair<unsigned_type, unsigned_type> count_violated_constraints_SD(
                                                                     const pair_type& S,
                                                                     const pair_type& D,
                                                                     matrix_type G,
                                                                     const float_type& u,
                                                                     const float_type& l,
                                                                     const unsigned_type& iterations = 200,
                                                                     const bool DEBUG = false
){

    srand(time(0));

    // Create 2 threads to count violated constraints between S and D
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
        const pair_type& S,
        const pair_type& D,
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
        const matrix_type& A,
        const float_type& u,
        const float_type& l
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
        const std::vector<constraint_type>& constraints,
        const std::vector<constraint_type>& b_constraints
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

unrolled_h_prime_type unroll_h_prime(
        const h_prime_type& h_prime
        ) {

    unrolled_h_prime_type result;

    bool s_skip = false;
    bool d_skip = false;

    auto it_s = h_prime.find("S");
    auto it_d = h_prime.find("D");

    if(it_s == h_prime.end()) s_skip = true;
    if(it_d == h_prime.end()) d_skip = true;

    if(s_skip && d_skip) return unrolled_h_prime_type();

    if(!s_skip) {
        for(auto c: it_s->second) {
            result.push_back(std::make_pair(c, "S"));
        }
    }


    if(!d_skip) {
        for(auto c: it_d->second) {
            result.push_back(std::make_pair(c, "D"));
        }
    }

    return result;

}

h_prime_type get_permutation(
        const h_prime_type& B,
        const h_prime_type& C,
        const bool DEBUG = false
        ) {

    h_prime_type C_prime;

    unrolled_h_prime_type b_constraints = unroll_h_prime(B);
    unrolled_h_prime_type c_constraints = unroll_h_prime(C);

    for(unsigned_type i = 0; i < c_constraints.size(); ++i){
        bool exists_in_both = false;
        for(unsigned_type j = 0; j < b_constraints.size(); ++j){
            if(
                (b_constraints[j].second == c_constraints[i].second) &&
                deep_equal(b_constraints[j].first.first, c_constraints[i].first.first) &&
                deep_equal(b_constraints[j].first.second, c_constraints[i].first.second)
                ) {
                exists_in_both = true;
            }
        }
        if(!exists_in_both) C_prime[c_constraints[i].second].push_back(c_constraints[i].first);
    }

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

    if(constraints.size() == 0) {
        e_ret = -1;
        return;
    }
    G = cholesky_transformer(A);

    std::thread s([&worst_index_S, &worst_value_S, &constraints, &G, u, &idx_S, skip, &empty_S](){
        auto it_s = constraints.find("S");
        if(it_s == constraints.end() || it_s->second.size() == 0) {
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
        if(it_s == constraints.end() || it_s->second.size() == 0) {
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
    if((empty_D  && !empty_S) && worst_value_S <= worst_value_D) {
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

matrix_type semidefsolver(
        h_prime_type H,
        const float_type u,
        const float_type l,
        const bool DEBUG = false
        ) {

        bool s_empty = false;
        bool d_empty = false;
        auto it_s = H.find("S");
        auto it_d = H.find("D");

        if(it_s == H.end()){
            s_empty = true;
        }


        if(it_d == H.end()){
            d_empty = true;
        }

        if(d_empty && s_empty) {
            return matrix_type(); // an empty set somehow
        }

        unsigned_type d = 0;

        if(d_empty) d = H["S"][0].first.size();
        else d = H["D"][0].first.size();

        if(DEBUG) std::cout << "Dimension "  << d << std::endl;

        Model::t M = new Model(); auto _M = finally([&]() { M->dispose(); });
        Variable::t A = M->variable("A", Domain::inPSDCone(d));
        if(!s_empty){
            for(auto constraint: it_s -> second) {
                auto i = constraint.first;
                auto j = constraint.second;

                std::vector<double> sub(i.size());

                for(int ii = 0; ii < i.size(); ++ii) {
                    sub[ii] = i[ii] - j[ii];
                }

                auto sub_ptr = new_array_ptr<double, 1>(sub.size());
                std::copy(sub.begin(), sub.end(), sub_ptr->begin());
                M->constraint(Expr::mulElm(sub_ptr, Expr::mul(A, sub_ptr)), Domain::lessThan(u * u));
            }
        }
        if(!d_empty){
            for(auto constraint: it_d -> second) {
                auto i = constraint.first;
                auto j = constraint.second;

                std::vector<double> sub(i.size());

                for(int ii = 0; ii < i.size(); ++ii) {
                    sub[ii] = i[ii] - j[ii];
                }

                auto sub_ptr = new_array_ptr<double, 1>(sub.size());
                std::copy(sub.begin(), sub.end(), sub_ptr->begin());
                M->constraint(Expr::mulElm(sub_ptr, Expr::mul(A, sub_ptr)), Domain::greaterThan(l * l));
            }
        }
        M->objective( ObjectiveSense::Minimize, 0);
        try {
            M->solve();

            if(M->getPrimalSolutionStatus() != SolutionStatus::Optimal) return identity(d);

            if(DEBUG) std::cout << "Solved" << std::endl;
            auto sol = *(A->level());
            matrix_type matrix_sol(d, row_type(d, 0));
            for(int row = 0; row < d; ++row) {
                for(int col = 0; col < d; ++col) {
                    matrix_sol[row][col] = sol[row * d + col];
                }
            }
            return matrix_sol;
        } catch(std::exception& e){
            std::cout << e.what() << std::endl;
            return matrix_type();

        }
}

float_type cost_fn(
        const matrix_type& A
        ) {
    if(A.size() == 0) return -RAND_MAX;
    unsigned_type d = A.size();
    auto ei = get_rand_vector_r(d);

    auto right = cpu_mmult(A, ei);
    float_type acc = 0.0;
    for(int i = 0; i < right.size(); ++i) acc += right[i] * ei[i];

    return acc;
}

std::pair<h_prime_type, float_type> compBasis (
        h_prime_type B,
        const float_type cost,
        const float_type u,
        const float_type l,
        const unsigned_type d
        ) {
    unsigned_type len_constraints = B["S"].size() + B["D"].size();
    if(len_constraints > d * d) {
        // This should return all possible combinations of length len_constraints - 1
        // Meaning that i need to iterate among all constraints and remove one at the time

        for(unsigned_type i = 0; i < len_constraints; ++i){
            // Try to remove the i-th constraint in S / D
            // -> Copy all elements of vector skipping the i-th index
            std::vector<constraint_type> slice;
            h_prime_type tmp_B;

            int cur_idx = (B["S"].size() >= i ? (len_constraints - B["S"].size()) : i);
            bool in_s   = (B["S"].size() < i);

            if(in_s) {
                slice = B["S"];
                slice.erase(slice.begin() + cur_idx);
                tmp_B["S"] = slice;
                tmp_B["D"] = B["D"];
            } else {
                slice = B["D"];
                slice.erase(slice.begin() + cur_idx);
                tmp_B["D"] = slice;
                tmp_B["S"] = B["S"];
            }

            auto A = semidefsolver(tmp_B, u, l);
            auto cost_1 = cost_fn(A);

            if(cost_1 >= cost) {
                auto new_results = compBasis(tmp_B, cost_1, u, l, d);
                auto A2 = semidefsolver(new_results.first, u, l);
                auto cost_2 = cost_fn(A2);

                if(cost_2 >= cost_1) return std::make_pair(new_results.first, cost_2);
                else return std::make_pair(tmp_B, cost_1);
            }
        }
    }

    return std::make_pair(B, cost);
}

std::pair<h_prime_type, matrix_type> pivot_LPType(
        h_prime_type &B,
        const h_prime_type& C,
        const float_type u,
        const float_type l,
        const unsigned_type d,
        const unsigned_type last_cost,
        const bool use_last_cost,
        const matrix_type& basis_A,
        const bool DEBUG = false
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
        auto ABh = semidefsolver(Bh, u, l);
        if(ABh.size() == 0) return std::make_pair(h_prime_type(), matrix_type());

        float_type Bh_cost = cost_fn(ABh);

        if(DEBUG) std::cout << "cost is = " << Bh_cost << std::endl;

        float_type B_cost = 0.0;
        if(calculate_basis_cost) {
            A = semidefsolver(B, u, l);
            if(A.size() == 0) return std::make_pair(h_prime_type(), matrix_type());
            B_cost = cost_fn(A);
            if(DEBUG) std::cout << "cost of basis = " << B_cost << std::endl;
        } else {
            B_cost = current_basis_cost;
        }

        if(Bh_cost > B_cost){

            auto tz = compBasis(Bh, Bh_cost, u, l, d);
            auto T = tz.first;
            auto z = tz.second;

            auto BA = pivot_LPType(T, Buc, u, l, d, Bh_cost, true, ABh);
            B = BA.first;
            A = BA.second;

            if(A.size() == 0) return std::make_pair(h_prime_type(), matrix_type());
            calculate_basis_cost = false;
            current_basis_cost = cost_fn(A);

        } else  {
            calculate_basis_cost = false;
            current_basis_cost = B_cost;
        }

    }
    return std::make_pair(B, A);
}

matrix_type learn_metric (
        const pair_type& S,
        const pair_type& D,
        const float_type u,
        const float_type l,
        const unsigned_type t,
        matrix_type initial_solution = matrix_type(),
        const bool DEBUG = false
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

    unsigned_type max_best_solution_d = viol_d;
    unsigned_type max_best_solution_s = viol_s;
    matrix_type best_A = identity(dimension);

    const unsigned_type initial_violation_count = viol_d + viol_s;

    for(int i = 0; i < t; ++i) {

        if(DEBUG) std::cout << "=========================ITERATION: " << i << "=========================" << std::endl;

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

        auto BnA = pivot_LPType(B0, R, u, l, dimension, 0, false, matrix_type());
        auto Bm = BnA.first;
        auto A = BnA.second;

        auto violated_constraints = count_violated_constraints_SD(S, D, cholesky_transformer(A), u, l);
        unsigned_type violated_constraints_d = violated_constraints.first;
        unsigned_type violated_constraints_s = violated_constraints.second;

        if((max_best_solution_d + max_best_solution_s) >= (violated_constraints_d + violated_constraints_s)){
            best_A = A;
            max_best_solution_d = violated_constraints_d;
            max_best_solution_s = violated_constraints_s;
        }

        if(violated_constraints_d + violated_constraints_s == 0) break;

    }

    return best_A;
}


matrix_type fit(
        const matrix_type& x,                          // Dataset
        const label_row_type& y,                       // Labels
        const float_type& u,                           // Upper bound
        const float_type& l,                           // Lower bound
        const size_t& DIM_Y,                           // Number of datapoints
        const size_t& DIM_X,                           // Number of features
        const pair_index_type& all_pairs,              // Combinations of all pairs in the dataset
        matrix_type initial_solution = matrix_type(), // Initial solution
        const bool DEBUG = false
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
    auto new_G = learn_metric(similar_pairs_S, dissimilar_pairs_D, u, l, 20, initial_solution);
    return new_G;
}
template <typename T>
unsigned_type argmax(
        const std::vector<T>& v
        ){

    unsigned_type ret = 0;
    for (int i = 0; i < v.size(); ++i) {
        if(v[ret] < v[i]) ret = i;
    }
    return ret;
}

template <typename DistFnType, unsigned_type N_LABELS = 3>
label_row_type knn(
        const matrix_type& x_train,
        const label_row_type& y_train,
        const matrix_type& x_test,
        const unsigned_type& num_neighbours,
        const DistFnType dist_fn
        ) {

    label_row_type res;

    // for every point in the test set
    for (unsigned_type i = 0; i < x_test.size(); ++i) {
        auto curr_test_row = x_test[i];
        // All distances
        std::vector<std::pair<unsigned_type, float_type>> distances(x_train.size());

        // calculate the distance between the current test point and all elements in the dataset.
        for (unsigned_type j = 0; j < distances.size(); ++j) {
            auto cur_train_row = x_train[j];
            distances[j] = std::make_pair(j, dist_fn(cur_train_row, curr_test_row));
        }

        std::sort(distances.begin(), distances.end(), [](std::pair<unsigned_type, float_type> l, std::pair<unsigned_type, float_type> r){
            return l.second < r.second;
        });

        distances.resize(num_neighbours);

        std::vector<unsigned_type> class_count(N_LABELS, 0);

        for(unsigned_type j = 0; j < class_count.size(); ++j){
            const unsigned_type index = distances[j].first;
            const unsigned_type curr_class = y_train[index];
            class_count[curr_class]++;
        }
        res.push_back(argmax(class_count));
    }
    return res;
}

float_type predict(
        const matrix_type& x_train,
        const label_row_type& y_train,
        const matrix_type& x_test,
        const label_row_type& y_test
        ) {

    auto results = knn(x_train, y_train, x_test, 3, [](row_type a, row_type b) -> float_type {
        float_type acc = 0.0;
        for(int i = 0; i < a.size();++i){
            float_type tmp = a[i] - b[i];
            tmp *= tmp;
            tmp = sqrt(tmp);
            acc += tmp;
        }
        return acc;
    });

    int errs = 0;
    for(int i = 0; i < results.size(); ++i) {
        int result = results[i];
        int test = y_test[i];
        if(results[i] != y_test[i]) errs ++;
    }

    return 1 - (float_type) errs / results.size();

}

#endif //LPTML_LPTML_H
