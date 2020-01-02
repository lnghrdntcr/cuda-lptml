//
// Created by francesco on 12/23/19.
//
#include "types.h"
#include "../clion_utils.h"
#ifndef LPTML_LPTML_CUH
#define LPTML_LPTML_CUH

template <typename T, unsigned_type column_count = 4>
__device__
__forceinline__
T l2norm(T *row1, T *row2){
    T acc = 0.0;
    for (int i = 0; i < column_count; ++i) {
        T tmp = (row1[i] - row2[i]);
        acc += tmp * tmp;
    }
    return sqrt(acc);
}

// Note, for obvious reasons res must be PAIR_DIM in size...
template <typename T, unsigned_type column_count = 4>
__global__
void pairwise_norm(T *res, T *data, const size_t PAIR_DIM){
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    // Change it with a grid stride loop
    // Could be done something like
    // `if(tid % 2 == 0) do l2norm`
    // But that could lead to warped divergence
    // and load unbalance between threads in the warp
    for (int i = tid; i < PAIR_DIM - 1; i += (2 + stride)) {
        T* row1 = &data[i];
        T* row2 = &data[i + column_count];
        res[i] = l2norm(row1, row2);
    }

}

template <typename T, unsigned_type column_count = 4>
__device__
T mahalanobis_distance_kernel(T *p1, T *p2, T *reference){

}

#endif //LPTML_LPTML_CUH
