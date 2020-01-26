//
// Created by francesco on 12/23/19.
//
#include "types.h"
#include "../clion_utils.h"

#ifndef LPTML_LPTML_CUH
#define LPTML_LPTML_CUH

template<typename T, unsigned_type column_count = 4>
__device__
__forceinline__
        T
l2norm(T
*row1,
T *row2
){
T acc = 0.0;
for (
int i = 0;
i<column_count;
++i) {
T tmp = (row1[i] - row2[i]);
acc +=
tmp *tmp;
}
return
sqrt(acc);
}

// Note, for obvious reasons res must be PAIR_DIM in size...
template<typename T, unsigned_type column_count = 4>
__global__
void pairwise_norm(T *res, T *data, const size_t PAIR_DIM) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    // Change it with a grid stride loop
    // Could be done something like
    // `if(tid % 2 == 0) do l2norm`
    // But that could lead to warped divergence
    // and load unbalance between threads in the warp
    for (int i = tid; i < PAIR_DIM - 1; i += (2 + stride)) {
        T *row1 = &data[i];
        T *row2 = &data[i + column_count];
        res[i] = l2norm(row1, row2);
    }

}

template<typename T, unsigned_type column_count = 4>
__device__
void mmult(T *M, T *b, T result[column_count]) {

    for (int i = 0; i < column_count; i += 1) {
        for (int j = 0; j < column_count; ++j) {
            result[i] += M[i * column_count + j] * b[j];
        }
    }

}

template<typename T, unsigned_type column_count = 4>
__device__
void difference(T result[column_count], T by[column_count]) {
    for (int i = 0; i < column_count; ++i) {
        result[i] -= by[i];
    }
}

template<typename T, unsigned_type column_count = 4>
__global__
void count_violated_constraints_S(unsigned_type *distance_bitmap, T *constraints, T u, T *G, const size_t DIM) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid % (column_count * 2) == 0) {
        for (int i = tid; i < DIM; i += (column_count * 2)) {
            T *c1 = &constraints[i];
            T *c2 = &constraints[i + column_count];

            T p1[column_count];
            T p2[column_count];
            for (int j = 0; j < column_count; ++j) {
                p1[j] = 0;
                p2[j] = 0;
            }

            mmult(G, c1, p1);
            mmult(G, c2, p2);

            T val = l2norm(p1, p2);

            distance_bitmap[i / (column_count * 2)] = val > u;
        }
    }

}

template<typename T, unsigned_type column_count = 4>
__global__
void count_violated_constraints_D(unsigned_type *distance_bitmap, T *constraints, T l, T *G, const size_t DIM) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid % (column_count * 2) != 0) {
        for (int i = tid; i < DIM; i += (column_count * 2)) {
            T *c1 = &constraints[i];
            T *c2 = &constraints[i + column_count];

            T p1[column_count];
            T p2[column_count];
            for (int j = 0; j < column_count; ++j) {
                p1[j] = 0;
                p2[j] = 0;
            }

            mmult(G, c1, p1);
            mmult(G, c2, p2);

            T val = l2norm(p1, p2);

            distance_bitmap[i / (column_count * 2)] = val < l;
        }
    }

}

template<size_t blockSize, typename T>
__device__ void warpReduce(volatile T *sdata, size_t tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}


template<typename T>
__global__ void reduce(T *input) {
    const int tid = threadIdx.x;

    auto step_size = 1;
    int number_of_threads = blockDim.x;

    while (number_of_threads > 0) {
        if (tid < number_of_threads) // still alive?
        {
            const auto fst = tid * step_size * 2;
            const auto snd = fst + step_size;
            input[fst] += input[snd];
        }

        step_size <<= 1;
        number_of_threads >>= 1;
    }
}

#endif //LPTML_LPTML_CUH
