//
// Created by francesco on 12/30/19.
//

#ifndef LPTML_CUDA_CUDA_WRAPPER_CUH
#define LPTML_CUDA_CUDA_WRAPPER_CUH

#include <algorithm>
#include "types.h"
#include "utils.h"
#include "lptml.cuh"

#define BLOCKS 4
#define THREADS_PER_BLOCK 256

void cuda_get_bounds(
        float_type *u,
        float_type *l,
        pair_index_type combinations,
        matrix_type dataset
) {

    std::vector<float_type> unrolled_dataset;
    std::vector<float_type> distances(combinations.size() * 2, 0.0);

    float_type *d_unrolled_dataset;
    float_type *d_distances;


    // Unroll the dataset according to the combinations of indeces
    for (auto combination: combinations) {
        auto first = combination.first;
        auto second = combination.second;

        auto row_1 = dataset[first];
        auto row_2 = dataset[second];

        for (int i = 0; i < row_1.size(); ++i) {
            unrolled_dataset.push_back(row_1[i]);
        }
        for (int i = 0; i < row_2.size(); ++i) {
            unrolled_dataset.push_back(row_2[i]);
        }

    }

    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_unrolled_dataset, sizeof(float_type) * unrolled_dataset.size()));
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_distances, sizeof(float_type) * unrolled_dataset.size()));

    CUDA_CHECK_ERROR(cudaMemcpy(d_unrolled_dataset, &unrolled_dataset[0], sizeof(float_type) * unrolled_dataset.size(),
                                cudaMemcpyHostToDevice));

    // Execute the kernel
    pairwise_norm << < BLOCKS, THREADS_PER_BLOCK >> > (d_distances, d_unrolled_dataset, combinations.size() * 2);

    CUDA_CHECK_ERROR(cudaMemcpy(&distances[0], d_distances, sizeof(float_type) * combinations.size() * 2,
                                cudaMemcpyDeviceToHost));

    *u = percentile(distances, 10, distances.size());
    *l = percentile(distances, 90, distances.size());

    CUDA_CHECK_ERROR(cudaFree(d_unrolled_dataset));
    CUDA_CHECK_ERROR(cudaFree(d_distances));
}

std::pair<unsigned_type, unsigned_type> cuda_reduce_violated_constraints(
        unsigned_type *d_distance_bitmap_S,
        unsigned_type *d_distance_bitmap_D,
        cudaStream_t stream_S,
        cudaStream_t stream_D,
        size_t size_s,
        size_t size_d
) {
    std::vector<unsigned_type> d_count(size_d);
    std::vector<unsigned_type> s_count(size_s);
    unsigned_type result_d;
    unsigned_type result_s;

    reduce << < BLOCKS, size_s / 2, 0, stream_S >> > (d_distance_bitmap_S);
    reduce << < BLOCKS, size_d / 2, 0, stream_D >> > (d_distance_bitmap_D);

    CUDA_CHECK_ERROR(
            cudaMemcpyAsync(&result_d, d_distance_bitmap_D, sizeof(unsigned_type), cudaMemcpyDeviceToHost, stream_D));
    CUDA_CHECK_ERROR(
            cudaMemcpyAsync(&result_s, d_distance_bitmap_S, sizeof(unsigned_type), cudaMemcpyDeviceToHost, stream_S));

    cudaStreamSynchronize(stream_S);
    cudaStreamSynchronize(stream_D);

    return std::make_pair(result_d, result_s);

}

std::pair<unsigned_type, unsigned_type> cuda_count_violated_constraints_SD(
        const pair_type& S,
        const pair_type& D,
        matrix_type &G,
        const float_type& u,
        const float_type& l,
        const bool DEBUG = false
    ){

    float_type *d_unwrapped_S;
    float_type *d_unwrapped_D;
    float_type *d_unwrapped_G;
    unsigned_type *d_distance_bitmap_S;
    unsigned_type *d_distance_bitmap_D;

    cudaStream_t stream_S;
    cudaStream_t stream_D;
    cudaStream_t stream_G;

    cudaStreamCreate(&stream_S);
    cudaStreamCreate(&stream_D);
    cudaStreamCreate(&stream_G);

    if (G.size() == 0) G = identity(4);

    auto unwrapped_S = unwrap_constraint<float_type>(S);
    auto unwrapped_D = unwrap_constraint<float_type>(D);
    auto unwrapped_G = unwrap_matrix<float_type>(G);

    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_unwrapped_D, sizeof(float_type) * unwrapped_D.size()));
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_unwrapped_S, sizeof(float_type) * unwrapped_S.size()));
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_unwrapped_G, sizeof(float_type) * unwrapped_G.size()));
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_distance_bitmap_D, sizeof(unsigned_type) * S.size()));
    CUDA_CHECK_ERROR(cudaMalloc((void **) &d_distance_bitmap_S, sizeof(unsigned_type) * D.size()));

    CUDA_CHECK_ERROR(
            cudaMemcpyAsync(d_unwrapped_D, &unwrapped_D[0], sizeof(float_type) * unwrapped_D.size(), cudaMemcpyHostToDevice,
                            stream_D));
    CUDA_CHECK_ERROR(
            cudaMemcpyAsync(d_unwrapped_S, &unwrapped_S[0], sizeof(float_type) * unwrapped_S.size(), cudaMemcpyHostToDevice,
                            stream_S));
    CUDA_CHECK_ERROR(
            cudaMemcpyAsync(d_unwrapped_G, &unwrapped_G[0], sizeof(float_type) * unwrapped_G.size(), cudaMemcpyHostToDevice,
                            stream_G));

    // Sync all streams until the data transfer is complete
    cudaStreamSynchronize(stream_S);
    cudaStreamSynchronize(stream_D);
    cudaStreamSynchronize(stream_G);

    // count_violated_constraints_D();
    count_violated_constraints_S<<<BLOCKS, THREADS_PER_BLOCK, 0, stream_S>>>(d_distance_bitmap_S, d_unwrapped_S, u, d_unwrapped_G, unwrapped_S.size());
    count_violated_constraints_D<<<BLOCKS, THREADS_PER_BLOCK, 0, stream_S>>>(d_distance_bitmap_D, d_unwrapped_S, l, d_unwrapped_G, unwrapped_S.size());

    auto count = cuda_reduce_violated_constraints(d_distance_bitmap_S, d_distance_bitmap_D, stream_S, stream_D, S.size(), D.size());

    cudaStreamSynchronize(stream_S);
    cudaStreamSynchronize(stream_D);

    CUDA_CHECK_ERROR(cudaFree(d_unwrapped_D));
    CUDA_CHECK_ERROR(cudaFree(d_unwrapped_S));
    CUDA_CHECK_ERROR(cudaFree(d_unwrapped_G));
    CUDA_CHECK_ERROR(cudaFree(d_distance_bitmap_S));
    CUDA_CHECK_ERROR(cudaFree(d_distance_bitmap_D));

    CUDA_CHECK_ERROR(cudaStreamDestroy(stream_S));
    CUDA_CHECK_ERROR(cudaStreamDestroy(stream_D));
    CUDA_CHECK_ERROR(cudaStreamDestroy(stream_G));

if(DEBUG){
        std::cout << "Size S = " << unwrapped_S.size()<< "\nSize D = " << unwrapped_D.size()<<std::endl;
    }
    return count;
}

#endif //LPTML_CUDA_CUDA_WRAPPER_CUH
