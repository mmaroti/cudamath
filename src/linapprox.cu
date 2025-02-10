#include <iostream>
#include <cassert>
#include "linapprox.hpp"

void cudaCheckError()
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        std::exit(1);
    }
}

__device__ int
check(const uint16_t *graph, uint16_t hyper0, uint16_t hyper1)
{
    int counts[4] = {0, 0, 0, 0};
    for (int i = 0; i < 256; ++i)
    {
        uint16_t point = graph[i];
        int count0 = __popc(hyper0 & point) & 1;
        int count1 = __popc(hyper1 & point) & 1;
        counts[(count1 << 1) + count0] += 1;
    }
    int a = counts[0] > counts[1] ? counts[0] : counts[1];
    int b = counts[2] > counts[3] ? counts[2] : counts[3];
    return a > b ? a : b;
}

/**
 * Choice of (hyper0,hyper1) pairs we need to check for n=8. Notice, that
 * hyper values cannot be zero (not a proper hyperplane), and must have
 * distinct values. We pair up the first and last rows in index1.
 *
 * 21                   index0 = 0, index1 = 0
 * 31 32                index0 = 1, index1 = 0, 1
 * 41 42 43             index0 = 2, index1 = 0, 1, 2
 * 51 52 53 54          index0 = 2, index1 = 6, 5, 4, 3
 * 61 62 63 64 65       index0 = 1, index1 = 6, 5, 4, 3, 2
 * 71 72 73 74 75 76    index0 = 0, index1 = 6, 5, 4, 3, 2, 1
 */

__device__ __host__ static inline int get_hyper0(int index0, int index1)
{
    return index1 <= index0 ? index0 + 2 : (65536 - 1) - index0;
}

__device__ __host__ static inline int get_hyper1(int index0, int index1)
{
    return index1 <= index0 ? index1 + 1 : (65536 - 1) - index1;
}

__global__ void
kernel(const uint16_t *graph, LinApprox16::output_t *output)
{
    int index0 = threadIdx.x + blockIdx.x * blockDim.x;
    if (index0 >= 32768 - 1)
        return;

    int best_index1 = -1;
    int best_value = -1;
    for (int index1 = 0; index1 < 65536 - 1; ++index1)
    {
        int hyper0 = get_hyper0(index0, index1);
        int hyper1 = get_hyper1(index0, index1);

        int value = check(graph, hyper0, hyper1);
        if (value > best_value)
        {
            best_value = value;
            best_index1 = index1;
        }
    }

    output[index0] = {.index1 = static_cast<uint16_t>(best_index1),
                      .value = static_cast<uint16_t>(best_value)};
}

LinApprox16::LinApprox16() : output(65536)
{
    cudaMalloc(&cuda_graph, 256 * sizeof(uint16_t));
    cudaCheckError();

    cudaMalloc(&cuda_output, 32768 * sizeof(output_t));
    cudaCheckError();
}

int slow_popcount(int val)
{
    int c = 0;
    while (val != 0)
    {
        c += 1;
        val &= val - 1;
    }
    return c;
}

std::tuple<int, uint16_t, uint16_t> LinApprox16::evaluate(const std::vector<uint16_t> &graph)
{
    assert(graph.size() == 256);

    cudaMemcpy(cuda_graph, graph.data(), 256 * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaCheckError();

    kernel<<<128, 256>>>(cuda_graph, cuda_output);
    cudaCheckError();

    cudaMemcpy(output.data(), cuda_output, 32768 * sizeof(output_t), cudaMemcpyDeviceToHost);
    cudaCheckError();

    int best_index0 = -1;
    int best_index1 = -1;
    int best_value = -1;
    for (int index0 = 0; index0 < 32768 - 1; ++index0)
    {
        if (output[index0].value > best_value)
        {
            best_index0 = index0;
            best_index1 = output[index0].index1;
            best_value = output[index0].value;
        }
    }

    int hyper0 = get_hyper0(best_index0, best_index1);
    int hyper1 = get_hyper1(best_index0, best_index1);

    if (false)
    {
        int counts[4] = {0, 0, 0, 0};
        for (int i = 0; i < 256; ++i)
        {
            uint16_t point = graph[i];
            int count0 = slow_popcount(hyper0 & point) & 1;
            int count1 = slow_popcount(hyper1 & point) & 1;
            counts[(count1 << 1) + count0] += 1;
        }
        std::cout << counts[0] << " " << counts[1] << " " << counts[2] << " " << counts[3] << std::endl;
    }

    return std::make_tuple(best_value, hyper0, hyper1);
}
