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

__global__ void
kernel(const uint16_t *graph, LinApprox16::output_t *output)
{
    int hyper0 = threadIdx.x + blockIdx.x * blockDim.x;
    assert(0 <= hyper0 && hyper0 < 65536);

    int best_hyper1 = 0;
    int best_value = 0;
    for (int hyper1 = 1; hyper1 < 65536; ++hyper1)
    {
        int val = check(graph, hyper0, hyper1);
        if (val > best_value && hyper0 != hyper1)
        {
            best_value = val;
            best_hyper1 = hyper1;
        }
    }

    output[hyper0] = {.hyper1 = static_cast<uint16_t>(best_hyper1),
                      .value = static_cast<uint16_t>(best_value)};
}

LinApprox16::LinApprox16() : output(65536)
{
    cudaMalloc(&cuda_graph, 256 * sizeof(uint16_t));
    cudaCheckError();

    cudaMalloc(&cuda_output, 65536 * sizeof(output_t));
    cudaCheckError();
}

std::tuple<int, uint16_t, uint16_t> LinApprox16::evaluate(const std::vector<uint16_t> &graph)
{
    assert(graph.size() == 256);

    cudaMemcpy(cuda_graph, graph.data(), 256 * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaCheckError();

    kernel<<<256, 256>>>(cuda_graph, cuda_output);
    cudaCheckError();

    cudaMemcpy(output.data(), cuda_output, 65536 * sizeof(output_t), cudaMemcpyDeviceToHost);
    cudaCheckError();

    int best_hyper0 = -1;
    int best_hyper1 = -1;
    int best_value = -1;
    for (int hyper0 = 1; hyper0 < 65536; ++hyper0)
    {
        if (output[hyper0].value > best_value)
        {
            best_hyper0 = hyper0;
            best_hyper1 = output[hyper0].hyper1;
            best_value = output[hyper0].value;
        }
    }

    return std::make_tuple(best_value, best_hyper0, best_hyper1);
}
