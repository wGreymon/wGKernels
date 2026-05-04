#include "wgkernel/cuda/error_check.hpp"
#include "wgkernel/cuda/reduce.hpp"
#include "wgkernel/cuda/timer.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string op = "sum";
    std::int64_t numel = 1 << 24;
    int warmup = 10;
    int repeat = 100;
};

std::vector<float> make_input(const std::int64_t numel) {
    std::vector<float> values(static_cast<std::size_t>(numel));
    for (std::int64_t index = 0; index < numel; ++index) {
        const std::int64_t periodic = (index * 17) % 97;
        values[static_cast<std::size_t>(index)] =
            static_cast<float>(periodic - 48) * 0.125f +
            static_cast<float>(index % 13) * 0.01f;
    }
    return values;
}

Options parse_options(const int argc, char** argv) {
    Options options;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--op" && index + 1 < argc) {
            options.op = argv[++index];
        } else if (argument == "--numel" && index + 1 < argc) {
            options.numel = std::stoll(argv[++index]);
        } else if (argument == "--warmup" && index + 1 < argc) {
            options.warmup = std::stoi(argv[++index]);
        } else if (argument == "--repeat" && index + 1 < argc) {
            options.repeat = std::stoi(argv[++index]);
        } else {
            throw std::runtime_error("Unknown argument: " + argument);
        }
    }

    if (options.numel <= 0 || options.warmup < 0 || options.repeat <= 0) {
        throw std::runtime_error("Invalid benchmark arguments");
    }

    return options;
}

float benchmark_sum(const Options& options, const std::vector<float>& input) {
    float* device_input = nullptr;
    float* device_output = nullptr;
    void* workspace = nullptr;
    const auto workspace_bytes = wgkernel::cuda::reduce_sum_workspace_size(options.numel);

    WGKERNEL_CUDA_CHECK(cudaMalloc(&device_input, sizeof(float) * input.size()));
    WGKERNEL_CUDA_CHECK(cudaMalloc(&device_output, sizeof(float)));
    if (workspace_bytes > 0) {
        WGKERNEL_CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));
    }

    WGKERNEL_CUDA_CHECK(cudaMemcpy(
        device_input,
        input.data(),
        sizeof(float) * input.size(),
        cudaMemcpyHostToDevice));

    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        WGKERNEL_CUDA_CHECK(wgkernel::cuda::reduce_sum(
            device_input,
            device_output,
            options.numel,
            workspace,
            workspace_bytes));
    }
    WGKERNEL_CUDA_CHECK(cudaDeviceSynchronize());

    wgkernel::cuda::CudaEventTimer timer;
    timer.start();
    for (int iteration = 0; iteration < options.repeat; ++iteration) {
        WGKERNEL_CUDA_CHECK(wgkernel::cuda::reduce_sum(
            device_input,
            device_output,
            options.numel,
            workspace,
            workspace_bytes));
    }
    const float elapsed_ms = timer.stop() / static_cast<float>(options.repeat);

    cudaFree(workspace);
    cudaFree(device_output);
    cudaFree(device_input);
    return elapsed_ms;
}

float benchmark_max(const Options& options, const std::vector<float>& input) {
    float* device_input = nullptr;
    float* device_output = nullptr;
    void* workspace = nullptr;
    const auto workspace_bytes = wgkernel::cuda::reduce_max_workspace_size(options.numel);

    WGKERNEL_CUDA_CHECK(cudaMalloc(&device_input, sizeof(float) * input.size()));
    WGKERNEL_CUDA_CHECK(cudaMalloc(&device_output, sizeof(float)));
    if (workspace_bytes > 0) {
        WGKERNEL_CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));
    }

    WGKERNEL_CUDA_CHECK(cudaMemcpy(
        device_input,
        input.data(),
        sizeof(float) * input.size(),
        cudaMemcpyHostToDevice));

    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        WGKERNEL_CUDA_CHECK(wgkernel::cuda::reduce_max(
            device_input,
            device_output,
            options.numel,
            workspace,
            workspace_bytes));
    }
    WGKERNEL_CUDA_CHECK(cudaDeviceSynchronize());

    wgkernel::cuda::CudaEventTimer timer;
    timer.start();
    for (int iteration = 0; iteration < options.repeat; ++iteration) {
        WGKERNEL_CUDA_CHECK(wgkernel::cuda::reduce_max(
            device_input,
            device_output,
            options.numel,
            workspace,
            workspace_bytes));
    }
    const float elapsed_ms = timer.stop() / static_cast<float>(options.repeat);

    cudaFree(workspace);
    cudaFree(device_output);
    cudaFree(device_input);
    return elapsed_ms;
}

float benchmark_argmax(const Options& options, const std::vector<float>& input) {
    float* device_input = nullptr;
    std::int64_t* device_output = nullptr;
    void* workspace = nullptr;
    const auto workspace_bytes = wgkernel::cuda::reduce_argmax_workspace_size(options.numel);

    WGKERNEL_CUDA_CHECK(cudaMalloc(&device_input, sizeof(float) * input.size()));
    WGKERNEL_CUDA_CHECK(cudaMalloc(&device_output, sizeof(std::int64_t)));
    WGKERNEL_CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));

    WGKERNEL_CUDA_CHECK(cudaMemcpy(
        device_input,
        input.data(),
        sizeof(float) * input.size(),
        cudaMemcpyHostToDevice));

    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        WGKERNEL_CUDA_CHECK(wgkernel::cuda::reduce_argmax(
            device_input,
            device_output,
            options.numel,
            workspace,
            workspace_bytes));
    }
    WGKERNEL_CUDA_CHECK(cudaDeviceSynchronize());

    wgkernel::cuda::CudaEventTimer timer;
    timer.start();
    for (int iteration = 0; iteration < options.repeat; ++iteration) {
        WGKERNEL_CUDA_CHECK(wgkernel::cuda::reduce_argmax(
            device_input,
            device_output,
            options.numel,
            workspace,
            workspace_bytes));
    }
    const float elapsed_ms = timer.stop() / static_cast<float>(options.repeat);

    cudaFree(workspace);
    cudaFree(device_output);
    cudaFree(device_input);
    return elapsed_ms;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto input = make_input(options.numel);

        float latency_ms = 0.0f;
        if (options.op == "sum") {
            latency_ms = benchmark_sum(options, input);
        } else if (options.op == "max") {
            latency_ms = benchmark_max(options, input);
        } else if (options.op == "argmax") {
            latency_ms = benchmark_argmax(options, input);
        } else {
            throw std::runtime_error("Unsupported op: " + options.op);
        }

        const double bytes = static_cast<double>(options.numel) * sizeof(float);
        const double bandwidth_gb_s = bytes / (static_cast<double>(latency_ms) * 1.0e6);

        std::cout << std::fixed << std::setprecision(4)
                  << "benchmark op=" << options.op
                  << " numel=" << options.numel
                  << " warmup=" << options.warmup
                  << " repeat=" << options.repeat
                  << " latency_ms=" << latency_ms
                  << " bandwidth_gb_s=" << bandwidth_gb_s << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
