#include "wgkernel/cuda/error_check.hpp"
#include "cuda/reduce.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string op;
    std::int64_t numel = 0;
    std::uint64_t seed = 0;
};

std::vector<float> make_input(const std::int64_t numel, const std::uint64_t seed) {
    std::vector<float> values(static_cast<std::size_t>(numel));
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (std::int64_t index = 0; index < numel; ++index) {
        values[static_cast<std::size_t>(index)] = distribution(generator);
    }
    return values;
}

Options parse_options(const int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next = [&]() -> const char* {
            if (index + 1 >= argc) {
                throw std::runtime_error("Missing value for argument: " + argument);
            }
            return argv[++index];
        };

        if (argument == "--op") {
            options.op = next();
        } else if (argument == "--numel") {
            options.numel = std::stoll(next());
        } else if (argument == "--seed") {
            options.seed = std::stoull(next());
        } else {
            throw std::runtime_error("Unknown argument: " + argument);
        }
    }

    if (options.op.empty() || options.numel <= 0) {
        throw std::runtime_error("Reduce tool requires --op and --numel");
    }
    return options;
}

float run_sum(const std::vector<float>& input) {
    float* device_input = nullptr;
    float* device_output = nullptr;
    void* workspace = nullptr;

    const auto workspace_bytes = wgkernel::cuda::reduce_sum_workspace_size(static_cast<std::int64_t>(input.size()));
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
    WGKERNEL_CUDA_CHECK(wgkernel::cuda::reduce_sum(
        device_input,
        device_output,
        static_cast<std::int64_t>(input.size()),
        workspace,
        workspace_bytes));
    WGKERNEL_CUDA_CHECK(cudaDeviceSynchronize());

    float result = 0.0f;
    WGKERNEL_CUDA_CHECK(cudaMemcpy(&result, device_output, sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(workspace);
    cudaFree(device_output);
    cudaFree(device_input);
    return result;
}

float run_max(const std::vector<float>& input) {
    float* device_input = nullptr;
    float* device_output = nullptr;
    void* workspace = nullptr;

    const auto workspace_bytes = wgkernel::cuda::reduce_max_workspace_size(static_cast<std::int64_t>(input.size()));
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
    WGKERNEL_CUDA_CHECK(wgkernel::cuda::reduce_max(
        device_input,
        device_output,
        static_cast<std::int64_t>(input.size()),
        workspace,
        workspace_bytes));
    WGKERNEL_CUDA_CHECK(cudaDeviceSynchronize());

    float result = 0.0f;
    WGKERNEL_CUDA_CHECK(cudaMemcpy(&result, device_output, sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(workspace);
    cudaFree(device_output);
    cudaFree(device_input);
    return result;
}

std::int64_t run_argmax(const std::vector<float>& input) {
    float* device_input = nullptr;
    std::int64_t* device_output = nullptr;
    void* workspace = nullptr;

    const auto workspace_bytes =
        wgkernel::cuda::reduce_argmax_workspace_size(static_cast<std::int64_t>(input.size()));
    WGKERNEL_CUDA_CHECK(cudaMalloc(&device_input, sizeof(float) * input.size()));
    WGKERNEL_CUDA_CHECK(cudaMalloc(&device_output, sizeof(std::int64_t)));
    WGKERNEL_CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));

    WGKERNEL_CUDA_CHECK(cudaMemcpy(
        device_input,
        input.data(),
        sizeof(float) * input.size(),
        cudaMemcpyHostToDevice));
    WGKERNEL_CUDA_CHECK(wgkernel::cuda::reduce_argmax(
        device_input,
        device_output,
        static_cast<std::int64_t>(input.size()),
        workspace,
        workspace_bytes));
    WGKERNEL_CUDA_CHECK(cudaDeviceSynchronize());

    std::int64_t result = -1;
    WGKERNEL_CUDA_CHECK(cudaMemcpy(&result, device_output, sizeof(std::int64_t), cudaMemcpyDeviceToHost));

    cudaFree(workspace);
    cudaFree(device_output);
    cudaFree(device_input);
    return result;
}

}  // namespace

// Single-run driver: print a single `result=<value>` line on stdout for the
// Python test harness to consume. Correctness and performance checks are
// owned by the Python test harness.
int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto input = make_input(options.numel, options.seed);

        std::cout << std::setprecision(10);
        if (options.op == "sum") {
            std::cout << "result=" << run_sum(input) << "\n";
            return 0;
        }
        if (options.op == "max") {
            std::cout << "result=" << run_max(input) << "\n";
            return 0;
        }
        if (options.op == "argmax") {
            std::cout << "result=" << run_argmax(input) << "\n";
            return 0;
        }
        throw std::runtime_error("Unsupported op: " + options.op);
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
