#include "wgkernel/cuda/error_check.hpp"
#include "wgkernel/cuda/reduce.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    bool single_run = false;
    std::string op;
    std::int64_t numel = 0;
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
            options.single_run = true;
            options.op = argv[++index];
        } else if (argument == "--numel" && index + 1 < argc) {
            options.numel = std::stoll(argv[++index]);
        } else {
            throw std::runtime_error("Unknown argument: " + argument);
        }
    }

    if (options.single_run && (options.op.empty() || options.numel <= 0)) {
        throw std::runtime_error("Single run mode requires --op and --numel");
    }

    return options;
}

float cpu_sum(const std::vector<float>& input) {
    double accumulator = 0.0;
    for (const float value : input) {
        accumulator += value;
    }
    return static_cast<float>(accumulator);
}

float cpu_max(const std::vector<float>& input) {
    return *std::max_element(input.begin(), input.end());
}

std::int64_t cpu_argmax(const std::vector<float>& input) {
    std::int64_t best_index = 0;
    float best_value = input.front();

    for (std::int64_t index = 1; index < static_cast<std::int64_t>(input.size()); ++index) {
        const float value = input[static_cast<std::size_t>(index)];
        if (value > best_value) {
            best_value = value;
            best_index = index;
        }
    }

    return best_index;
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

bool check_close(const float actual, const float expected, const float atol, const float rtol) {
    return std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

int run_self_test() {
    const std::vector<std::int64_t> shapes { 1, 7, 257, 4099, 65536, 1048576 };

    for (const std::int64_t numel : shapes) {
        const auto input = make_input(numel);

        const float actual_sum = run_sum(input);
        const float expected_sum = cpu_sum(input);
        if (!check_close(actual_sum, expected_sum, 1e-2f, 1e-4f)) {
            std::cerr << "sum check failed for numel=" << numel << ", actual=" << actual_sum
                      << ", expected=" << expected_sum << "\n";
            return 1;
        }

        const float actual_max = run_max(input);
        const float expected_max = cpu_max(input);
        if (!check_close(actual_max, expected_max, 1e-6f, 1e-6f)) {
            std::cerr << "max check failed for numel=" << numel << ", actual=" << actual_max
                      << ", expected=" << expected_max << "\n";
            return 1;
        }

        const std::int64_t actual_argmax = run_argmax(input);
        const std::int64_t expected_argmax = cpu_argmax(input);
        if (actual_argmax != expected_argmax) {
            std::cerr << "argmax check failed for numel=" << numel << ", actual=" << actual_argmax
                      << ", expected=" << expected_argmax << "\n";
            return 1;
        }
    }

    std::cout << "reduce correctness self-test passed\n";
    return 0;
}

int run_single(const Options& options) {
    const auto input = make_input(options.numel);

    std::cout << std::fixed << std::setprecision(10);
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
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.single_run) {
            return run_single(options);
        }
        return run_self_test();
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
