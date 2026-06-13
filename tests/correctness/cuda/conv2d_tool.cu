#include "wgkernel/cuda/error_check.hpp"
#include "cuda/convolution.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ConvParams {
    int n = 1;
    int c_in = 1;
    int h_in = 1;
    int w_in = 1;
    int c_out = 1;
    int k_h = 1;
    int k_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    int pad_h = 0;
    int pad_w = 0;
    int dilation_h = 1;
    int dilation_w = 1;
    int groups = 1;

    int h_out() const {
        return (h_in + 2 * pad_h - dilation_h * (k_h - 1) - 1) / stride_h + 1;
    }
    int w_out() const {
        return (w_in + 2 * pad_w - dilation_w * (k_w - 1) - 1) / stride_w + 1;
    }
    std::int64_t input_numel() const {
        return static_cast<std::int64_t>(n) * c_in * h_in * w_in;
    }
    std::int64_t weight_numel() const {
        return static_cast<std::int64_t>(c_out) * (c_in / groups) * k_h * k_w;
    }
    std::int64_t output_numel() const {
        return static_cast<std::int64_t>(n) * c_out * h_out() * w_out();
    }
};

struct Options {
    ConvParams params;
    std::string algo = "naive";
    std::string input_path;
    std::string weight_path;
    std::string bias_path;  // empty -> no bias
    std::string output_path;
};

int parse_int(const char* text) {
    return static_cast<int>(std::stoll(text));
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

        if (argument == "--n") {
            options.params.n = parse_int(next());
        } else if (argument == "--c_in") {
            options.params.c_in = parse_int(next());
        } else if (argument == "--h_in") {
            options.params.h_in = parse_int(next());
        } else if (argument == "--w_in") {
            options.params.w_in = parse_int(next());
        } else if (argument == "--c_out") {
            options.params.c_out = parse_int(next());
        } else if (argument == "--k_h") {
            options.params.k_h = parse_int(next());
        } else if (argument == "--k_w") {
            options.params.k_w = parse_int(next());
        } else if (argument == "--stride_h") {
            options.params.stride_h = parse_int(next());
        } else if (argument == "--stride_w") {
            options.params.stride_w = parse_int(next());
        } else if (argument == "--pad_h") {
            options.params.pad_h = parse_int(next());
        } else if (argument == "--pad_w") {
            options.params.pad_w = parse_int(next());
        } else if (argument == "--dilation_h") {
            options.params.dilation_h = parse_int(next());
        } else if (argument == "--dilation_w") {
            options.params.dilation_w = parse_int(next());
        } else if (argument == "--groups") {
            options.params.groups = parse_int(next());
        } else if (argument == "--algo") {
            options.algo = next();
        } else if (argument == "--input") {
            options.input_path = next();
        } else if (argument == "--weight") {
            options.weight_path = next();
        } else if (argument == "--bias") {
            options.bias_path = next();
        } else if (argument == "--output") {
            options.output_path = next();
        } else {
            throw std::runtime_error("Unknown argument: " + argument);
        }
    }

    if (options.input_path.empty() || options.weight_path.empty() || options.output_path.empty()) {
        throw std::runtime_error(
            "File mode requires --input, --weight and --output. "
            "Correctness and performance checks are owned by the Python test harness.");
    }
    return options;
}

std::vector<float> read_floats(const std::string& path, const std::int64_t expected) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::vector<float> data(static_cast<std::size_t>(expected));
    stream.read(reinterpret_cast<char*>(data.data()), expected * static_cast<std::int64_t>(sizeof(float)));
    if (stream.gcount() != expected * static_cast<std::int64_t>(sizeof(float))) {
        throw std::runtime_error("Unexpected file size: " + path);
    }
    return data;
}

void write_floats(const std::string& path, const std::vector<float>& data) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    stream.write(
        reinterpret_cast<const char*>(data.data()),
        static_cast<std::int64_t>(data.size()) * static_cast<std::int64_t>(sizeof(float)));
}

// Run conv2d on device given host buffers; returns host output.
std::vector<float> run_conv2d(
    const ConvParams& params,
    const std::string& algo,
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::vector<float>* bias) {
    float* device_input = nullptr;
    float* device_weight = nullptr;
    float* device_bias = nullptr;
    float* device_output = nullptr;

    const std::int64_t output_numel = params.output_numel();

    WGKERNEL_CUDA_CHECK(cudaMalloc(&device_input, sizeof(float) * input.size()));
    WGKERNEL_CUDA_CHECK(cudaMalloc(&device_weight, sizeof(float) * weight.size()));
    WGKERNEL_CUDA_CHECK(cudaMalloc(&device_output, sizeof(float) * static_cast<std::size_t>(output_numel)));
    if (bias != nullptr) {
        WGKERNEL_CUDA_CHECK(cudaMalloc(&device_bias, sizeof(float) * bias->size()));
    }

    WGKERNEL_CUDA_CHECK(cudaMemcpy(
        device_input, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice));
    WGKERNEL_CUDA_CHECK(cudaMemcpy(
        device_weight, weight.data(), sizeof(float) * weight.size(), cudaMemcpyHostToDevice));
    if (bias != nullptr) {
        WGKERNEL_CUDA_CHECK(cudaMemcpy(
            device_bias, bias->data(), sizeof(float) * bias->size(), cudaMemcpyHostToDevice));
    }

    auto dispatch = [&](auto fn) {
        return fn(
            device_input,
            device_weight,
            device_bias,
            device_output,
            params.n,
            params.c_in,
            params.h_in,
            params.w_in,
            params.c_out,
            params.k_h,
            params.k_w,
            params.h_out(),
            params.w_out(),
            params.stride_h,
            params.stride_w,
            params.pad_h,
            params.pad_w,
            params.dilation_h,
            params.dilation_w,
            params.groups,
            static_cast<cudaStream_t>(nullptr));
    };

    cudaError_t status = cudaSuccess;
    if (algo == "naive") {
        status = dispatch(wgkernel::cuda::conv2d_nchw);
    } else if (algo == "im2col_gemm") {
        status = dispatch(wgkernel::cuda::conv2d_nchw_im2col_gemm);
    } else if (algo == "direct_tiled") {
        status = dispatch(wgkernel::cuda::conv2d_nchw_direct_tiled);
    } else if (algo == "implicit_gemm") {
        status = dispatch(wgkernel::cuda::conv2d_nchw_implicit_gemm);
    } else {
        cudaFree(device_bias);
        cudaFree(device_output);
        cudaFree(device_weight);
        cudaFree(device_input);
        throw std::runtime_error("Unknown algo: " + algo);
    }
    WGKERNEL_CUDA_CHECK(status);
    WGKERNEL_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> output(static_cast<std::size_t>(output_numel));
    WGKERNEL_CUDA_CHECK(cudaMemcpy(
        output.data(), device_output, sizeof(float) * output.size(), cudaMemcpyDeviceToHost));

    cudaFree(device_bias);
    cudaFree(device_output);
    cudaFree(device_weight);
    cudaFree(device_input);
    return output;
}

}  // namespace

// File mode driver: read input/weight[/bias] from disk, run the requested
// algorithm, write the output tensor to disk. Correctness and performance
// checks are owned by the Python test harness.
int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const ConvParams& params = options.params;
        const auto input = read_floats(options.input_path, params.input_numel());
        const auto weight = read_floats(options.weight_path, params.weight_numel());

        std::vector<float> bias;
        const std::vector<float>* bias_ptr = nullptr;
        if (!options.bias_path.empty()) {
            bias = read_floats(options.bias_path, params.c_out);
            bias_ptr = &bias;
        }

        const auto output = run_conv2d(params, options.algo, input, weight, bias_ptr);
        write_floats(options.output_path, output);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
