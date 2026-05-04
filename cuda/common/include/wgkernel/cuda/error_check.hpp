#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace wgkernel::cuda {

inline void throw_if_cuda_error(
    const cudaError_t error,
    const char* expression,
    const char* file,
    const int line) {
    if (error == cudaSuccess) {
        return;
    }

    std::ostringstream stream;
    stream << "CUDA call failed: " << expression << " at " << file << ":" << line
           << " -> " << cudaGetErrorString(error);
    throw std::runtime_error(stream.str());
}

}  // namespace wgkernel::cuda

#define WGKERNEL_CUDA_CHECK(expression) \
    ::wgkernel::cuda::throw_if_cuda_error((expression), #expression, __FILE__, __LINE__)
