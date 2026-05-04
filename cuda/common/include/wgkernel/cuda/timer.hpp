#pragma once

#include <cuda_runtime.h>

#include <stdexcept>

namespace wgkernel::cuda {

class CudaEventTimer {
public:
    CudaEventTimer() {
        if (cudaEventCreate(&start_) != cudaSuccess || cudaEventCreate(&stop_) != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA events");
        }
    }

    ~CudaEventTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    void start(cudaStream_t stream = nullptr) {
        cudaEventRecord(start_, stream);
    }

    float stop(cudaStream_t stream = nullptr) {
        cudaEventRecord(stop_, stream);
        cudaEventSynchronize(stop_);

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start_, stop_);
        return elapsed_ms;
    }

private:
    cudaEvent_t start_ {};
    cudaEvent_t stop_ {};
};

}  // namespace wgkernel::cuda
