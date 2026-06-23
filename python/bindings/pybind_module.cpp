#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "cuda/reduce.hpp"
#include "cuda/activation_elementwise.hpp"
#include "cuda/embedding_indexing.hpp"
#include "cuda/transpose_layout_transform.hpp"
#include "cuda/norm.hpp"
#include "cuda/convolution.hpp"
#include "cuda/pooling.hpp"
#include "cuda/resize.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace py = pybind11;
using namespace pybind11::literals;

// RAII wrapper for CUDA allocations used as temporary workspace.
struct CudaBuffer {
    void* ptr = nullptr;
    std::size_t bytes = 0;

    CudaBuffer() = default;
    explicit CudaBuffer(std::size_t bytes) : bytes(bytes) {
        if (bytes > 0) {
            auto err = cudaMalloc(&ptr, bytes);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMalloc failed");
            }
        }
    }

    ~CudaBuffer() {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    // Non-copyable, movable
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&&) = default;
    CudaBuffer& operator=(CudaBuffer&&) = default;
};

// Copy a numpy array (host) to a pre-allocated device buffer.
void host_to_device(const float* host_ptr, float* device_ptr, std::size_t numel) {
    auto err = cudaMemcpy(device_ptr, host_ptr, numel * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy H2D failed");
    }
}

// Copy a device buffer to host numpy array.
void device_to_host(float* device_ptr, float* host_ptr, std::size_t numel) {
    auto err = cudaMemcpy(host_ptr, device_ptr, numel * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy D2H failed");
    }
}

void device_to_host_i64(std::int64_t* device_ptr, std::int64_t* host_ptr) {
    auto err = cudaMemcpy(host_ptr, device_ptr, sizeof(std::int64_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy D2H i64 failed");
    }
}

py::module_ import_torch() {
    return py::module_::import("torch");
}

void require_cuda_float32_tensor(const py::handle& tensor, const char* name) {
    const py::module_ torch = import_torch();

    if (!tensor.attr("is_cuda").cast<bool>()) {
        throw std::runtime_error(std::string(name) + " must be a CUDA tensor");
    }
    if (!tensor.attr("is_contiguous")().cast<bool>()) {
        throw std::runtime_error(std::string(name) + " must be contiguous");
    }
    if (!tensor.attr("dtype").equal(torch.attr("float32"))) {
        throw std::runtime_error(std::string(name) + " must have dtype torch.float32");
    }
}

template <typename T>
T* tensor_data_ptr(const py::handle& tensor) {
    const std::uintptr_t ptr_value = tensor.attr("data_ptr")().cast<std::uintptr_t>();
    return reinterpret_cast<T*>(ptr_value);
}

py::object make_torch_tensor_1d(const py::handle& like_tensor, const py::handle& dtype, const std::int64_t size) {
    const py::module_ torch = import_torch();
    return torch.attr("empty")(
        py::make_tuple(size),
        "device"_a = like_tensor.attr("device"),
        "dtype"_a = dtype);
}

py::object make_torch_tensor_4d(
    const py::handle& like_tensor,
    const int n,
    const int c,
    const int h,
    const int w) {
    const py::module_ torch = import_torch();
    return torch.attr("empty")(
        py::make_tuple(n, c, h, w),
        "device"_a = like_tensor.attr("device"),
        "dtype"_a = torch.attr("float32"));
}

void check_cuda_status(const cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

cudaError_t synchronize_if_success(const cudaError_t status) {
    if (status != cudaSuccess) {
        return status;
    }
    return cudaDeviceSynchronize();
}

// Require contiguous C-order float array.
py::array_t<float> require_contiguous_float32(py::array_t<float>& arr) {
    py::buffer_info info = arr.request();
    if (info.format != py::format_descriptor<float>::format()) {
        throw std::runtime_error("Expected float32 array");
    }
    if (!info.strides.empty()) {
        // Ensure C-contiguous by making a copy if needed.
        py::array_t<float> dense(info.size);
        auto* src = static_cast<const float*>(info.ptr);
        auto* dst = static_cast<float*>(dense.request().ptr);
        std::memcpy(dst, src, info.size * sizeof(float));
        return dense;
    }
    return arr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Reduce
// ─────────────────────────────────────────────────────────────────────────────

float py_reduce_sum(py::array_t<float> input) {
    auto host = require_contiguous_float32(input);
    auto numel = static_cast<std::int64_t>(host.request().size);

    float* d_in = nullptr;
    float* d_out = nullptr;
    CudaBuffer ws(wgkernel::cuda::reduce_sum_workspace_size(numel));

    auto err = cudaMalloc(&d_in, numel * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");
    err = cudaMalloc(&d_out, sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_in);
        throw std::runtime_error("cudaMalloc d_out failed");
    }

    host_to_device(static_cast<float*>(host.request().ptr), d_in, numel);

    wgkernel::cuda::reduce_sum(d_in, d_out, numel, ws.ptr, ws.bytes);
    cudaDeviceSynchronize();

    float result = 0.0f;
    device_to_host(d_out, &result, 1);

    cudaFree(d_out);
    cudaFree(d_in);
    return result;
}

float py_reduce_max(py::array_t<float> input) {
    auto host = require_contiguous_float32(input);
    auto numel = static_cast<std::int64_t>(host.request().size);

    float* d_in = nullptr;
    float* d_out = nullptr;
    CudaBuffer ws(wgkernel::cuda::reduce_max_workspace_size(numel));

    auto err = cudaMalloc(&d_in, numel * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");
    err = cudaMalloc(&d_out, sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_in);
        throw std::runtime_error("cudaMalloc d_out failed");
    }

    host_to_device(static_cast<float*>(host.request().ptr), d_in, numel);

    wgkernel::cuda::reduce_max(d_in, d_out, numel, ws.ptr, ws.bytes);
    cudaDeviceSynchronize();

    float result = 0.0f;
    device_to_host(d_out, &result, 1);

    cudaFree(d_out);
    cudaFree(d_in);
    return result;
}

std::int64_t py_reduce_argmax(py::array_t<float> input) {
    auto host = require_contiguous_float32(input);
    auto numel = static_cast<std::int64_t>(host.request().size);

    float* d_in = nullptr;
    std::int64_t* d_out = nullptr;
    CudaBuffer ws(wgkernel::cuda::reduce_argmax_workspace_size(numel));

    auto err = cudaMalloc(&d_in, numel * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");
    err = cudaMalloc(&d_out, sizeof(std::int64_t));
    if (err != cudaSuccess) {
        cudaFree(d_in);
        throw std::runtime_error("cudaMalloc d_out failed");
    }

    host_to_device(static_cast<float*>(host.request().ptr), d_in, numel);

    wgkernel::cuda::reduce_argmax(d_in, d_out, numel, ws.ptr, ws.bytes);
    cudaDeviceSynchronize();

    std::int64_t result = -1;
    device_to_host_i64(d_out, &result);

    cudaFree(d_out);
    cudaFree(d_in);
    return result;
}

float py_reduce_sum_torch(const py::object& values) {
    require_cuda_float32_tensor(values, "values");
    const auto numel = values.attr("numel")().cast<std::int64_t>();
    py::object output = make_torch_tensor_1d(values, import_torch().attr("float32"), 1);

    void* workspace = nullptr;
    const std::size_t workspace_bytes = wgkernel::cuda::reduce_sum_workspace_size(numel);
    if (workspace_bytes > 0) {
        check_cuda_status(cudaMalloc(&workspace, workspace_bytes), "cudaMalloc workspace");
    }

    const cudaError_t status = wgkernel::cuda::reduce_sum(
        tensor_data_ptr<float>(values),
        tensor_data_ptr<float>(output),
        numel,
        workspace,
        workspace_bytes);
    const cudaError_t sync_status = synchronize_if_success(status);
    if (workspace != nullptr) {
        cudaFree(workspace);
    }
    check_cuda_status(sync_status, "reduce_sum_torch");
    return output.attr("item")().cast<float>();
}

float py_reduce_max_torch(const py::object& values) {
    require_cuda_float32_tensor(values, "values");
    const auto numel = values.attr("numel")().cast<std::int64_t>();
    py::object output = make_torch_tensor_1d(values, import_torch().attr("float32"), 1);

    void* workspace = nullptr;
    const std::size_t workspace_bytes = wgkernel::cuda::reduce_max_workspace_size(numel);
    if (workspace_bytes > 0) {
        check_cuda_status(cudaMalloc(&workspace, workspace_bytes), "cudaMalloc workspace");
    }

    const cudaError_t status = wgkernel::cuda::reduce_max(
        tensor_data_ptr<float>(values),
        tensor_data_ptr<float>(output),
        numel,
        workspace,
        workspace_bytes);
    const cudaError_t sync_status = synchronize_if_success(status);
    if (workspace != nullptr) {
        cudaFree(workspace);
    }
    check_cuda_status(sync_status, "reduce_max_torch");
    return output.attr("item")().cast<float>();
}

std::int64_t py_reduce_argmax_torch(const py::object& values) {
    require_cuda_float32_tensor(values, "values");
    const auto numel = values.attr("numel")().cast<std::int64_t>();
    py::object output = make_torch_tensor_1d(values, import_torch().attr("int64"), 1);

    void* workspace = nullptr;
    const std::size_t workspace_bytes = wgkernel::cuda::reduce_argmax_workspace_size(numel);
    check_cuda_status(cudaMalloc(&workspace, workspace_bytes), "cudaMalloc workspace");

    const cudaError_t status = wgkernel::cuda::reduce_argmax(
        tensor_data_ptr<float>(values),
        tensor_data_ptr<std::int64_t>(output),
        numel,
        workspace,
        workspace_bytes);
    const cudaError_t sync_status = synchronize_if_success(status);
    cudaFree(workspace);
    check_cuda_status(sync_status, "reduce_argmax_torch");
    return output.attr("item")().cast<std::int64_t>();
}

// ─────────────────────────────────────────────────────────────────────────────
// Activation / Elementwise
// ─────────────────────────────────────────────────────────────────────────────

py::array_t<float> py_silu(py::array_t<float> input) {
    auto host = require_contiguous_float32(input);
    auto numel = static_cast<std::int64_t>(host.request().size);

    py::array_t<float> output(numel);
    float* d_in = nullptr;
    float* d_out = static_cast<float*>(output.request().ptr);

    auto err = cudaMalloc(&d_in, numel * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");

    host_to_device(static_cast<float*>(host.request().ptr), d_in, numel);
    wgkernel::cuda::silu(d_in, d_out, numel);
    cudaDeviceSynchronize();
    device_to_host(d_out, static_cast<float*>(output.request().ptr), numel);

    cudaFree(d_in);
    return output;
}

py::array_t<float> py_sigmoid(py::array_t<float> input) {
    auto host = require_contiguous_float32(input);
    auto numel = static_cast<std::int64_t>(host.request().size);

    py::array_t<float> output(numel);
    float* d_in = nullptr;
    float* d_out = static_cast<float*>(output.request().ptr);

    auto err = cudaMalloc(&d_in, numel * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");

    host_to_device(static_cast<float*>(host.request().ptr), d_in, numel);
    wgkernel::cuda::sigmoid(d_in, d_out, numel);
    cudaDeviceSynchronize();
    device_to_host(d_out, static_cast<float*>(output.request().ptr), numel);

    cudaFree(d_in);
    return output;
}

py::array_t<float> py_exp(py::array_t<float> input) {
    auto host = require_contiguous_float32(input);
    auto numel = static_cast<std::int64_t>(host.request().size);

    py::array_t<float> output(numel);
    float* d_in = nullptr;
    float* d_out = static_cast<float*>(output.request().ptr);

    auto err = cudaMalloc(&d_in, numel * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");

    host_to_device(static_cast<float*>(host.request().ptr), d_in, numel);
    wgkernel::cuda::exp(d_in, d_out, numel);
    cudaDeviceSynchronize();
    device_to_host(d_out, static_cast<float*>(output.request().ptr), numel);

    cudaFree(d_in);
    return output;
}

py::array_t<float> py_add(py::array_t<float> a, py::array_t<float> b) {
    auto ha = require_contiguous_float32(a);
    auto hb = require_contiguous_float32(b);
    auto numel = static_cast<std::int64_t>(ha.request().size);

    if (static_cast<std::size_t>(numel) != hb.request().size) {
        throw std::runtime_error("Elementwise op: input sizes must match");
    }

    py::array_t<float> output(numel);
    float *d_a = nullptr, *d_b = nullptr;
    float* d_out = static_cast<float*>(output.request().ptr);

    auto err = cudaMalloc(&d_a, numel * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_a failed");
    err = cudaMalloc(&d_b, numel * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_a);
        throw std::runtime_error("cudaMalloc d_b failed");
    }

    host_to_device(static_cast<float*>(ha.request().ptr), d_a, numel);
    host_to_device(static_cast<float*>(hb.request().ptr), d_b, numel);

    wgkernel::cuda::add(d_a, d_b, d_out, numel);
    cudaDeviceSynchronize();
    device_to_host(d_out, static_cast<float*>(output.request().ptr), numel);

    cudaFree(d_b);
    cudaFree(d_a);
    return output;
}

py::array_t<float> py_sub(py::array_t<float> a, py::array_t<float> b) {
    auto ha = require_contiguous_float32(a);
    auto hb = require_contiguous_float32(b);
    auto numel = static_cast<std::int64_t>(ha.request().size);

    if (static_cast<std::size_t>(numel) != hb.request().size) {
        throw std::runtime_error("Elementwise op: input sizes must match");
    }

    py::array_t<float> output(numel);
    float *d_a = nullptr, *d_b = nullptr;
    float* d_out = static_cast<float*>(output.request().ptr);

    auto err = cudaMalloc(&d_a, numel * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_a failed");
    err = cudaMalloc(&d_b, numel * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_a);
        throw std::runtime_error("cudaMalloc d_b failed");
    }

    host_to_device(static_cast<float*>(ha.request().ptr), d_a, numel);
    host_to_device(static_cast<float*>(hb.request().ptr), d_b, numel);

    wgkernel::cuda::sub(d_a, d_b, d_out, numel);
    cudaDeviceSynchronize();
    device_to_host(d_out, static_cast<float*>(output.request().ptr), numel);

    cudaFree(d_b);
    cudaFree(d_a);
    return output;
}

py::array_t<float> py_mul(py::array_t<float> a, py::array_t<float> b) {
    auto ha = require_contiguous_float32(a);
    auto hb = require_contiguous_float32(b);
    auto numel = static_cast<std::int64_t>(ha.request().size);

    if (static_cast<std::size_t>(numel) != hb.request().size) {
        throw std::runtime_error("Elementwise op: input sizes must match");
    }

    py::array_t<float> output(numel);
    float *d_a = nullptr, *d_b = nullptr;
    float* d_out = static_cast<float*>(output.request().ptr);

    auto err = cudaMalloc(&d_a, numel * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_a failed");
    err = cudaMalloc(&d_b, numel * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_a);
        throw std::runtime_error("cudaMalloc d_b failed");
    }

    host_to_device(static_cast<float*>(ha.request().ptr), d_a, numel);
    host_to_device(static_cast<float*>(hb.request().ptr), d_b, numel);

    wgkernel::cuda::mul(d_a, d_b, d_out, numel);
    cudaDeviceSynchronize();
    device_to_host(d_out, static_cast<float*>(output.request().ptr), numel);

    cudaFree(d_b);
    cudaFree(d_a);
    return output;
}

// ─────────────────────────────────────────────────────────────────────────────
// Embedding / Indexing
// ─────────────────────────────────────────────────────────────────────────────

py::array_t<float> py_slice_1d(
    py::array_t<float> input,
    std::int64_t start,
    std::int64_t step,
    std::int64_t output_numel) {
    auto host = require_contiguous_float32(input);
    auto input_numel = static_cast<std::int64_t>(host.request().size);

    py::array_t<float> output(output_numel);

    float *d_in = nullptr, *d_out = static_cast<float*>(output.request().ptr);
    auto err = cudaMalloc(&d_in, input_numel * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");

    host_to_device(static_cast<float*>(host.request().ptr), d_in, input_numel);
    wgkernel::cuda::slice_1d(d_in, d_out, input_numel, start, step, output_numel);
    cudaDeviceSynchronize();
    device_to_host(d_out, static_cast<float*>(output.request().ptr), output_numel);

    cudaFree(d_in);
    return output;
}

py::array_t<float> py_gather_1d(py::array_t<float> input, py::array_t<std::int64_t> indices) {
    auto host = require_contiguous_float32(input);
    auto idx = indices.request();
    auto output_numel = static_cast<std::int64_t>(idx.size);

    py::array_t<float> output(output_numel);

    float *d_in = nullptr, *d_out = static_cast<float*>(output.request().ptr);
    std::int64_t* d_idx = nullptr;

    auto in_numel = static_cast<std::int64_t>(host.request().size);
    auto err = cudaMalloc(&d_in, in_numel * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");
    err = cudaMalloc(&d_idx, output_numel * sizeof(std::int64_t));
    if (err != cudaSuccess) {
        cudaFree(d_in);
        throw std::runtime_error("cudaMalloc d_idx failed");
    }

    host_to_device(static_cast<float*>(host.request().ptr), d_in, in_numel);
    auto* h_idx = static_cast<std::int64_t*>(idx.ptr);
    err = cudaMemcpy(d_idx, h_idx, output_numel * sizeof(std::int64_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy indices failed");

    wgkernel::cuda::gather_1d(d_in, d_idx, d_out, output_numel);
    cudaDeviceSynchronize();
    device_to_host(d_out, static_cast<float*>(output.request().ptr), output_numel);

    cudaFree(d_idx);
    cudaFree(d_in);
    return output;
}

std::tuple<py::array_t<float>, py::array_t<std::int64_t>> py_topk_1d(
    py::array_t<float> input, int k) {
    auto host = require_contiguous_float32(input);
    auto numel = static_cast<std::int64_t>(host.request().size);

    py::array_t<float> values(k);
    py::array_t<std::int64_t> indices(k);

    float* d_in = nullptr;
    float* d_vals = static_cast<float*>(values.request().ptr);
    std::int64_t* d_idx = static_cast<std::int64_t*>(indices.request().ptr);

    auto err = cudaMalloc(&d_in, numel * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");

    host_to_device(static_cast<float*>(host.request().ptr), d_in, numel);
    wgkernel::cuda::topk_1d(d_in, d_vals, d_idx, numel, k);
    cudaDeviceSynchronize();

    device_to_host(d_vals, static_cast<float*>(values.request().ptr), k);
    auto* h_idx = static_cast<std::int64_t*>(indices.request().ptr);
    err = cudaMemcpy(h_idx, d_idx, k * sizeof(std::int64_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy indices D2H failed");

    cudaFree(d_in);
    return std::make_tuple(values, indices);
}

std::tuple<py::array_t<float>, py::array_t<std::int64_t>> py_sort_1d(
    py::array_t<float> input, bool descending) {
    auto host = require_contiguous_float32(input);
    auto numel = static_cast<std::int64_t>(host.request().size);

    py::array_t<float> values(numel);
    py::array_t<std::int64_t> indices(numel);

    float *d_in = nullptr;
    float* d_vals = static_cast<float*>(values.request().ptr);
    std::int64_t* d_idx = static_cast<std::int64_t*>(indices.request().ptr);

    auto err = cudaMalloc(&d_in, numel * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");

    host_to_device(static_cast<float*>(host.request().ptr), d_in, numel);
    wgkernel::cuda::sort_1d(d_in, d_vals, d_idx, numel, descending);
    cudaDeviceSynchronize();

    device_to_host(d_vals, static_cast<float*>(values.request().ptr), numel);
    auto* h_idx = static_cast<std::int64_t*>(indices.request().ptr);
    err = cudaMemcpy(h_idx, d_idx, numel * sizeof(std::int64_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy indices D2H failed");

    cudaFree(d_in);
    return std::make_tuple(values, indices);
}

// ─────────────────────────────────────────────────────────────────────────────
// Transpose / Layout Transform
// ─────────────────────────────────────────────────────────────────────────────

py::array_t<float> py_concat_nchw_axis1(
    py::array_t<float> a, py::array_t<float> b, int n, int c_a, int c_b, int h, int w) {
    auto ha = require_contiguous_float32(a);
    auto hb = require_contiguous_float32(b);

    const std::size_t total = static_cast<std::size_t>(n) * (c_a + c_b) * h * w;
    py::array_t<float> output(total);

    float *d_a = nullptr, *d_b = nullptr;
    float* d_out = static_cast<float*>(output.request().ptr);

    const std::size_t size_a = static_cast<std::size_t>(n) * c_a * h * w;
    const std::size_t size_b = static_cast<std::size_t>(n) * c_b * h * w;

    auto err = cudaMalloc(&d_a, size_a * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_a failed");
    err = cudaMalloc(&d_b, size_b * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_a);
        throw std::runtime_error("cudaMalloc d_b failed");
    }

    host_to_device(static_cast<float*>(ha.request().ptr), d_a, size_a);
    host_to_device(static_cast<float*>(hb.request().ptr), d_b, size_b);

    wgkernel::cuda::concat_nchw_axis1(d_a, d_b, d_out, n, c_a, c_b, h, w);
    cudaDeviceSynchronize();
    device_to_host(d_out, static_cast<float*>(output.request().ptr), total);

    cudaFree(d_b);
    cudaFree(d_a);
    return output;
}

py::array_t<float> py_permute_nchw_to_nhwc(
    py::array_t<float> input, int n, int c, int h, int w) {
    auto host = require_contiguous_float32(input);
    const std::size_t total = static_cast<std::size_t>(n) * h * w * c;

    py::array_t<float> output(total);

    float* d_in = nullptr;
    float* d_out = static_cast<float*>(output.request().ptr);

    const std::size_t in_size = static_cast<std::size_t>(n) * c * h * w;
    auto err = cudaMalloc(&d_in, in_size * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");

    host_to_device(static_cast<float*>(host.request().ptr), d_in, in_size);
    wgkernel::cuda::permute_nchw_to_nhwc(d_in, d_out, n, c, h, w);
    cudaDeviceSynchronize();
    device_to_host(d_out, static_cast<float*>(output.request().ptr), total);

    cudaFree(d_in);
    return output;
}

py::array_t<float> py_permute_nhwc_to_nchw(
    py::array_t<float> input, int n, int c, int h, int w) {
    auto host = require_contiguous_float32(input);
    const std::size_t total = static_cast<std::size_t>(n) * c * h * w;

    py::array_t<float> output(total);

    float* d_in = nullptr;
    float* d_out = static_cast<float*>(output.request().ptr);

    const std::size_t in_size = static_cast<std::size_t>(n) * h * w * c;
    auto err = cudaMalloc(&d_in, in_size * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");

    host_to_device(static_cast<float*>(host.request().ptr), d_in, in_size);
    wgkernel::cuda::permute_nhwc_to_nchw(d_in, d_out, n, c, h, w);
    cudaDeviceSynchronize();
    device_to_host(d_out, static_cast<float*>(output.request().ptr), total);

    cudaFree(d_in);
    return output;
}

py::array_t<float> py_copy_1d(py::array_t<float> input) {
    auto host = require_contiguous_float32(input);
    auto numel = static_cast<std::int64_t>(host.request().size);

    py::array_t<float> output(numel);

    float *d_in = nullptr;
    float* d_out = static_cast<float*>(output.request().ptr);

    auto err = cudaMalloc(&d_in, numel * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");

    host_to_device(static_cast<float*>(host.request().ptr), d_in, numel);
    wgkernel::cuda::copy_1d(d_in, d_out, numel);
    cudaDeviceSynchronize();
    device_to_host(d_out, static_cast<float*>(output.request().ptr), numel);

    cudaFree(d_in);
    return output;
}

// ─────────────────────────────────────────────────────────────────────────────
// Norm
// ─────────────────────────────────────────────────────────────────────────────

py::array_t<float> py_batchnorm2d_inference_nchw(
    py::array_t<float> input,
    py::array_t<float> scale,
    py::array_t<float> bias,
    int n, int c, int h, int w) {
    auto h_input = require_contiguous_float32(input);
    auto h_scale = require_contiguous_float32(scale);
    auto h_bias = require_contiguous_float32(bias);

    const std::size_t total = static_cast<std::size_t>(n) * c * h * w;
    py::array_t<float> output(total);

    float *d_in = nullptr, *d_scale = nullptr, *d_bias = nullptr;
    float* d_out = static_cast<float*>(output.request().ptr);

    auto err = cudaMalloc(&d_in, total * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");
    err = cudaMalloc(&d_scale, c * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_in);
        throw std::runtime_error("cudaMalloc d_scale failed");
    }
    err = cudaMalloc(&d_bias, c * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_scale);
        throw std::runtime_error("cudaMalloc d_bias failed");
    }

    host_to_device(static_cast<float*>(h_input.request().ptr), d_in, total);
    host_to_device(static_cast<float*>(h_scale.request().ptr), d_scale, c);
    host_to_device(static_cast<float*>(h_bias.request().ptr), d_bias, c);

    wgkernel::cuda::batchnorm2d_inference_nchw(d_in, d_scale, d_bias, d_out, n, c, h, w);
    cudaDeviceSynchronize();
    device_to_host(d_out, static_cast<float*>(output.request().ptr), total);

    cudaFree(d_bias);
    cudaFree(d_scale);
    cudaFree(d_in);
    return output;
}

// ─────────────────────────────────────────────────────────────────────────────
// Convolution
// conv2d output NCHW shape: n, c_out, h_out, w_out
// ─────────────────────────────────────────────────────────────────────────────

py::array_t<float> py_conv2d_nchw(
    py::array_t<float> input,
    py::array_t<float> weight,
    py::array_t<float> bias,
    int n, int c_in, int h_in, int w_in,
    int c_out, int k_h, int k_w,
    int h_out, int w_out,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups) {
    auto h_inp = require_contiguous_float32(input);
    auto h_wgt = require_contiguous_float32(weight);
    auto h_bias = require_contiguous_float32(bias);

    const std::size_t total = static_cast<std::size_t>(n) * c_out * h_out * w_out;
    py::array_t<float> output(total);

    float *d_in = nullptr, *d_wgt = nullptr, *d_bias = nullptr;
    float* d_out = static_cast<float*>(output.request().ptr);

    const std::size_t in_size = static_cast<std::size_t>(n) * c_in * h_in * w_in;
    const std::size_t wgt_size = static_cast<std::size_t>(c_out) * c_in * k_h * k_w / groups;

    auto err = cudaMalloc(&d_in, in_size * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");
    err = cudaMalloc(&d_wgt, wgt_size * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_in);
        throw std::runtime_error("cudaMalloc d_wgt failed");
    }
    err = cudaMalloc(&d_bias, c_out * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_wgt);
        throw std::runtime_error("cudaMalloc d_bias failed");
    }

    host_to_device(static_cast<float*>(h_inp.request().ptr), d_in, in_size);
    host_to_device(static_cast<float*>(h_wgt.request().ptr), d_wgt, wgt_size);
    host_to_device(static_cast<float*>(h_bias.request().ptr), d_bias, c_out);

    wgkernel::cuda::conv2d_nchw(d_in, d_wgt, d_bias, d_out,
        n, c_in, h_in, w_in, c_out, k_h, k_w, h_out, w_out,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups);
    cudaDeviceSynchronize();
    device_to_host(d_out, static_cast<float*>(output.request().ptr), total);

    cudaFree(d_bias);
    cudaFree(d_wgt);
    cudaFree(d_in);
    return output;
}

py::array_t<float> py_conv2d_nchw_im2col_gemm(
    py::array_t<float> input,
    py::array_t<float> weight,
    py::array_t<float> bias,
    int n, int c_in, int h_in, int w_in,
    int c_out, int k_h, int k_w,
    int h_out, int w_out,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups) {
    auto h_inp = require_contiguous_float32(input);
    auto h_wgt = require_contiguous_float32(weight);
    auto h_bias = require_contiguous_float32(bias);

    const std::size_t total = static_cast<std::size_t>(n) * c_out * h_out * w_out;
    py::array_t<float> output(total);

    float *d_in = nullptr, *d_wgt = nullptr, *d_bias = nullptr;
    float* d_out = static_cast<float*>(output.request().ptr);

    const std::size_t in_size = static_cast<std::size_t>(n) * c_in * h_in * w_in;
    const std::size_t wgt_size = static_cast<std::size_t>(c_out) * c_in * k_h * k_w / groups;

    auto err = cudaMalloc(&d_in, in_size * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");
    err = cudaMalloc(&d_wgt, wgt_size * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_in);
        throw std::runtime_error("cudaMalloc d_wgt failed");
    }
    err = cudaMalloc(&d_bias, c_out * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_wgt);
        throw std::runtime_error("cudaMalloc d_bias failed");
    }

    host_to_device(static_cast<float*>(h_inp.request().ptr), d_in, in_size);
    host_to_device(static_cast<float*>(h_wgt.request().ptr), d_wgt, wgt_size);
    host_to_device(static_cast<float*>(h_bias.request().ptr), d_bias, c_out);

    wgkernel::cuda::conv2d_nchw_im2col_gemm(d_in, d_wgt, d_bias, d_out,
        n, c_in, h_in, w_in, c_out, k_h, k_w, h_out, w_out,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups);
    cudaDeviceSynchronize();
    device_to_host(d_out, static_cast<float*>(output.request().ptr), total);

    cudaFree(d_bias);
    cudaFree(d_wgt);
    cudaFree(d_in);
    return output;
}

py::array_t<float> py_conv2d_nchw_direct_tiled(
    py::array_t<float> input,
    py::array_t<float> weight,
    py::array_t<float> bias,
    int n, int c_in, int h_in, int w_in,
    int c_out, int k_h, int k_w,
    int h_out, int w_out,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups) {
    auto h_inp = require_contiguous_float32(input);
    auto h_wgt = require_contiguous_float32(weight);
    auto h_bias = require_contiguous_float32(bias);

    const std::size_t total = static_cast<std::size_t>(n) * c_out * h_out * w_out;
    py::array_t<float> output(total);

    float *d_in = nullptr, *d_wgt = nullptr, *d_bias = nullptr;
    float* d_out = static_cast<float*>(output.request().ptr);

    const std::size_t in_size = static_cast<std::size_t>(n) * c_in * h_in * w_in;
    const std::size_t wgt_size = static_cast<std::size_t>(c_out) * c_in * k_h * k_w / groups;

    auto err = cudaMalloc(&d_in, in_size * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");
    err = cudaMalloc(&d_wgt, wgt_size * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_in);
        throw std::runtime_error("cudaMalloc d_wgt failed");
    }
    err = cudaMalloc(&d_bias, c_out * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_wgt);
        throw std::runtime_error("cudaMalloc d_bias failed");
    }

    host_to_device(static_cast<float*>(h_inp.request().ptr), d_in, in_size);
    host_to_device(static_cast<float*>(h_wgt.request().ptr), d_wgt, wgt_size);
    host_to_device(static_cast<float*>(h_bias.request().ptr), d_bias, c_out);

    wgkernel::cuda::conv2d_nchw_direct_tiled(d_in, d_wgt, d_bias, d_out,
        n, c_in, h_in, w_in, c_out, k_h, k_w, h_out, w_out,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups);
    cudaDeviceSynchronize();
    device_to_host(d_out, static_cast<float*>(output.request().ptr), total);

    cudaFree(d_bias);
    cudaFree(d_wgt);
    cudaFree(d_in);
    return output;
}

py::array_t<float> py_conv2d_nchw_implicit_gemm(
    py::array_t<float> input,
    py::array_t<float> weight,
    py::array_t<float> bias,
    int n, int c_in, int h_in, int w_in,
    int c_out, int k_h, int k_w,
    int h_out, int w_out,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups) {
    auto h_inp = require_contiguous_float32(input);
    auto h_wgt = require_contiguous_float32(weight);
    auto h_bias = require_contiguous_float32(bias);

    const std::size_t total = static_cast<std::size_t>(n) * c_out * h_out * w_out;
    py::array_t<float> output(total);

    float *d_in = nullptr, *d_wgt = nullptr, *d_bias = nullptr;
    float* d_out = static_cast<float*>(output.request().ptr);

    const std::size_t in_size = static_cast<std::size_t>(n) * c_in * h_in * w_in;
    const std::size_t wgt_size = static_cast<std::size_t>(c_out) * c_in * k_h * k_w / groups;

    auto err = cudaMalloc(&d_in, in_size * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");
    err = cudaMalloc(&d_wgt, wgt_size * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_in);
        throw std::runtime_error("cudaMalloc d_wgt failed");
    }
    err = cudaMalloc(&d_bias, c_out * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_wgt);
        throw std::runtime_error("cudaMalloc d_bias failed");
    }

    host_to_device(static_cast<float*>(h_inp.request().ptr), d_in, in_size);
    host_to_device(static_cast<float*>(h_wgt.request().ptr), d_wgt, wgt_size);
    host_to_device(static_cast<float*>(h_bias.request().ptr), d_bias, c_out);

    wgkernel::cuda::conv2d_nchw_implicit_gemm(d_in, d_wgt, d_bias, d_out,
        n, c_in, h_in, w_in, c_out, k_h, k_w, h_out, w_out,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups);
    cudaDeviceSynchronize();
    device_to_host(d_out, static_cast<float*>(output.request().ptr), total);

    cudaFree(d_bias);
    cudaFree(d_wgt);
    cudaFree(d_in);
    return output;
}

template <typename Fn>
py::object py_conv2d_torch_impl(
    Fn fn,
    const py::object& input,
    const py::object& weight,
    const py::object& bias,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups) {
    require_cuda_float32_tensor(input, "input");
    require_cuda_float32_tensor(weight, "weight");
    if (!bias.is_none()) {
        require_cuda_float32_tensor(bias, "bias");
    }

    const int n = input.attr("size")(0).cast<int>();
    const int c_in = input.attr("size")(1).cast<int>();
    const int h_in = input.attr("size")(2).cast<int>();
    const int w_in = input.attr("size")(3).cast<int>();
    const int c_out = weight.attr("size")(0).cast<int>();
    const int k_h = weight.attr("size")(2).cast<int>();
    const int k_w = weight.attr("size")(3).cast<int>();
    const int h_out = (h_in + 2 * pad_h - dilation_h * (k_h - 1) - 1) / stride_h + 1;
    const int w_out = (w_in + 2 * pad_w - dilation_w * (k_w - 1) - 1) / stride_w + 1;

    py::object output = make_torch_tensor_4d(input, n, c_out, h_out, w_out);
    const float* bias_ptr = bias.is_none() ? nullptr : tensor_data_ptr<float>(bias);

    const cudaError_t status = fn(
        tensor_data_ptr<float>(input),
        tensor_data_ptr<float>(weight),
        bias_ptr,
        tensor_data_ptr<float>(output),
        n,
        c_in,
        h_in,
        w_in,
        c_out,
        k_h,
        k_w,
        h_out,
        w_out,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        static_cast<cudaStream_t>(nullptr));
    check_cuda_status(synchronize_if_success(status), "conv2d_torch");
    return output;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pooling
// ─────────────────────────────────────────────────────────────────────────────

py::array_t<float> py_maxpool2d_nchw(
    py::array_t<float> input,
    int n, int c, int h_in, int w_in, int h_out, int w_out,
    int k_h, int k_w, int stride_h, int stride_w, int pad_h, int pad_w) {
    auto h_input = require_contiguous_float32(input);
    const std::size_t total = static_cast<std::size_t>(n) * c * h_out * w_out;
    py::array_t<float> output(total);

    float* d_in = nullptr;
    float* d_out = static_cast<float*>(output.request().ptr);
    const std::size_t in_size = static_cast<std::size_t>(n) * c * h_in * w_in;

    auto err = cudaMalloc(&d_in, in_size * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");

    host_to_device(static_cast<float*>(h_input.request().ptr), d_in, in_size);
    wgkernel::cuda::maxpool2d_nchw(d_in, d_out, n, c, h_in, w_in, h_out, w_out,
        k_h, k_w, stride_h, stride_w, pad_h, pad_w);
    cudaDeviceSynchronize();
    device_to_host(d_out, static_cast<float*>(output.request().ptr), total);

    cudaFree(d_in);
    return output;
}

// ─────────────────────────────────────────────────────────────────────────────
// Resize
// ─────────────────────────────────────────────────────────────────────────────

py::array_t<float> py_upsample_nearest2d_nchw(
    py::array_t<float> input, int n, int c, int h_in, int w_in, int h_out, int w_out) {
    auto h_input = require_contiguous_float32(input);
    const std::size_t total = static_cast<std::size_t>(n) * c * h_out * w_out;
    py::array_t<float> output(total);

    float* d_in = nullptr;
    float* d_out = static_cast<float*>(output.request().ptr);
    const std::size_t in_size = static_cast<std::size_t>(n) * c * h_in * w_in;

    auto err = cudaMalloc(&d_in, in_size * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");

    host_to_device(static_cast<float*>(h_input.request().ptr), d_in, in_size);
    wgkernel::cuda::upsample_nearest2d_nchw(d_in, d_out, n, c, h_in, w_in, h_out, w_out);
    cudaDeviceSynchronize();
    device_to_host(d_out, static_cast<float*>(output.request().ptr), total);

    cudaFree(d_in);
    return output;
}

py::array_t<float> py_upsample_bilinear2d_nchw(
    py::array_t<float> input, int n, int c, int h_in, int w_in, int h_out, int w_out,
    bool align_corners) {
    auto h_input = require_contiguous_float32(input);
    const std::size_t total = static_cast<std::size_t>(n) * c * h_out * w_out;
    py::array_t<float> output(total);

    float* d_in = nullptr;
    float* d_out = static_cast<float*>(output.request().ptr);
    const std::size_t in_size = static_cast<std::size_t>(n) * c * h_in * w_in;

    auto err = cudaMalloc(&d_in, in_size * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_in failed");

    host_to_device(static_cast<float*>(h_input.request().ptr), d_in, in_size);
    wgkernel::cuda::upsample_bilinear2d_nchw(d_in, d_out, n, c, h_in, w_in, h_out, w_out, align_corners);
    cudaDeviceSynchronize();
    device_to_host(d_out, static_cast<float*>(output.request().ptr), total);

    cudaFree(d_in);
    return output;
}

// ─────────────────────────────────────────────────────────────────────────────
// Module entry point
// ─────────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(wgkernel_cuda, m) {
    m.doc() = "wGKernels CUDA operator library";

    // Reduce
    m.def("reduce_sum", &py_reduce_sum, py::return_value_policy::move, "Reduce sum over float32 array");
    m.def("reduce_max", &py_reduce_max, py::return_value_policy::move, "Reduce max over float32 array");
    m.def("reduce_argmax", &py_reduce_argmax, py::return_value_policy::move, "Reduce argmax over float32 array");
    m.def("reduce_sum_torch", &py_reduce_sum_torch, "Reduce sum over a CUDA torch.float32 tensor");
    m.def("reduce_max_torch", &py_reduce_max_torch, "Reduce max over a CUDA torch.float32 tensor");
    m.def("reduce_argmax_torch", &py_reduce_argmax_torch, "Reduce argmax over a CUDA torch.float32 tensor");

    // Activation / Elementwise
    m.def("silu", &py_silu, py::return_value_policy::move, "SiLU activation");
    m.def("sigmoid", &py_sigmoid, py::return_value_policy::move, "Sigmoid activation");
    m.def("exp", &py_exp, py::return_value_policy::move, "Elementwise exp");
    m.def("add", &py_add, py::return_value_policy::move, "Elementwise add");
    m.def("sub", &py_sub, py::return_value_policy::move, "Elementwise sub");
    m.def("mul", &py_mul, py::return_value_policy::move, "Elementwise mul");

    // Embedding / Indexing
    m.def("slice_1d", &py_slice_1d, py::return_value_policy::move, "1D slice");
    m.def("gather_1d", &py_gather_1d, py::return_value_policy::move, "1D gather");
    m.def("topk_1d", &py_topk_1d, py::return_value_policy::move, "1D topk", py::arg("input"), py::arg("k"));
    m.def("sort_1d", &py_sort_1d, py::return_value_policy::move, "1D sort", py::arg("input"), py::arg("descending") = false);

    // Transpose / Layout Transform
    m.def("concat_nchw_axis1", &py_concat_nchw_axis1, py::return_value_policy::move, "Concat NCHW tensors along axis 1");
    m.def("permute_nchw_to_nhwc", &py_permute_nchw_to_nhwc, py::return_value_policy::move, "Permute NCHW to NHWC");
    m.def("permute_nhwc_to_nchw", &py_permute_nhwc_to_nchw, py::return_value_policy::move, "Permute NHWC to NCHW");
    m.def("copy_1d", &py_copy_1d, py::return_value_policy::move, "Copy 1D tensor");

    // Norm
    m.def("batchnorm2d_inference_nchw", &py_batchnorm2d_inference_nchw,
          py::return_value_policy::move, "BatchNorm2d inference (NCHW)");

    // Convolution
    m.def("conv2d_nchw", &py_conv2d_nchw, py::return_value_policy::move, "Conv2d NCHW (naive)");
    m.def("conv2d_nchw_im2col_gemm", &py_conv2d_nchw_im2col_gemm, py::return_value_policy::move, "Conv2d NCHW (im2col+GEMM)");
    m.def("conv2d_nchw_direct_tiled", &py_conv2d_nchw_direct_tiled, py::return_value_policy::move, "Conv2d NCHW (direct tiled)");
    m.def("conv2d_nchw_implicit_gemm", &py_conv2d_nchw_implicit_gemm, py::return_value_policy::move, "Conv2d NCHW (implicit GEMM)");
    m.def(
        "conv2d_nchw_torch",
        [](const py::object& input,
           const py::object& weight,
           const py::object& bias,
           const int stride_h,
           const int stride_w,
           const int pad_h,
           const int pad_w,
           const int dilation_h,
           const int dilation_w,
           const int groups) {
            return py_conv2d_torch_impl(
                wgkernel::cuda::conv2d_nchw,
                input,
                weight,
                bias,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dilation_h,
                dilation_w,
                groups);
        },
        "input"_a,
        "weight"_a,
        "bias"_a = py::none(),
        "stride_h"_a = 1,
        "stride_w"_a = 1,
        "pad_h"_a = 0,
        "pad_w"_a = 0,
        "dilation_h"_a = 1,
        "dilation_w"_a = 1,
        "groups"_a = 1,
        "Conv2d NCHW on CUDA torch tensors (naive)");
    m.def(
        "conv2d_nchw_im2col_gemm_torch",
        [](const py::object& input,
           const py::object& weight,
           const py::object& bias,
           const int stride_h,
           const int stride_w,
           const int pad_h,
           const int pad_w,
           const int dilation_h,
           const int dilation_w,
           const int groups) {
            return py_conv2d_torch_impl(
                wgkernel::cuda::conv2d_nchw_im2col_gemm,
                input,
                weight,
                bias,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dilation_h,
                dilation_w,
                groups);
        },
        "input"_a,
        "weight"_a,
        "bias"_a = py::none(),
        "stride_h"_a = 1,
        "stride_w"_a = 1,
        "pad_h"_a = 0,
        "pad_w"_a = 0,
        "dilation_h"_a = 1,
        "dilation_w"_a = 1,
        "groups"_a = 1,
        "Conv2d NCHW on CUDA torch tensors (im2col+GEMM)");
    m.def(
        "conv2d_nchw_direct_tiled_torch",
        [](const py::object& input,
           const py::object& weight,
           const py::object& bias,
           const int stride_h,
           const int stride_w,
           const int pad_h,
           const int pad_w,
           const int dilation_h,
           const int dilation_w,
           const int groups) {
            return py_conv2d_torch_impl(
                wgkernel::cuda::conv2d_nchw_direct_tiled,
                input,
                weight,
                bias,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dilation_h,
                dilation_w,
                groups);
        },
        "input"_a,
        "weight"_a,
        "bias"_a = py::none(),
        "stride_h"_a = 1,
        "stride_w"_a = 1,
        "pad_h"_a = 0,
        "pad_w"_a = 0,
        "dilation_h"_a = 1,
        "dilation_w"_a = 1,
        "groups"_a = 1,
        "Conv2d NCHW on CUDA torch tensors (direct tiled)");
    m.def(
        "conv2d_nchw_implicit_gemm_torch",
        [](const py::object& input,
           const py::object& weight,
           const py::object& bias,
           const int stride_h,
           const int stride_w,
           const int pad_h,
           const int pad_w,
           const int dilation_h,
           const int dilation_w,
           const int groups) {
            return py_conv2d_torch_impl(
                wgkernel::cuda::conv2d_nchw_implicit_gemm,
                input,
                weight,
                bias,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dilation_h,
                dilation_w,
                groups);
        },
        "input"_a,
        "weight"_a,
        "bias"_a = py::none(),
        "stride_h"_a = 1,
        "stride_w"_a = 1,
        "pad_h"_a = 0,
        "pad_w"_a = 0,
        "dilation_h"_a = 1,
        "dilation_w"_a = 1,
        "groups"_a = 1,
        "Conv2d NCHW on CUDA torch tensors (implicit GEMM)");

    // Pooling
    m.def("maxpool2d_nchw", &py_maxpool2d_nchw, py::return_value_policy::move, "MaxPool2d NCHW");

    // Resize
    m.def("upsample_nearest2d_nchw", &py_upsample_nearest2d_nchw, py::return_value_policy::move, "Upsample nearest 2D NCHW");
    m.def("upsample_bilinear2d_nchw", &py_upsample_bilinear2d_nchw, py::return_value_policy::move, "Upsample bilinear 2D NCHW");
}
