#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "cpu/activation.hpp"
#include "cpu/attention.hpp"
#include "cpu/convolution.hpp"
#include "cpu/elementwise.hpp"
#include "cpu/embedding_indexing.hpp"
#include "cpu/gemm.hpp"
#include "cpu/norm.hpp"
#include "cpu/pooling.hpp"
#include "cpu/reduce.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

py::array_t<float> require_float32_array(const py::array_t<float, py::array::c_style | py::array::forcecast>& input) {
    py::buffer_info info = input.request();
    if (info.ndim == 0 || info.size <= 0) {
        throw std::runtime_error("input must have at least one element");
    }
    return input;
}

py::buffer_info require_float32_ndim(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& input,
    const char* name,
    const int ndim) {
    py::buffer_info info = input.request();
    if (info.ndim != ndim) {
        throw std::runtime_error(std::string(name) + " must be a rank-" + std::to_string(ndim) + " array");
    }
    if (info.size <= 0) {
        throw std::runtime_error(std::string(name) + " must have at least one element");
    }
    return info;
}

py::buffer_info require_int64_array(
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& input,
    const char* name) {
    py::buffer_info info = input.request();
    if (info.ndim == 0 || info.size <= 0) {
        throw std::runtime_error(std::string(name) + " must have at least one element");
    }
    return info;
}

py::array_t<float> py_silu(const py::array_t<float, py::array::c_style | py::array::forcecast>& input) {
    const py::array_t<float> values = require_float32_array(input);
    const py::buffer_info info = values.request();
    py::array_t<float> output(info.size);
    wgkernel::cpu::silu(
        static_cast<const float*>(info.ptr),
        static_cast<float*>(output.request().ptr),
        static_cast<std::int64_t>(info.size));
    return output;
}

py::array_t<float> py_sigmoid(const py::array_t<float, py::array::c_style | py::array::forcecast>& input) {
    const py::array_t<float> values = require_float32_array(input);
    const py::buffer_info info = values.request();
    py::array_t<float> output(info.size);
    wgkernel::cpu::sigmoid(
        static_cast<const float*>(info.ptr),
        static_cast<float*>(output.request().ptr),
        static_cast<std::int64_t>(info.size));
    return output;
}

py::array_t<float> py_exp(const py::array_t<float, py::array::c_style | py::array::forcecast>& input) {
    const py::array_t<float> values = require_float32_array(input);
    const py::buffer_info info = values.request();
    py::array_t<float> output(info.size);
    wgkernel::cpu::exp(
        static_cast<const float*>(info.ptr),
        static_cast<float*>(output.request().ptr),
        static_cast<std::int64_t>(info.size));
    return output;
}

py::array_t<float> py_add(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& lhs,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& rhs) {
    const py::array_t<float> lhs_values = require_float32_array(lhs);
    const py::array_t<float> rhs_values = require_float32_array(rhs);
    const py::buffer_info lhs_info = lhs_values.request();
    const py::buffer_info rhs_info = rhs_values.request();

    if (lhs_info.ndim != rhs_info.ndim) {
        throw std::runtime_error("lhs and rhs must have the same rank");
    }
    for (int dim = 0; dim < lhs_info.ndim; ++dim) {
        if (lhs_info.shape[dim] != rhs_info.shape[dim]) {
            throw std::runtime_error("lhs and rhs must have the same shape");
        }
    }

    py::array_t<float> output(lhs_info.shape);
    wgkernel::cpu::add(
        static_cast<const float*>(lhs_info.ptr),
        static_cast<const float*>(rhs_info.ptr),
        static_cast<float*>(output.request().ptr),
        static_cast<std::int64_t>(lhs_info.size));
    return output;
}

py::array_t<float> py_embedding(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& weight,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& indices) {
    const py::buffer_info weight_info = require_float32_ndim(weight, "weight", 2);
    const py::buffer_info indices_info = require_int64_array(indices, "indices");

    const auto num_embeddings = static_cast<std::int64_t>(weight_info.shape[0]);
    const auto embedding_dim = static_cast<std::int64_t>(weight_info.shape[1]);
    const auto num_indices = static_cast<std::int64_t>(indices_info.size);

    std::vector<py::ssize_t> output_shape;
    output_shape.reserve(static_cast<std::size_t>(indices_info.ndim) + 1);
    for (int dim = 0; dim < indices_info.ndim; ++dim) {
        output_shape.push_back(indices_info.shape[dim]);
    }
    output_shape.push_back(weight_info.shape[1]);

    py::array_t<float> output(output_shape);
    wgkernel::cpu::embedding(
        static_cast<const float*>(weight_info.ptr),
        static_cast<const std::int64_t*>(indices_info.ptr),
        static_cast<float*>(output.request().ptr),
        num_indices,
        num_embeddings,
        embedding_dim);
    return output;
}

py::array_t<float> py_sgemm(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& lhs,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& rhs) {
    const py::buffer_info lhs_info = require_float32_ndim(lhs, "lhs", 2);
    const py::buffer_info rhs_info = require_float32_ndim(rhs, "rhs", 2);

    const auto m = static_cast<std::int64_t>(lhs_info.shape[0]);
    const auto k = static_cast<std::int64_t>(lhs_info.shape[1]);
    const auto rhs_k = static_cast<std::int64_t>(rhs_info.shape[0]);
    const auto n = static_cast<std::int64_t>(rhs_info.shape[1]);

    if (k != rhs_k) {
        throw std::runtime_error("lhs.shape[1] must equal rhs.shape[0]");
    }

    py::array_t<float> output({m, n});
    wgkernel::cpu::sgemm(
        static_cast<const float*>(lhs_info.ptr),
        static_cast<const float*>(rhs_info.ptr),
        static_cast<float*>(output.request().ptr),
        m,
        n,
        k);
    return output;
}

py::array_t<float> py_batchnorm2d_inference_nchw(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& input,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& scale,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& bias,
    int n,
    int c,
    int h,
    int w) {
    const py::buffer_info input_info = require_float32_ndim(input, "input", 4);
    const py::buffer_info scale_info = require_float32_ndim(scale, "scale", 1);
    const py::buffer_info bias_info = require_float32_ndim(bias, "bias", 1);

    const int inferred_n = static_cast<int>(input_info.shape[0]);
    const int inferred_c = static_cast<int>(input_info.shape[1]);
    const int inferred_h = static_cast<int>(input_info.shape[2]);
    const int inferred_w = static_cast<int>(input_info.shape[3]);

    if (n < 0) {
        n = inferred_n;
    }
    if (c < 0) {
        c = inferred_c;
    }
    if (h < 0) {
        h = inferred_h;
    }
    if (w < 0) {
        w = inferred_w;
    }
    if (n != inferred_n || c != inferred_c || h != inferred_h || w != inferred_w) {
        throw std::runtime_error("explicit n, c, h, w must match input shape");
    }
    if (scale_info.shape[0] != c || bias_info.shape[0] != c) {
        throw std::runtime_error("scale and bias length must equal input channels");
    }

    py::array_t<float> output({n, c, h, w});
    wgkernel::cpu::batchnorm2d_inference_nchw(
        static_cast<const float*>(input_info.ptr),
        static_cast<const float*>(scale_info.ptr),
        static_cast<const float*>(bias_info.ptr),
        static_cast<float*>(output.request().ptr),
        n,
        c,
        h,
        w);
    return output;
}

py::array_t<float> py_rmsnorm(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& input,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& weight,
    const float eps) {
    const py::array_t<float> values = require_float32_array(input);
    const py::buffer_info input_info = values.request();
    const py::buffer_info weight_info = require_float32_ndim(weight, "weight", 1);

    const auto hidden_size = static_cast<std::int64_t>(input_info.shape[input_info.ndim - 1]);
    if (weight_info.shape[0] != hidden_size) {
        throw std::runtime_error("weight length must equal input.shape[-1]");
    }
    const auto outer_size = static_cast<std::int64_t>(input_info.size) / hidden_size;

    py::array_t<float> output(input_info.shape);
    wgkernel::cpu::rmsnorm(
        static_cast<const float*>(input_info.ptr),
        static_cast<const float*>(weight_info.ptr),
        static_cast<float*>(output.request().ptr),
        outer_size,
        hidden_size,
        eps);
    return output;
}

py::array_t<float> py_maxpool2d_nchw(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& input,
    const int k_h,
    const int k_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w) {
    const py::buffer_info input_info = require_float32_ndim(input, "input", 4);
    const auto n = static_cast<std::int64_t>(input_info.shape[0]);
    const auto c = static_cast<std::int64_t>(input_info.shape[1]);
    const auto h_in = static_cast<std::int64_t>(input_info.shape[2]);
    const auto w_in = static_cast<std::int64_t>(input_info.shape[3]);

    if (k_h <= 0 || k_w <= 0 || stride_h <= 0 || stride_w <= 0) {
        throw std::runtime_error("kernel size and stride must be positive");
    }

    const std::int64_t h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    const std::int64_t w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;
    if (h_out <= 0 || w_out <= 0) {
        throw std::runtime_error("maxpool output shape must be positive");
    }

    py::array_t<float> output({n, c, h_out, w_out});
    wgkernel::cpu::maxpool2d_nchw(
        static_cast<const float*>(input_info.ptr),
        static_cast<float*>(output.request().ptr),
        n,
        c,
        h_in,
        w_in,
        h_out,
        w_out,
        k_h,
        k_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w);
    return output;
}

float py_reduce_sum(const py::array_t<float, py::array::c_style | py::array::forcecast>& input) {
    const py::array_t<float> values = require_float32_array(input);
    const py::buffer_info info = values.request();
    return wgkernel::cpu::reduce_sum(static_cast<const float*>(info.ptr), static_cast<std::int64_t>(info.size));
}

float py_reduce_max(const py::array_t<float, py::array::c_style | py::array::forcecast>& input) {
    const py::array_t<float> values = require_float32_array(input);
    const py::buffer_info info = values.request();
    return wgkernel::cpu::reduce_max(static_cast<const float*>(info.ptr), static_cast<std::int64_t>(info.size));
}

std::int64_t py_reduce_argmax(const py::array_t<float, py::array::c_style | py::array::forcecast>& input) {
    const py::array_t<float> values = require_float32_array(input);
    const py::buffer_info info = values.request();
    return wgkernel::cpu::reduce_argmax(static_cast<const float*>(info.ptr), static_cast<std::int64_t>(info.size));
}

py::array_t<float> py_softmax(const py::array_t<float, py::array::c_style | py::array::forcecast>& input) {
    const py::array_t<float> values = require_float32_array(input);
    const py::buffer_info info = values.request();
    py::array_t<float> output(info.shape);
    wgkernel::cpu::softmax(
        static_cast<const float*>(info.ptr),
        static_cast<float*>(output.request().ptr),
        static_cast<std::int64_t>(info.size));
    return output;
}

py::array_t<float> py_self_attention(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& query,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& key,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& value,
    const bool causal) {
    const py::buffer_info q_info = require_float32_ndim(query, "query", 3);
    const py::buffer_info k_info = require_float32_ndim(key, "key", 3);
    const py::buffer_info v_info = require_float32_ndim(value, "value", 3);

    const auto batch = static_cast<std::int64_t>(q_info.shape[0]);
    const auto query_length = static_cast<std::int64_t>(q_info.shape[1]);
    const auto head_dim = static_cast<std::int64_t>(q_info.shape[2]);
    const auto key_value_length = static_cast<std::int64_t>(k_info.shape[1]);
    const auto value_dim = static_cast<std::int64_t>(v_info.shape[2]);

    if (k_info.shape[0] != q_info.shape[0] || v_info.shape[0] != q_info.shape[0]) {
        throw std::runtime_error("query, key and value must have the same batch size");
    }
    if (k_info.shape[2] != q_info.shape[2]) {
        throw std::runtime_error("query and key must have the same head_dim");
    }
    if (v_info.shape[1] != k_info.shape[1]) {
        throw std::runtime_error("value length must match key length");
    }

    py::array_t<float> output({batch, query_length, value_dim});
    wgkernel::cpu::self_attention(
        static_cast<const float*>(q_info.ptr),
        static_cast<const float*>(k_info.ptr),
        static_cast<const float*>(v_info.ptr),
        static_cast<float*>(output.request().ptr),
        batch,
        query_length,
        key_value_length,
        head_dim,
        value_dim,
        causal);
    return output;
}

py::array_t<float> py_conv2d_nchw(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& input,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& weight,
    const py::object& bias,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups) {
    const py::buffer_info input_info = require_float32_ndim(input, "input", 4);
    const py::buffer_info weight_info = require_float32_ndim(weight, "weight", 4);

    const auto n = static_cast<int>(input_info.shape[0]);
    const auto c_in = static_cast<int>(input_info.shape[1]);
    const auto h_in = static_cast<int>(input_info.shape[2]);
    const auto w_in = static_cast<int>(input_info.shape[3]);
    const auto c_out = static_cast<int>(weight_info.shape[0]);
    const auto c_in_per_group = static_cast<int>(weight_info.shape[1]);
    const auto k_h = static_cast<int>(weight_info.shape[2]);
    const auto k_w = static_cast<int>(weight_info.shape[3]);

    if (stride_h <= 0 || stride_w <= 0 || dilation_h <= 0 || dilation_w <= 0 || groups <= 0) {
        throw std::runtime_error("stride, dilation and groups must be positive");
    }
    if (c_in % groups != 0 || c_out % groups != 0) {
        throw std::runtime_error("input channels and output channels must be divisible by groups");
    }
    if (c_in_per_group != c_in / groups) {
        throw std::runtime_error("weight.shape[1] must equal input channels / groups");
    }

    const int h_out = (h_in + 2 * pad_h - dilation_h * (k_h - 1) - 1) / stride_h + 1;
    const int w_out = (w_in + 2 * pad_w - dilation_w * (k_w - 1) - 1) / stride_w + 1;
    if (h_out <= 0 || w_out <= 0) {
        throw std::runtime_error("conv2d output shape must be positive");
    }

    const float* bias_ptr = nullptr;
    py::array_t<float, py::array::c_style | py::array::forcecast> bias_array;
    if (!bias.is_none()) {
        bias_array = bias.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        const py::buffer_info bias_info = require_float32_ndim(bias_array, "bias", 1);
        if (bias_info.shape[0] != c_out) {
            throw std::runtime_error("bias length must equal output channels");
        }
        bias_ptr = static_cast<const float*>(bias_info.ptr);
    }

    py::array_t<float> output({n, c_out, h_out, w_out});
    wgkernel::cpu::conv2d_nchw(
        static_cast<const float*>(input_info.ptr),
        static_cast<const float*>(weight_info.ptr),
        bias_ptr,
        static_cast<float*>(output.request().ptr),
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
        groups);
    return output;
}

PYBIND11_MODULE(wgkernel_cpu, m) {
    m.doc() = "wGKernels CPU baseline operator library";

    m.def("silu", &py_silu, py::return_value_policy::move, "SiLU activation over a float32 array");
    m.def("sigmoid", &py_sigmoid, py::return_value_policy::move, "Sigmoid activation over a float32 array");
    m.def("exp", &py_exp, py::return_value_policy::move, "Exp activation over a float32 array");
    m.def("add", &py_add, py::arg("lhs"), py::arg("rhs"), py::return_value_policy::move,
          "Elementwise add over same-shape float32 arrays");
    m.def("embedding", &py_embedding, py::arg("weight"), py::arg("indices"), py::return_value_policy::move,
          "Embedding lookup over float32 weights and int64 indices");
    m.def("sgemm", &py_sgemm, py::arg("lhs"), py::arg("rhs"), py::return_value_policy::move,
          "Naive row-major float32 matrix multiply: lhs[M,K] @ rhs[K,N]");
    m.def("batchnorm2d_inference_nchw", &py_batchnorm2d_inference_nchw, py::arg("input"), py::arg("scale"), py::arg("bias"),
          py::arg("n") = -1, py::arg("c") = -1, py::arg("h") = -1, py::arg("w") = -1, py::return_value_policy::move,
          "BatchNorm2d inference over NCHW float32 arrays using folded scale and bias");
    m.def("rmsnorm", &py_rmsnorm, py::arg("input"), py::arg("weight"), py::arg("eps") = 1.0e-6F,
          py::return_value_policy::move, "RMSNorm over the last dimension of a float32 array");
    m.def("maxpool2d_nchw", &py_maxpool2d_nchw, py::arg("input"), py::arg("k_h"), py::arg("k_w"), py::arg("stride_h") = 1,
          py::arg("stride_w") = 1, py::arg("pad_h") = 0, py::arg("pad_w") = 0, py::return_value_policy::move,
          "MaxPool2d over NCHW float32 arrays");
    m.def("reduce_sum", &py_reduce_sum, py::arg("input"), py::return_value_policy::move, "Reduce sum over a float32 array");
    m.def("reduce_max", &py_reduce_max, py::arg("input"), py::return_value_policy::move, "Reduce max over a float32 array");
    m.def("reduce_argmax", &py_reduce_argmax, py::arg("input"), py::return_value_policy::move, "Reduce argmax over a float32 array");
    m.def("softmax", &py_softmax, py::arg("input"), py::return_value_policy::move, "Softmax over a float32 array");
    m.def("self_attention", &py_self_attention, py::arg("query"), py::arg("key"), py::arg("value"), py::arg("causal") = false,
          py::return_value_policy::move, "Scaled dot-product attention over float32 arrays");
    m.def("conv2d_nchw", &py_conv2d_nchw, py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("pad_h") = 0, py::arg("pad_w") = 0,
          py::arg("dilation_h") = 1, py::arg("dilation_w") = 1, py::arg("groups") = 1, py::return_value_policy::move,
          "Naive NCHW conv2d over float32 arrays");
}
