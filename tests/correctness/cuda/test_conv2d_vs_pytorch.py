import argparse
import os
import subprocess
import tempfile

import torch
import torch.nn.functional as F


def write_tensor(path: str, tensor: torch.Tensor) -> None:
    tensor.detach().cpu().contiguous().numpy().astype("float32").tofile(path)


def read_tensor(path: str, shape) -> torch.Tensor:
    import numpy as np

    data = np.fromfile(path, dtype="float32")
    return torch.from_numpy(data).reshape(shape)


def run_case(tool_path: str, workdir: str, case: dict, algo: str) -> None:
    n = case["n"]
    c_in = case["c_in"]
    h_in = case["h_in"]
    w_in = case["w_in"]
    c_out = case["c_out"]
    k_h = case["k_h"]
    k_w = case["k_w"]
    stride_h = case.get("stride_h", 1)
    stride_w = case.get("stride_w", 1)
    pad_h = case.get("pad_h", 0)
    pad_w = case.get("pad_w", 0)
    dilation_h = case.get("dilation_h", 1)
    dilation_w = case.get("dilation_w", 1)
    groups = case.get("groups", 1)
    use_bias = case.get("bias", True)

    generator = torch.Generator().manual_seed(case["seed"])
    input_tensor = torch.randn(n, c_in, h_in, w_in, generator=generator)
    weight_tensor = torch.randn(c_out, c_in // groups, k_h, k_w, generator=generator)
    bias_tensor = torch.randn(c_out, generator=generator) if use_bias else None

    reference = F.conv2d(
        input_tensor,
        weight_tensor,
        bias=bias_tensor,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation_h, dilation_w),
        groups=groups,
    )

    input_path = os.path.join(workdir, "input.bin")
    weight_path = os.path.join(workdir, "weight.bin")
    bias_path = os.path.join(workdir, "bias.bin")
    output_path = os.path.join(workdir, "output.bin")

    write_tensor(input_path, input_tensor)
    write_tensor(weight_path, weight_tensor)
    if bias_tensor is not None:
        write_tensor(bias_path, bias_tensor)

    command = [
        tool_path,
        "--algo", algo,
        "--n", str(n),
        "--c_in", str(c_in),
        "--h_in", str(h_in),
        "--w_in", str(w_in),
        "--c_out", str(c_out),
        "--k_h", str(k_h),
        "--k_w", str(k_w),
        "--stride_h", str(stride_h),
        "--stride_w", str(stride_w),
        "--pad_h", str(pad_h),
        "--pad_w", str(pad_w),
        "--dilation_h", str(dilation_h),
        "--dilation_w", str(dilation_w),
        "--groups", str(groups),
        "--input", input_path,
        "--weight", weight_path,
        "--output", output_path,
    ]
    if bias_tensor is not None:
        command += ["--bias", bias_path]

    subprocess.run(command, capture_output=True, check=True, text=True)

    actual = read_tensor(output_path, reference.shape)
    if not torch.allclose(actual, reference, atol=1e-3, rtol=1e-4):
        max_diff = (actual - reference).abs().max().item()
        raise AssertionError(
            f"conv2d mismatch for case {case['name']} algo={algo}: max abs diff={max_diff}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the PyTorch conv2d test")

    cases = [
        {"name": "plain_3x3", "seed": 1, "n": 2, "c_in": 3, "h_in": 16, "w_in": 16,
         "c_out": 8, "k_h": 3, "k_w": 3, "pad_h": 1, "pad_w": 1},
        {"name": "pointwise_1x1", "seed": 2, "n": 1, "c_in": 8, "h_in": 10, "w_in": 10,
         "c_out": 16, "k_h": 1, "k_w": 1},
        {"name": "strided_3x3", "seed": 3, "n": 2, "c_in": 4, "h_in": 17, "w_in": 19,
         "c_out": 6, "k_h": 3, "k_w": 3, "stride_h": 2, "stride_w": 2, "pad_h": 1, "pad_w": 1},
        {"name": "dilated_3x3", "seed": 4, "n": 1, "c_in": 3, "h_in": 20, "w_in": 20,
         "c_out": 5, "k_h": 3, "k_w": 3, "pad_h": 2, "pad_w": 2,
         "dilation_h": 2, "dilation_w": 2},
        {"name": "depthwise_3x3", "seed": 5, "n": 1, "c_in": 8, "h_in": 12, "w_in": 12,
         "c_out": 8, "k_h": 3, "k_w": 3, "pad_h": 1, "pad_w": 1, "groups": 8},
        {"name": "grouped_3x3", "seed": 6, "n": 2, "c_in": 6, "h_in": 9, "w_in": 9,
         "c_out": 12, "k_h": 3, "k_w": 3, "pad_h": 1, "pad_w": 1, "groups": 3},
        {"name": "nonsquare_kernel", "seed": 7, "n": 1, "c_in": 3, "h_in": 14, "w_in": 18,
         "c_out": 4, "k_h": 3, "k_w": 5, "pad_h": 1, "pad_w": 2},
        {"name": "no_bias", "seed": 8, "n": 1, "c_in": 3, "h_in": 16, "w_in": 16,
         "c_out": 8, "k_h": 3, "k_w": 3, "pad_h": 1, "pad_w": 1, "bias": False},
        # Larger shape: catches tiling / shared-memory capacity issues that
        # the toy shapes above may paper over.
        {"name": "yolox_stem", "seed": 9, "n": 1, "c_in": 16, "h_in": 64, "w_in": 64,
         "c_out": 32, "k_h": 3, "k_w": 3, "stride_h": 1, "stride_w": 1,
         "pad_h": 1, "pad_w": 1},
        # Asymmetric padding: edges the implicit GEMM gather path.
        {"name": "asymmetric_pad", "seed": 10, "n": 1, "c_in": 4, "h_in": 13, "w_in": 17,
         "c_out": 4, "k_h": 3, "k_w": 3, "pad_h": 0, "pad_w": 1},
    ]

    algos = ["naive", "im2col_gemm", "direct_tiled", "implicit_gemm"]

    with tempfile.TemporaryDirectory() as workdir:
        for algo in algos:
            for case in cases:
                run_case(args.tool, workdir, case, algo)

    print("PyTorch conv2d alignment check passed")


if __name__ == "__main__":
    main()
