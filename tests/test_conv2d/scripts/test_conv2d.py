import argparse
import os
import sys

import torch
import torch.nn.functional as F

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, TEST_ROOT)

from test_utils import add_device_argument, benchmark, load_wgkernel, torch_device


OP_CASES = {
    "naive": {},
    "im2col_gemm": {},
    "direct_tiled": {},
    "implicit_gemm": {},
}


def selected_ops(op_name: str, device_name: str) -> list[str]:
    if device_name == "cpu":
        if op_name not in {"all", "naive"}:
            raise RuntimeError("CPU backend currently supports only op-name 'naive'")
        return ["naive"]
    if op_name == "all":
        return list(OP_CASES.keys())
    return [op_name]


def torch_conv2d(input_tensor, weight_tensor, bias_tensor, case: dict):
    return F.conv2d(
        input_tensor,
        weight_tensor,
        bias=bias_tensor,
        stride=(case.get("stride_h", 1), case.get("stride_w", 1)),
        padding=(case.get("pad_h", 0), case.get("pad_w", 0)),
        dilation=(case.get("dilation_h", 1), case.get("dilation_w", 1)),
        groups=case.get("groups", 1),
    )


def wgkernel_conv2d(wgk, algo: str, input_tensor, weight_tensor, bias_tensor, case: dict, device_name: str):
    if device_name == "cpu":
        if algo != "naive":
            raise RuntimeError("CPU backend currently supports only naive conv2d")
        return torch.as_tensor(
            wgk.conv2d_nchw(
                input_tensor.cpu().numpy(),
                weight_tensor.cpu().numpy(),
                None if bias_tensor is None else bias_tensor.cpu().numpy(),
                stride_h=case.get("stride_h", 1),
                stride_w=case.get("stride_w", 1),
                pad_h=case.get("pad_h", 0),
                pad_w=case.get("pad_w", 0),
                dilation_h=case.get("dilation_h", 1),
                dilation_w=case.get("dilation_w", 1),
                groups=case.get("groups", 1),
            ),
            dtype=torch.float32,
            device=input_tensor.device,
        )

    conv2d_by_algo = {
        "naive": wgk.conv2d_nchw_torch,
        "im2col_gemm": wgk.conv2d_nchw_im2col_gemm_torch,
        "direct_tiled": wgk.conv2d_nchw_direct_tiled_torch,
        "implicit_gemm": wgk.conv2d_nchw_implicit_gemm_torch,
    }
    return conv2d_by_algo[algo](
        input_tensor,
        weight_tensor,
        bias_tensor,
        stride_h=case.get("stride_h", 1),
        stride_w=case.get("stride_w", 1),
        pad_h=case.get("pad_h", 0),
        pad_w=case.get("pad_w", 0),
        dilation_h=case.get("dilation_h", 1),
        dilation_w=case.get("dilation_w", 1),
        groups=case.get("groups", 1),
    )


def make_tensors(case: dict):
    groups = case.get("groups", 1)
    use_bias = case.get("bias", True)
    generator = torch.Generator().manual_seed(case["seed"])

    input_cpu = torch.randn(case["n"], case["c_in"], case["h_in"], case["w_in"], generator=generator)
    weight_cpu = torch.randn(
        case["c_out"],
        case["c_in"] // groups,
        case["k_h"],
        case["k_w"],
        generator=generator,
    )
    bias_cpu = torch.randn(case["c_out"], generator=generator) if use_bias else None
    return input_cpu, weight_cpu, bias_cpu


def test_op_conv2d(wgk, case: dict, algo: str, device_name: str, profile: bool = False) -> None:
    print(f"   algo {algo:<14} case {case['name']}")
    input_cpu, weight_cpu, bias_cpu = make_tensors(case)
    reference = torch_conv2d(input_cpu, weight_cpu, bias_cpu, case)

    device = torch_device(device_name)
    input_cuda = input_cpu.to(device)
    weight_cuda = weight_cpu.to(device)
    bias_cuda = bias_cpu.to(device) if bias_cpu is not None else None
    actual = wgkernel_conv2d(wgk, algo, input_cuda, weight_cuda, bias_cuda, case, device_name).cpu()

    if not torch.allclose(actual, reference, atol=1e-3, rtol=1e-4):
        max_diff = (actual - reference).abs().max().item()
        raise AssertionError(
            f"conv2d mismatch for case {case['name']} algo={algo}: max abs diff={max_diff}"
        )

    if profile:
        torch_ms, wgkernel_ms = benchmark(
            lambda: torch_conv2d(input_cuda, weight_cuda, bias_cuda, case),
            lambda: wgkernel_conv2d(wgk, algo, input_cuda, weight_cuda, bias_cuda, case, device_name),
            device_name,
        )
        speedup = torch_ms / wgkernel_ms if wgkernel_ms > 0 else float("inf")
        print(f"      torch={torch_ms:.4f} ms wgkernel={wgkernel_ms:.4f} ms speedup={speedup:.3f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir", default=None)
    add_device_argument(parser)
    parser.add_argument("--op-name", default="all", choices=["all", *OP_CASES.keys()])
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    wgk = load_wgkernel(args.module_dir, args.device)
    test_cases = [
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
        {"name": "yolox_stem", "seed": 9, "n": 1, "c_in": 16, "h_in": 64, "w_in": 64,
         "c_out": 32, "k_h": 3, "k_w": 3, "stride_h": 1, "stride_w": 1,
         "pad_h": 1, "pad_w": 1},
        {"name": "imagenet_stem", "seed": 11, "n": 1, "c_in": 3, "h_in": 224, "w_in": 224,
         "c_out": 64, "k_h": 7, "k_w": 7, "stride_h": 2, "stride_w": 2,
         "pad_h": 3, "pad_w": 3},
        {"name": "asymmetric_pad", "seed": 10, "n": 1, "c_in": 4, "h_in": 13, "w_in": 17,
         "c_out": 4, "k_h": 3, "k_w": 3, "pad_h": 0, "pad_w": 1},
    ]
    algos = selected_ops(args.op_name, args.device)

    print(f"Testing conv2d {args.op_name} on {args.device}")
    for algo in algos:
        for case in test_cases:
            test_op_conv2d(wgk, case, algo, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
