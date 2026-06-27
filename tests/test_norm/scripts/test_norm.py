import argparse
import os
import sys

import torch

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, TEST_ROOT)

from test_utils import add_device_argument, load_wgkernel, torch_device


OP_CASES = {
    "batchnorm2d_inference_nchw": {
        "atol": 1e-6,
        "rtol": 1e-6,
    },
    "rmsnorm": {
        "atol": 1e-5,
        "rtol": 1e-5,
    },
}


def selected_ops(op_name: str) -> list[str]:
    if op_name == "all":
        return list(OP_CASES.keys())
    return [op_name]


def input_for_wgkernel(tensor: torch.Tensor, device_name: str):
    if device_name == "cpu":
        return tensor.cpu().numpy()
    return tensor


def assert_allclose(name: str, actual, expected: torch.Tensor, atol: float, rtol: float) -> None:
    actual_tensor = torch.as_tensor(actual, dtype=torch.float32, device=expected.device).reshape(expected.shape)
    if not torch.allclose(actual_tensor, expected, atol=atol, rtol=rtol):
        max_diff = (actual_tensor - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch: max abs diff={max_diff}")


def make_batchnorm_inputs(case: dict, device_name: str):
    device = torch_device(device_name)
    generator = torch.Generator(device=device).manual_seed(case["seed"])
    values = torch.randn(case["n"], case["c"], case["h"], case["w"], device=device, generator=generator)
    scale = torch.randn(case["c"], device=device, generator=generator)
    bias = torch.randn(case["c"], device=device, generator=generator)
    return values, scale, bias


def test_batchnorm2d_inference_nchw(wgk, case: dict, device_name: str) -> None:
    print(f"   op batchnorm2d_inference_nchw case {case['name']} shape=({case['n']}, {case['c']}, {case['h']}, {case['w']})")
    values, scale, bias = make_batchnorm_inputs(case, device_name)
    expected = values * scale.reshape(1, case["c"], 1, 1) + bias.reshape(1, case["c"], 1, 1)
    actual = wgk.batchnorm2d_inference_nchw(
        input_for_wgkernel(values, device_name),
        input_for_wgkernel(scale, device_name),
        input_for_wgkernel(bias, device_name),
    )
    op_case = OP_CASES["batchnorm2d_inference_nchw"]
    assert_allclose("batchnorm2d_inference_nchw", actual, expected, op_case["atol"], op_case["rtol"])


def make_rmsnorm_inputs(case: dict, device_name: str):
    device = torch_device(device_name)
    generator = torch.Generator(device=device).manual_seed(case["seed"])
    values = torch.randn(case["shape"], device=device, generator=generator)
    weight = torch.randn(case["shape"][-1], device=device, generator=generator)
    return values, weight


def test_rmsnorm(wgk, case: dict, device_name: str) -> None:
    print(f"   op rmsnorm case {case['name']} shape={case['shape']} eps={case['eps']}")
    values, weight = make_rmsnorm_inputs(case, device_name)
    inv_rms = torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + case["eps"])
    expected = values * inv_rms * weight
    actual = wgk.rmsnorm(input_for_wgkernel(values, device_name), input_for_wgkernel(weight, device_name), case["eps"])
    op_case = OP_CASES["rmsnorm"]
    assert_allclose("rmsnorm", actual, expected, op_case["atol"], op_case["rtol"])


def test_one_norm(wgk, op_name: str, device_name: str) -> None:
    if op_name == "batchnorm2d_inference_nchw":
        test_cases = [
            {"name": "tiny", "seed": 11, "n": 1, "c": 1, "h": 1, "w": 1},
            {"name": "plain", "seed": 23, "n": 2, "c": 3, "h": 4, "w": 5},
            {"name": "channel_heavy", "seed": 31, "n": 1, "c": 32, "h": 7, "w": 9},
            {"name": "batch_heavy", "seed": 47, "n": 8, "c": 4, "h": 3, "w": 3},
            {"name": "throughput_like", "seed": 53, "n": 8, "c": 64, "h": 56, "w": 56},
        ]
        for case in test_cases:
            test_batchnorm2d_inference_nchw(wgk, case, device_name)
        return

    if op_name == "rmsnorm":
        test_cases = [
            {"name": "vector", "seed": 59, "shape": (8,), "eps": 1.0e-6},
            {"name": "matrix", "seed": 71, "shape": (4, 16), "eps": 1.0e-6},
            {"name": "batch_sequence", "seed": 89, "shape": (2, 5, 32), "eps": 1.0e-5},
            {"name": "wide_hidden", "seed": 97, "shape": (3, 257), "eps": 1.0e-6},
            {"name": "throughput_like", "seed": 101, "shape": (16, 512, 768), "eps": 1.0e-6},
        ]
        for case in test_cases:
            test_rmsnorm(wgk, case, device_name)
        return

    raise ValueError(f"unsupported op: {op_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir", default=None)
    add_device_argument(parser)
    parser.add_argument("--op-name", default="all", choices=["all", *OP_CASES.keys()])
    args = parser.parse_args()

    wgk = load_wgkernel(args.module_dir, args.device)

    print(f"Testing norm {args.op_name} on {args.device}")
    for op_name in selected_ops(args.op_name):
        test_one_norm(wgk, op_name, args.device)

    print("\033[92mTest passed!\033[0m\n")
