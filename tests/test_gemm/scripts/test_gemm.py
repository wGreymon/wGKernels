import argparse
import os
import sys

import torch

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, TEST_ROOT)

from test_utils import add_device_argument, load_wgkernel, torch_device


OP_CASES = {
    "sgemm": {
        "wgkernel": "sgemm",
        "atol": 1e-4,
        "rtol": 1e-4,
    },
}


def selected_ops(op_name: str) -> list[str]:
    if op_name == "all":
        return list(OP_CASES.keys())
    return [op_name]


def make_inputs(case: dict, device_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch_device(device_name)
    generator = torch.Generator(device=device).manual_seed(case["seed"])
    lhs = torch.randn(case["m"], case["k"], device=device, generator=generator)
    rhs = torch.randn(case["k"], case["n"], device=device, generator=generator)
    return lhs, rhs


def input_for_wgkernel(tensor: torch.Tensor, device_name: str):
    if device_name == "cpu":
        return tensor.cpu().numpy()
    return tensor


def assert_allclose(name: str, actual, expected: torch.Tensor, atol: float = 1e-4, rtol: float = 1e-4) -> None:
    actual_tensor = torch.as_tensor(actual, dtype=torch.float32, device=expected.device).reshape(expected.shape)
    if not torch.allclose(actual_tensor, expected, atol=atol, rtol=rtol):
        max_diff = (actual_tensor - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch: max abs diff={max_diff}")


def test_sgemm(wgk, case: dict, device_name: str) -> None:
    print(f"   op sgemm case {case['name']} m={case['m']} n={case['n']} k={case['k']}")
    lhs, rhs = make_inputs(case, device_name)
    expected = torch.matmul(lhs, rhs)
    actual = wgk.sgemm(input_for_wgkernel(lhs, device_name), input_for_wgkernel(rhs, device_name))
    assert_allclose("sgemm", actual, expected, atol=OP_CASES["sgemm"]["atol"], rtol=OP_CASES["sgemm"]["rtol"])


def test_one_gemm(wgk, op_name: str, case: dict, device_name: str) -> None:
    if op_name == "sgemm":
        test_sgemm(wgk, case, device_name)
        return
    raise ValueError(f"unsupported op: {op_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir", default=None)
    add_device_argument(parser)
    parser.add_argument("--op-name", default="all", choices=["all", *OP_CASES.keys()])
    args = parser.parse_args()

    wgk = load_wgkernel(args.module_dir, args.device)
    test_cases = [
        {"name": "scalar_like", "seed": 11, "m": 1, "n": 1, "k": 1},
        {"name": "row_vector", "seed": 23, "m": 1, "n": 7, "k": 5},
        {"name": "col_vector", "seed": 31, "m": 7, "n": 1, "k": 5},
        {"name": "small_square", "seed": 47, "m": 8, "n": 8, "k": 8},
        {"name": "rectangular_mn", "seed": 59, "m": 16, "n": 9, "k": 13},
        {"name": "rectangular_k", "seed": 71, "m": 9, "n": 17, "k": 32},
        {"name": "medium", "seed": 89, "m": 64, "n": 48, "k": 96},
        {"name": "throughput_like", "seed": 113, "m": 256, "n": 256, "k": 256},
    ]

    print(f"Testing gemm {args.op_name} on {args.device}")
    for op_name in selected_ops(args.op_name):
        for case in test_cases:
            test_one_gemm(wgk, op_name, case, args.device)

    print("\033[92mTest passed!\033[0m\n")
