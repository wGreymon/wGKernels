import argparse
import os
import sys

import torch

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, TEST_ROOT)

from test_utils import add_device_argument, load_wgkernel, torch_device


OP_CASES = {
    "add": {
        "wgkernel": "add",
        "torch": torch.add,
        "atol": 1e-6,
        "rtol": 1e-6,
    },
}


def selected_ops(op_name: str) -> list[str]:
    if op_name == "all":
        return list(OP_CASES.keys())
    return [op_name]


def make_inputs(shape: tuple[int, ...], seed: int, device_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch_device(device_name)
    generator = torch.Generator(device=device).manual_seed(seed)
    lhs = torch.empty(shape, device=device, dtype=torch.float32).uniform_(-6.0, 6.0, generator=generator)
    rhs = torch.empty(shape, device=device, dtype=torch.float32).uniform_(-6.0, 6.0, generator=generator)
    return lhs, rhs


def input_for_wgkernel(values: torch.Tensor, device_name: str):
    if device_name == "cpu":
        return values.cpu().numpy()
    return values


def assert_allclose(name: str, actual, expected: torch.Tensor, atol: float = 1e-6, rtol: float = 1e-6) -> None:
    actual_tensor = torch.as_tensor(actual, dtype=torch.float32, device=expected.device).reshape(expected.shape)
    if not torch.allclose(actual_tensor, expected, atol=atol, rtol=rtol):
        max_diff = (actual_tensor - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch: max abs diff={max_diff}")


def test_one_elementwise(wgk, op_name: str, shape: tuple[int, ...], seed: int, device_name: str) -> None:
    print(f"   op {op_name:<8} shape {shape} seed {seed}")
    op_case = OP_CASES[op_name]
    lhs, rhs = make_inputs(shape, seed, device_name)

    wgkernel_func = getattr(wgk, op_case["wgkernel"])
    actual = wgkernel_func(input_for_wgkernel(lhs, device_name), input_for_wgkernel(rhs, device_name))
    expected = op_case["torch"](lhs, rhs)
    assert_allclose(op_name, actual, expected, atol=op_case["atol"], rtol=op_case["rtol"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir", default=None)
    add_device_argument(parser)
    parser.add_argument("--op-name", default="all", choices=["all", *OP_CASES.keys()])
    args = parser.parse_args()

    wgk = load_wgkernel(args.module_dir, args.device)
    test_cases = [
        ((1,), 11),
        ((7,), 23),
        ((257,), 31),
        ((4, 8), 47),
        ((2, 3, 5), 59),
        ((2, 3, 4, 5), 71),
        ((65536,), 89),
        ((1024, 1024), 113),
    ]

    print(f"Testing elementwise {args.op_name} on {args.device}")
    for op_name in selected_ops(args.op_name):
        for shape, seed in test_cases:
            test_one_elementwise(wgk, op_name, shape, seed, args.device)

    print("\033[92mTest passed!\033[0m\n")
