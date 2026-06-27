import argparse
import os
import sys

import torch

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, TEST_ROOT)

from test_utils import add_device_argument, load_wgkernel, torch_device


OP_CASES = {
    "silu": {
        "wgkernel": "silu",
        "torch": torch.nn.functional.silu,
        "atol": 1e-6,
        "rtol": 1e-6,
    },
    "sigmoid": {
        "wgkernel": "sigmoid",
        "torch": torch.sigmoid,
        "atol": 1e-6,
        "rtol": 1e-6,
    },
    "exp": {
        "wgkernel": "exp",
        "torch": torch.exp,
        "atol": 1e-5,
        "rtol": 1e-6,
    },
}


def selected_ops(op_name: str) -> list[str]:
    if op_name == "all":
        return list(OP_CASES.keys())
    return [op_name]


def make_input(numel: int, seed: int, device_name: str) -> torch.Tensor:
    device = torch_device(device_name)
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.empty(numel, device=device, dtype=torch.float32).uniform_(-6.0, 6.0, generator=generator)


def input_for_wgkernel(values: torch.Tensor, device_name: str):
    if device_name == "cpu":
        return values.cpu().numpy()
    return values


def assert_allclose(name: str, actual, expected: torch.Tensor, atol: float = 1e-6, rtol: float = 1e-6) -> None:
    actual_tensor = torch.as_tensor(actual, dtype=torch.float32, device=expected.device).reshape(expected.shape)
    if not torch.allclose(actual_tensor, expected, atol=atol, rtol=rtol):
        max_diff = (actual_tensor - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch: max abs diff={max_diff}")


def test_one_activation(wgk, op_name: str, numel: int, seed: int, device_name: str) -> None:
    print(f"   op {op_name:<8} numel {numel} seed {seed}")
    op_case = OP_CASES[op_name]
    values = make_input(numel, seed, device_name)
    wgk_input = input_for_wgkernel(values, device_name)

    wgkernel_func = getattr(wgk, op_case["wgkernel"])
    actual = wgkernel_func(wgk_input)
    expected = op_case["torch"](values)
    assert_allclose(op_name, actual, expected, atol=op_case["atol"], rtol=op_case["rtol"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir", default=None)
    add_device_argument(parser)
    parser.add_argument("--op-name", default="all", choices=["all", *OP_CASES.keys()])
    args = parser.parse_args()

    wgk = load_wgkernel(args.module_dir, args.device)
    test_cases = [
        (1, 11),
        (7, 23),
        (257, 31),
        (4099, 47),
        (65536, 89),
        (1048576, 113),
    ]

    print(f"Testing activation {args.op_name} on {args.device}")
    for op_name in selected_ops(args.op_name):
        for numel, seed in test_cases:
            test_one_activation(wgk, op_name, numel, seed, args.device)

    print("\033[92mTest passed!\033[0m\n")
