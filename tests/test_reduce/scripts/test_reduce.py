import argparse
import math
import os
import sys

import torch

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, TEST_ROOT)

from test_utils import add_device_argument, benchmark, load_wgkernel, torch_device


OP_CASES = {
    "sum": {},
    "max": {},
    "argmax": {},
    "softmax": {},
}


def selected_ops(op_name: str, device_name: str) -> list[str]:
    if device_name == "cuda":
        cuda_ops = ["sum", "max", "argmax"]
        if op_name == "all":
            return cuda_ops
        if op_name == "softmax":
            raise RuntimeError("CUDA backend does not expose reduce softmax yet")
        return [op_name]
    if op_name == "all":
        return list(OP_CASES.keys())
    return [op_name]


def make_input(numel: int, seed: int, device_name: str) -> torch.Tensor:
    device = torch_device(device_name)
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.empty(numel, device=device, dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator)


def assert_close(actual: float, expected: float, atol: float, rtol: float) -> None:
    if math.fabs(actual - expected) > atol + rtol * math.fabs(expected):
        raise AssertionError(f"actual={actual}, expected={expected}")


def reduce_input_for_wgkernel(values: torch.Tensor, device_name: str):
    if device_name == "cpu":
        return values.cpu().numpy()
    return values


def wgkernel_reduce_sum(wgk, values: torch.Tensor, device_name: str) -> float:
    if device_name == "cuda":
        return wgk.reduce_sum_torch(values)
    return wgk.reduce_sum(reduce_input_for_wgkernel(values, device_name))


def wgkernel_reduce_max(wgk, values: torch.Tensor, device_name: str) -> float:
    if device_name == "cuda":
        return wgk.reduce_max_torch(values)
    return wgk.reduce_max(reduce_input_for_wgkernel(values, device_name))


def wgkernel_reduce_argmax(wgk, values: torch.Tensor, device_name: str) -> int:
    if device_name == "cuda":
        return int(wgk.reduce_argmax_torch(values))
    return int(wgk.reduce_argmax(reduce_input_for_wgkernel(values, device_name)))


def wgkernel_softmax(wgk, values: torch.Tensor, device_name: str):
    if device_name == "cuda":
        raise RuntimeError("CUDA backend does not expose reduce softmax yet")
    return wgk.softmax(reduce_input_for_wgkernel(values, device_name))


def assert_allclose(name: str, actual, expected: torch.Tensor, atol: float, rtol: float) -> None:
    actual_tensor = torch.as_tensor(actual, dtype=torch.float32, device=expected.device).reshape(expected.shape)
    if not torch.allclose(actual_tensor, expected, atol=atol, rtol=rtol):
        max_diff = (actual_tensor - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch: max abs diff={max_diff}")


def test_one_reduce(wgk, op_name: str, numel: int, seed: int, device_name: str, profile: bool = False) -> None:
    print(f"   op {op_name:<8} numel {numel} seed {seed}")
    values = make_input(numel, seed, device_name)

    if op_name == "sum":
        actual_sum = wgkernel_reduce_sum(wgk, values, device_name)
        expected_sum = float(values.sum().item())
        assert_close(actual_sum, expected_sum, atol=1e-2, rtol=1e-4)
    elif op_name == "max":
        actual_max = wgkernel_reduce_max(wgk, values, device_name)
        expected_max = float(values.max().item())
        assert_close(actual_max, expected_max, atol=1e-6, rtol=1e-6)
    elif op_name == "argmax":
        actual_argmax = wgkernel_reduce_argmax(wgk, values, device_name)
        expected_argmax = int(values.argmax().item())
        if actual_argmax != expected_argmax:
            raise AssertionError(f"argmax mismatch: actual={actual_argmax}, expected={expected_argmax}")
    elif op_name == "softmax":
        actual_softmax = wgkernel_softmax(wgk, values, device_name)
        expected_softmax = torch.softmax(values, dim=0)
        assert_allclose("softmax", actual_softmax, expected_softmax, atol=1e-6, rtol=1e-6)
    else:
        raise ValueError(f"unsupported op: {op_name}")

    if profile:
        torch_func_by_op = {
            "sum": lambda: values.sum().item(),
            "max": lambda: values.max().item(),
            "argmax": lambda: values.argmax().item(),
            "softmax": lambda: torch.softmax(values, dim=0),
        }
        wgkernel_func_by_op = {
            "sum": lambda: wgkernel_reduce_sum(wgk, values, device_name),
            "max": lambda: wgkernel_reduce_max(wgk, values, device_name),
            "argmax": lambda: wgkernel_reduce_argmax(wgk, values, device_name),
            "softmax": lambda: wgkernel_softmax(wgk, values, device_name),
        }
        torch_ms, wgkernel_ms = benchmark(torch_func_by_op[op_name], wgkernel_func_by_op[op_name], device_name)
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
    test_cases = [(1, 11), (7, 23), (257, 31), (4099, 47), (65536, 89), (1048576, 113), (4194304, 127)]

    print(f"Testing reduce {args.op_name} on {args.device}")
    for op_name in selected_ops(args.op_name, args.device):
        for numel, seed in test_cases:
            test_one_reduce(wgk, op_name, numel, seed, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
