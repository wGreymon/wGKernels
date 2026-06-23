import argparse
import math
import os
import sys

import torch

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, TEST_ROOT)

from test_utils import benchmark, load_wgkernel


def make_input(numel: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.empty(numel, device="cuda", dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator)


def assert_close(actual: float, expected: float, atol: float, rtol: float) -> None:
    if math.fabs(actual - expected) > atol + rtol * math.fabs(expected):
        raise AssertionError(f"actual={actual}, expected={expected}")


def test_op_reduce(wgk, numel: int, seed: int, profile: bool = False) -> None:
    print(f"   numel {numel} seed {seed}")
    values = make_input(numel, seed)

    actual_sum = wgk.reduce_sum_torch(values)
    expected_sum = float(values.sum().item())
    assert_close(actual_sum, expected_sum, atol=1e-2, rtol=1e-4)

    actual_max = wgk.reduce_max_torch(values)
    expected_max = float(values.max().item())
    assert_close(actual_max, expected_max, atol=1e-6, rtol=1e-6)

    actual_argmax = wgk.reduce_argmax_torch(values)
    expected_argmax = int(values.argmax().item())
    if actual_argmax != expected_argmax:
        raise AssertionError(f"argmax mismatch: actual={actual_argmax}, expected={expected_argmax}")

    if profile:
        for name, torch_func, wgkernel_func in (
            ("sum", lambda: values.sum().item(), lambda: wgk.reduce_sum_torch(values)),
            ("max", lambda: values.max().item(), lambda: wgk.reduce_max_torch(values)),
            ("argmax", lambda: values.argmax().item(), lambda: wgk.reduce_argmax_torch(values)),
        ):
            torch_ms, wgkernel_ms = benchmark(torch_func, wgkernel_func)
            speedup = torch_ms / wgkernel_ms if wgkernel_ms > 0 else float("inf")
            print(f"      {name:<6} torch={torch_ms:.4f} ms wgkernel={wgkernel_ms:.4f} ms speedup={speedup:.3f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir", default=None)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the PyTorch reduce test")

    wgk = load_wgkernel(args.module_dir)
    test_cases = [(1, 11), (7, 23), (257, 31), (4099, 47), (65536, 89), (1048576, 113)]

    print("Testing reduce on cuda")
    for numel, seed in test_cases:
        test_op_reduce(wgk, numel, seed, args.profile)

    print("\033[92mTest passed!\033[0m\n")
