import argparse
import os
import sys

import torch
import torch.nn.functional as F

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, TEST_ROOT)

from test_utils import add_device_argument, load_wgkernel, torch_device


OP_CASES = {
    "maxpool2d_nchw": {
        "atol": 0.0,
        "rtol": 0.0,
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


def make_input(case: dict, device_name: str) -> torch.Tensor:
    device = torch_device(device_name)
    generator = torch.Generator(device=device).manual_seed(case["seed"])
    return torch.randn(case["n"], case["c"], case["h"], case["w"], device=device, generator=generator)


def test_maxpool2d_nchw(wgk, case: dict, device_name: str) -> None:
    print(
        f"   op maxpool2d_nchw case {case['name']} "
        f"shape=({case['n']}, {case['c']}, {case['h']}, {case['w']}) "
        f"kernel=({case['k_h']}, {case['k_w']}) stride=({case['stride_h']}, {case['stride_w']}) "
        f"pad=({case['pad_h']}, {case['pad_w']})"
    )
    values = make_input(case, device_name)
    expected = F.max_pool2d(
        values,
        kernel_size=(case["k_h"], case["k_w"]),
        stride=(case["stride_h"], case["stride_w"]),
        padding=(case["pad_h"], case["pad_w"]),
    )
    actual = wgk.maxpool2d_nchw(
        input_for_wgkernel(values, device_name),
        k_h=case["k_h"],
        k_w=case["k_w"],
        stride_h=case["stride_h"],
        stride_w=case["stride_w"],
        pad_h=case["pad_h"],
        pad_w=case["pad_w"],
    )
    op_case = OP_CASES["maxpool2d_nchw"]
    assert_allclose("maxpool2d_nchw", actual, expected, op_case["atol"], op_case["rtol"])


def test_one_pooling(wgk, op_name: str, device_name: str) -> None:
    if op_name == "maxpool2d_nchw":
        test_cases = [
            {"name": "tiny", "seed": 11, "n": 1, "c": 1, "h": 2, "w": 2, "k_h": 2, "k_w": 2, "stride_h": 1, "stride_w": 1, "pad_h": 0, "pad_w": 0},
            {"name": "plain_2x2", "seed": 23, "n": 2, "c": 3, "h": 8, "w": 8, "k_h": 2, "k_w": 2, "stride_h": 2, "stride_w": 2, "pad_h": 0, "pad_w": 0},
            {"name": "overlap_3x3", "seed": 31, "n": 1, "c": 4, "h": 9, "w": 11, "k_h": 3, "k_w": 3, "stride_h": 1, "stride_w": 1, "pad_h": 1, "pad_w": 1},
            {"name": "nonsquare", "seed": 47, "n": 2, "c": 2, "h": 10, "w": 13, "k_h": 2, "k_w": 3, "stride_h": 2, "stride_w": 1, "pad_h": 0, "pad_w": 1},
            {"name": "spp_like", "seed": 59, "n": 1, "c": 8, "h": 16, "w": 16, "k_h": 5, "k_w": 5, "stride_h": 1, "stride_w": 1, "pad_h": 2, "pad_w": 2},
            {"name": "throughput_like", "seed": 71, "n": 8, "c": 64, "h": 112, "w": 112, "k_h": 2, "k_w": 2, "stride_h": 2, "stride_w": 2, "pad_h": 0, "pad_w": 0},
        ]
        for case in test_cases:
            test_maxpool2d_nchw(wgk, case, device_name)
        return
    raise ValueError(f"unsupported op: {op_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir", default=None)
    add_device_argument(parser)
    parser.add_argument("--op-name", default="all", choices=["all", *OP_CASES.keys()])
    args = parser.parse_args()

    wgk = load_wgkernel(args.module_dir, args.device)

    print(f"Testing pooling {args.op_name} on {args.device}")
    for op_name in selected_ops(args.op_name):
        test_one_pooling(wgk, op_name, args.device)

    print("\033[92mTest passed!\033[0m\n")
