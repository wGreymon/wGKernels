import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, TEST_ROOT)

from test_utils import add_device_argument, load_wgkernel, torch_device


OP_CASES = {
    "self_attention": {
        "wgkernel": "self_attention",
    },
}


def selected_ops(op_name: str) -> list[str]:
    if op_name == "all":
        return list(OP_CASES.keys())
    return [op_name]


def make_tensors(case: dict, device_name: str):
    device = torch_device(device_name)
    generator = torch.Generator(device=device).manual_seed(case["seed"])
    q = torch.randn(case["batch"], case["q_len"], case["head_dim"], device=device, generator=generator)
    k = torch.randn(case["batch"], case["kv_len"], case["head_dim"], device=device, generator=generator)
    v = torch.randn(case["batch"], case["kv_len"], case["value_dim"], device=device, generator=generator)
    return q, k, v


def input_for_wgkernel(tensor: torch.Tensor, device_name: str):
    if device_name == "cpu":
        return tensor.cpu().numpy()
    return tensor


def torch_self_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    q4 = q.unsqueeze(1)
    k4 = k.unsqueeze(1)
    v4 = v.unsqueeze(1)
    return F.scaled_dot_product_attention(q4, k4, v4, is_causal=causal).squeeze(1)


def assert_allclose(name: str, actual, expected: torch.Tensor, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    actual_tensor = torch.as_tensor(actual, dtype=torch.float32, device=expected.device).reshape(expected.shape)
    if not torch.allclose(actual_tensor, expected, atol=atol, rtol=rtol):
        max_diff = (actual_tensor - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch: max abs diff={max_diff}")


def test_self_attention(wgk, case: dict, device_name: str) -> None:
    print(
        f"   op self_attention case {case['name']} "
        f"b={case['batch']} q={case['q_len']} kv={case['kv_len']} d={case['head_dim']} vd={case['value_dim']} causal={case['causal']}"
    )
    q, k, v = make_tensors(case, device_name)
    expected = torch_self_attention(q, k, v, case["causal"])
    actual = wgk.self_attention(
        input_for_wgkernel(q, device_name),
        input_for_wgkernel(k, device_name),
        input_for_wgkernel(v, device_name),
        case["causal"],
    )
    assert_allclose("self_attention", actual, expected)


def test_one_attention(wgk, op_name: str, case: dict, device_name: str) -> None:
    if op_name == "self_attention":
        test_self_attention(wgk, case, device_name)
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
        {"name": "tiny", "seed": 11, "batch": 1, "q_len": 1, "kv_len": 1, "head_dim": 4, "value_dim": 4, "causal": False},
        {"name": "square", "seed": 23, "batch": 2, "q_len": 4, "kv_len": 4, "head_dim": 8, "value_dim": 8, "causal": False},
        {"name": "causal", "seed": 31, "batch": 1, "q_len": 5, "kv_len": 5, "head_dim": 8, "value_dim": 8, "causal": True},
        {"name": "cross_attention", "seed": 47, "batch": 2, "q_len": 3, "kv_len": 6, "head_dim": 8, "value_dim": 5, "causal": False},
        {"name": "throughput_like", "seed": 59, "batch": 2, "q_len": 64, "kv_len": 64, "head_dim": 64, "value_dim": 64, "causal": False},
    ]

    print(f"Testing attention {args.op_name} on {args.device}")
    for op_name in selected_ops(args.op_name):
        for case in test_cases:
            test_one_attention(wgk, op_name, case, args.device)

    print("\033[92mTest passed!\033[0m\n")
