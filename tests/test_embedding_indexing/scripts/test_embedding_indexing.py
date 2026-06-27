import argparse
import os
import sys

import torch
import torch.nn.functional as F

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, TEST_ROOT)

from test_utils import add_device_argument, load_wgkernel, torch_device


OP_CASES = {
    "embedding": {
        "wgkernel": "embedding",
        "torch": F.embedding,
        "atol": 0.0,
        "rtol": 0.0,
    },
}


def selected_ops(op_name: str) -> list[str]:
    if op_name == "all":
        return list(OP_CASES.keys())
    return [op_name]


def make_inputs(case: dict, device_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch_device(device_name)
    generator = torch.Generator(device=device).manual_seed(case["seed"])
    weight = torch.randn(case["num_embeddings"], case["embedding_dim"], device=device, generator=generator)
    indices = torch.randint(
        0,
        case["num_embeddings"],
        case["index_shape"],
        device=device,
        generator=generator,
        dtype=torch.int64,
    )
    return weight, indices


def input_for_wgkernel(tensor: torch.Tensor, device_name: str):
    if device_name == "cpu":
        return tensor.cpu().numpy()
    return tensor


def assert_allclose(name: str, actual, expected: torch.Tensor, atol: float = 0.0, rtol: float = 0.0) -> None:
    actual_tensor = torch.as_tensor(actual, dtype=torch.float32, device=expected.device).reshape(expected.shape)
    if not torch.allclose(actual_tensor, expected, atol=atol, rtol=rtol):
        max_diff = (actual_tensor - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch: max abs diff={max_diff}")


def test_embedding(wgk, case: dict, device_name: str) -> None:
    print(
        f"   op embedding case {case['name']} "
        f"weight=({case['num_embeddings']}, {case['embedding_dim']}) indices={case['index_shape']}"
    )
    weight, indices = make_inputs(case, device_name)
    expected = F.embedding(indices, weight)
    actual = wgk.embedding(input_for_wgkernel(weight, device_name), input_for_wgkernel(indices, device_name))
    assert_allclose("embedding", actual, expected)


def test_one_embedding_indexing(wgk, op_name: str, case: dict, device_name: str) -> None:
    if op_name == "embedding":
        test_embedding(wgk, case, device_name)
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
        {"name": "single", "seed": 11, "num_embeddings": 8, "embedding_dim": 4, "index_shape": (1,)},
        {"name": "vector", "seed": 23, "num_embeddings": 16, "embedding_dim": 8, "index_shape": (7,)},
        {"name": "matrix", "seed": 31, "num_embeddings": 32, "embedding_dim": 16, "index_shape": (3, 5)},
        {"name": "batch_sequence", "seed": 47, "num_embeddings": 128, "embedding_dim": 32, "index_shape": (2, 4, 6)},
        {"name": "wide_dim", "seed": 59, "num_embeddings": 64, "embedding_dim": 257, "index_shape": (4, 3)},
        {"name": "throughput_like", "seed": 71, "num_embeddings": 65536, "embedding_dim": 128, "index_shape": (64, 128)},
    ]

    print(f"Testing embedding_indexing {args.op_name} on {args.device}")
    for op_name in selected_ops(args.op_name):
        for case in test_cases:
            test_one_embedding_indexing(wgk, op_name, case, args.device)

    print("\033[92mTest passed!\033[0m\n")
