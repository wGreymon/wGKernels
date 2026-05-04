import argparse

import torch


def make_input(numel: int) -> torch.Tensor:
    index = torch.arange(numel, device="cuda", dtype=torch.int64)
    periodic = (index * 17) % 97
    values = (periodic - 48).to(torch.float32) * 0.125
    values = values + (index % 13).to(torch.float32) * 0.01
    return values


def benchmark(op: str, values: torch.Tensor, warmup: int, repeat: int) -> float:
    if op == "sum":
        fn = lambda: values.sum()
    elif op == "max":
        fn = lambda: values.max()
    elif op == "argmax":
        fn = lambda: values.argmax()
    else:
        raise ValueError(f"Unsupported op: {op}")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", default="sum", choices=["sum", "max", "argmax"])
    parser.add_argument("--numel", type=int, default=1 << 24)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the PyTorch benchmark")

    values = make_input(args.numel)
    latency_ms = benchmark(args.op, values, args.warmup, args.repeat)
    bandwidth_gb_s = (args.numel * 4) / (latency_ms * 1.0e6)

    print(
        f"torch_benchmark op={args.op} numel={args.numel} warmup={args.warmup} "
        f"repeat={args.repeat} latency_ms={latency_ms:.4f} bandwidth_gb_s={bandwidth_gb_s:.4f}"
    )


if __name__ == "__main__":
    main()
