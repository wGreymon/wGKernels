import argparse
import math
import subprocess

import torch


def make_input(numel: int, seed: int) -> torch.Tensor:
    """Mirror the deterministic generator inside reduce_tool.cu so the same
    Python seed reproduces the same device buffer the C++ tool was given.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.empty(numel, device="cuda", dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator)


def run_tool(tool_path: str, op: str, numel: int, seed: int):
    completed = subprocess.run(
        [tool_path, "--op", op, "--numel", str(numel), "--seed", str(seed)],
        capture_output=True,
        check=True,
        text=True,
    )
    output = completed.stdout.strip()
    if not output.startswith("result="):
        raise RuntimeError(f"Unexpected tool output: {output}")
    value = output.split("=", 1)[1]
    return int(value) if op == "argmax" else float(value)


def assert_close(actual: float, expected: float, atol: float, rtol: float):
    if math.fabs(actual - expected) > atol + rtol * math.fabs(expected):
        raise AssertionError(f"actual={actual}, expected={expected}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the PyTorch reduce test")

    cases = [(1, 11), (7, 23), (257, 31), (4099, 47), (65536, 89), (1048576, 113)]
    for numel, seed in cases:
        values = make_input(numel, seed)

        tool_sum = run_tool(args.tool, "sum", numel, seed)
        torch_sum = float(values.sum().item())
        assert_close(tool_sum, torch_sum, atol=1e-2, rtol=1e-4)

        tool_max = run_tool(args.tool, "max", numel, seed)
        torch_max = float(values.max().item())
        assert_close(tool_max, torch_max, atol=1e-6, rtol=1e-6)

        tool_argmax = run_tool(args.tool, "argmax", numel, seed)
        torch_argmax = int(values.argmax().item())
        if tool_argmax != torch_argmax:
            raise AssertionError(f"argmax mismatch: actual={tool_argmax}, expected={torch_argmax}")

    print("PyTorch alignment check passed")


if __name__ == "__main__":
    main()
