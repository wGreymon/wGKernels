import argparse
import math
import subprocess

import torch


def make_input(numel: int) -> torch.Tensor:
    index = torch.arange(numel, device="cuda", dtype=torch.int64)
    periodic = (index * 17) % 97
    values = (periodic - 48).to(torch.float32) * 0.125
    values = values + (index % 13).to(torch.float32) * 0.01
    return values


def run_tool(tool_path: str, op: str, numel: int):
    completed = subprocess.run(
        [tool_path, "--op", op, "--numel", str(numel)],
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

    cases = [1, 7, 257, 4099, 65536, 1048576]
    for numel in cases:
        values = make_input(numel)

        tool_sum = run_tool(args.tool, "sum", numel)
        torch_sum = float(values.sum().item())
        assert_close(tool_sum, torch_sum, atol=1e-2, rtol=1e-4)

        tool_max = run_tool(args.tool, "max", numel)
        torch_max = float(values.max().item())
        assert_close(tool_max, torch_max, atol=1e-6, rtol=1e-6)

        tool_argmax = run_tool(args.tool, "argmax", numel)
        torch_argmax = int(values.argmax().item())
        if tool_argmax != torch_argmax:
            raise AssertionError(f"argmax mismatch: actual={tool_argmax}, expected={torch_argmax}")

    print("PyTorch alignment check passed")


if __name__ == "__main__":
    main()
