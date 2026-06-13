import argparse
import os
import subprocess
import tempfile

import torch
import torch.nn.functional as F


def write_tensor(path: str, tensor: torch.Tensor) -> None:
    tensor.detach().cpu().contiguous().numpy().astype("float32").tofile(path)


def read_tensor(path: str, shape, device: str = "cuda") -> torch.Tensor:
    import numpy as np

    data = np.fromfile(path, dtype="float32")
    return torch.from_numpy(data.copy()).reshape(shape).to(device)


def output_shape(case: dict) -> tuple:
    h_out = (case["h_in"] + 2 * case.get("pad_h", 0) -
             case.get("dilation_h", 1) * (case["k_h"] - 1) - 1) // case.get("stride_h", 1) + 1
    w_out = (case["w_in"] + 2 * case.get("pad_w", 0) -
             case.get("dilation_w", 1) * (case["k_w"] - 1) - 1) // case.get("stride_w", 1) + 1
    return (case["n"], case["c_out"], h_out, w_out)


def make_case_tensors(case: dict) -> tuple:
    generator = torch.Generator().manual_seed(case["seed"])
    input_tensor = torch.randn(
        case["n"], case["c_in"], case["h_in"], case["w_in"], generator=generator)
    weight_tensor = torch.randn(
        case["c_out"], case["c_in"] // case.get("groups", 1),
        case["k_h"], case["k_w"], generator=generator)
    use_bias = case.get("bias", True)
    bias_tensor = torch.randn(case["c_out"], generator=generator) if use_bias else None
    return input_tensor, weight_tensor, bias_tensor


def conv2d_flop_count(case: dict, out_shape: tuple) -> int:
    """Count multiply-adds as 2 FLOPs (mul + add) per output element."""
    out_numel = 1
    for dim in out_shape:
        out_numel *= dim
    per_out = (case["c_in"] // case.get("groups", 1)) * case["k_h"] * case["k_w"]
    return out_numel * per_out * 2


def bench_torch(case: dict, warmup: int, repeat: int) -> tuple:
    input_tensor, weight_tensor, bias_tensor = make_case_tensors(case)
    stride = (case.get("stride_h", 1), case.get("stride_w", 1))
    padding = (case.get("pad_h", 0), case.get("pad_w", 0))
    dilation = (case.get("dilation_h", 1), case.get("dilation_w", 1))
    groups = case.get("groups", 1)

    def fn() -> torch.Tensor:
        return F.conv2d(
            input_tensor, weight_tensor, bias=bias_tensor,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
        )

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
    latency_ms = start.elapsed_time(end) / repeat
    return fn(), latency_ms


def bench_wgkernel(
    tool_path: str, workdir: str, case: dict, algo: str, warmup: int, repeat: int
) -> tuple:
    input_tensor, weight_tensor, bias_tensor = make_case_tensors(case)
    out_shape = output_shape(case)

    input_path = os.path.join(workdir, "input.bin")
    weight_path = os.path.join(workdir, "weight.bin")
    bias_path = os.path.join(workdir, "bias.bin")
    output_path = os.path.join(workdir, "output.bin")

    write_tensor(input_path, input_tensor)
    write_tensor(weight_path, weight_tensor)
    if bias_tensor is not None:
        write_tensor(bias_path, bias_tensor)

    command = [
        tool_path,
        "--algo", algo,
        "--n", str(case["n"]),
        "--c_in", str(case["c_in"]),
        "--h_in", str(case["h_in"]),
        "--w_in", str(case["w_in"]),
        "--c_out", str(case["c_out"]),
        "--k_h", str(case["k_h"]),
        "--k_w", str(case["k_w"]),
        "--stride_h", str(case.get("stride_h", 1)),
        "--stride_w", str(case.get("stride_w", 1)),
        "--pad_h", str(case.get("pad_h", 0)),
        "--pad_w", str(case.get("pad_w", 0)),
        "--dilation_h", str(case.get("dilation_h", 1)),
        "--dilation_w", str(case.get("dilation_w", 1)),
        "--groups", str(case.get("groups", 1)),
        "--input", input_path,
        "--weight", weight_path,
        "--output", output_path,
    ]
    if bias_tensor is not None:
        command += ["--bias", bias_path]

    def fn() -> torch.Tensor:
        subprocess.run(command, capture_output=True, check=True, text=True)
        return read_tensor(output_path, out_shape, device="cuda")

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
    latency_ms = start.elapsed_time(end) / repeat
    return fn(), latency_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True,
                        help="Path to the wgkernel_conv2d_tool binary built by CMake.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--algos", nargs="*",
                        default=["naive", "im2col_gemm", "direct_tiled", "implicit_gemm"])
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the conv2d benchmark")

    cases = [
        # Toy shapes — these match the correctness suite and exercise the
        # full algorithm matrix at small size.
        {"name": "plain_3x3", "seed": 1, "n": 2, "c_in": 3, "h_in": 16, "w_in": 16,
         "c_out": 8, "k_h": 3, "k_w": 3, "pad_h": 1, "pad_w": 1},
        {"name": "pointwise_1x1", "seed": 2, "n": 1, "c_in": 8, "h_in": 10, "w_in": 10,
         "c_out": 16, "k_h": 1, "k_w": 1},
        {"name": "strided_3x3", "seed": 3, "n": 2, "c_in": 4, "h_in": 17, "w_in": 19,
         "c_out": 6, "k_h": 3, "k_w": 3, "stride_h": 2, "stride_w": 2,
         "pad_h": 1, "pad_w": 1},
        # YOLOX-flavoured shapes — closer to the workloads the operator was
        # written for, where algorithmic differences become visible.
        {"name": "yolox_backbone_64", "seed": 11, "n": 1, "c_in": 64, "h_in": 64,
         "w_in": 64, "c_out": 64, "k_h": 3, "k_w": 3, "pad_h": 1, "pad_w": 1},
        {"name": "yolox_backbone_128", "seed": 12, "n": 1, "c_in": 128,
         "h_in": 32, "w_in": 32, "c_out": 128, "k_h": 3, "k_w": 3,
         "pad_h": 1, "pad_w": 1},
    ]

    with tempfile.TemporaryDirectory() as workdir:
        for case in cases:
            print(f"=== case={case['name']} "
                  f"shape=(N={case['n']},C_in={case['c_in']},H={case['h_in']},W={case['w_in']}) "
                  f"-> C_out={case['c_out']} k=({case['k_h']}x{case['k_w']}) "
                  f"stride=({case.get('stride_h', 1)},{case.get('stride_w', 1)}) "
                  f"pad=({case.get('pad_h', 0)},{case.get('pad_w', 0)}) ===")
            torch_out, torch_ms = bench_torch(case, args.warmup, args.repeat)
            out_shape = output_shape(case)
            flops = conv2d_flop_count(case, out_shape)
            torch_tflops = flops / (torch_ms * 1.0e9)
            print(
                f"  torch       latency_ms={torch_ms:.4f} tflops={torch_tflops:.4f}"
            )

            for algo in args.algos:
                try:
                    wgk_out, wgk_ms = bench_wgkernel(
                        args.tool, workdir, case, algo, args.warmup, args.repeat)
                except subprocess.CalledProcessError as exc:
                    print(f"  {algo:<14} FAILED ({exc.stderr.strip() or 'tool error'})")
                    continue

                print(f"  DEBUG: wgk_out.device={wgk_out.device}, torch_out.device={torch_out.device}", flush=True)
                wgk_out = wgk_out.to(torch_out.device) if wgk_out.device != torch_out.device else wgk_out
                max_diff = (wgk_out - torch_out).abs().max().item()
                ok = torch.allclose(wgk_out, torch_out, atol=args.atol, rtol=args.rtol)
                speedup = torch_ms / wgk_ms if wgk_ms > 0 else float("inf")
                wgk_tflops = flops / (wgk_ms * 1.0e9)
                status = "ok" if ok else f"MISMATCH (max_diff={max_diff:.4e})"
                print(
                    f"  {algo:<14} latency_ms={wgk_ms:.4f} tflops={wgk_tflops:.4f} "
                    f"speedup_vs_torch={speedup:.3f}x correctness={status}"
                )


if __name__ == "__main__":
    main()
