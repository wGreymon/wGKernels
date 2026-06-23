import sys
import time
from pathlib import Path

import torch


def load_wgkernel(module_dir: str | None):
    if module_dir:
        sys.path.insert(0, module_dir)
    else:
        repo_root = Path(__file__).resolve().parents[2]
        default_module_dir = repo_root / "build" / "python"
        if default_module_dir.exists():
            sys.path.insert(0, str(default_module_dir))

    import wgkernel_cuda

    return wgkernel_cuda


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(torch_func, wgkernel_func, warmup: int = 10, repeat: int = 100) -> tuple[float, float]:
    for _ in range(warmup):
        torch_func()
    sync_cuda()
    start = time.perf_counter()
    for _ in range(repeat):
        torch_func()
    sync_cuda()
    torch_ms = (time.perf_counter() - start) * 1000.0 / repeat

    for _ in range(warmup):
        wgkernel_func()
    sync_cuda()
    start = time.perf_counter()
    for _ in range(repeat):
        wgkernel_func()
    sync_cuda()
    wgkernel_ms = (time.perf_counter() - start) * 1000.0 / repeat

    return torch_ms, wgkernel_ms
