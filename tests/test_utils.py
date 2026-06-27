import sys
import time
from importlib import import_module
from pathlib import Path

import torch


DEVICE_CHOICES = ("cpu", "simd", "cuda", "cutlass", "metax")


def add_device_argument(parser, default: str = "cuda") -> None:
    parser.add_argument(
        "--device",
        default=default,
        choices=DEVICE_CHOICES,
        type=str,
        help="operator backend/device to run",
    )


def require_device_available(device_name: str) -> str:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for device 'cuda'")
    return device_name


def torch_device(device_name: str, device_id: int = 0) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device(f"cuda:{device_id}")
    if device_name == "metax":
        return torch.device(f"cuda:{device_id}")
    raise RuntimeError(f"torch device mapping for device '{device_name}' is not available yet")


def load_wgkernel(module_dir: str | None, device_name: str = "cuda"):
    device_name = require_device_available(device_name)

    if module_dir:
        sys.path.insert(0, module_dir)
    else:
        repo_root = Path(__file__).resolve().parents[2]
        default_module_dir = repo_root / "build" / "python"
        if default_module_dir.exists():
            sys.path.insert(0, str(default_module_dir))

    module_name = f"wgkernel_{device_name}"
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise RuntimeError(
                f"Python module '{module_name}' is not built or not found. "
                f"Build the {device_name} pybind module first, or pass --module-dir."
            ) from exc
        raise


def sync_device(device_name: str = "cuda") -> None:
    if device_name in {"cuda", "metax"} and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(
    torch_func,
    wgkernel_func,
    device_name: str = "cuda",
    warmup: int = 10,
    repeat: int = 100,
) -> tuple[float, float]:
    for _ in range(warmup):
        torch_func()
    sync_device(device_name)
    start = time.perf_counter()
    for _ in range(repeat):
        torch_func()
    sync_device(device_name)
    torch_ms = (time.perf_counter() - start) * 1000.0 / repeat

    for _ in range(warmup):
        wgkernel_func()
    sync_device(device_name)
    start = time.perf_counter()
    for _ in range(repeat):
        wgkernel_func()
    sync_device(device_name)
    wgkernel_ms = (time.perf_counter() - start) * 1000.0 / repeat

    return torch_ms, wgkernel_ms
