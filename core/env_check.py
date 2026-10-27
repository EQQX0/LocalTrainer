"""Startup check of torch, CUDA and RTX 50-series (Blackwell, sm_120) support."""
from __future__ import annotations

import platform
import sys
from typing import Any

FIX_COMMAND = "pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128"
FIX_COMMAND_VENV = ".venv\\Scripts\\python.exe -m " + FIX_COMMAND


def arch_supported(capability: tuple[int, int], arch_list: list[str]) -> bool:
    """True if the GPU's sm_XY is compiled into this torch build (Blackwell RTX 50 needs sm_120)."""
    major, minor = capability
    return f"sm_{major}{minor}" in arch_list


def probe() -> dict[str, Any]:
    """Return a JSON-safe description of the compute environment."""
    info: dict[str, Any] = {
        "python": platform.python_version(), "executable": sys.executable,
        "torch": None, "cuda_build": None, "cuda_available": False, "device_name": None,
        "capability": None, "arch_list": [], "arch_supported": False, "vram_total_mb": None,
        "ultralytics": None, "problem": None, "fix": None,
    }
    try:
        import torch
    except Exception as exc:
        info["problem"] = f"PyTorch is not installed or failed to load ({exc})."
        info["fix"] = FIX_COMMAND_VENV
        return info
    info["torch"] = torch.__version__
    info["cuda_build"] = torch.version.cuda
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except Exception as exc:
        info["problem"] = f"Ultralytics failed to load ({exc})."
        info["fix"] = ".venv\\Scripts\\python.exe -m pip install -r requirements.txt"
    if torch.version.cuda is None:
        info["problem"] = info["problem"] or "This PyTorch build has no CUDA support (CPU only)."
        info["fix"] = info["fix"] or FIX_COMMAND_VENV
        return info
    try:
        available = torch.cuda.is_available()
    except Exception:
        available = False
    info["cuda_available"] = available
    if not available:
        info["problem"] = info["problem"] or "CUDA is not available. Check the NVIDIA driver, then reinstall PyTorch."
        info["fix"] = info["fix"] or FIX_COMMAND_VENV
        return info
    cap = torch.cuda.get_device_capability(0)
    arch_list = list(torch.cuda.get_arch_list())
    info.update({
        "device_name": torch.cuda.get_device_name(0), "capability": f"{cap[0]}.{cap[1]}", "arch_list": arch_list,
        "vram_total_mb": int(torch.cuda.get_device_properties(0).total_memory / 1024**2),
        "arch_supported": arch_supported(cap, arch_list),
    })
    if not info["arch_supported"]:
        info["problem"] = (f"This PyTorch build does not support your GPU (sm_{cap[0]}{cap[1]}). "
                           "RTX 50-series cards need the CUDA 12.8 build.")
        info["fix"] = FIX_COMMAND_VENV
        return info
    try:  # a tiny kernel proves the GPU really runs
        x = torch.ones(8, device="cuda")
        float((x * 2).sum().item())
    except Exception as exc:
        info["problem"] = f"CUDA is available but a test kernel failed ({exc})."
        info["fix"] = FIX_COMMAND_VENV
    return info
