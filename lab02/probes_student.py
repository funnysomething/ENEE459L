from __future__ import annotations

import json
import re
from typing import Any

from env import Env, ModuleNotAvailable, getattr_path, major_minor, read_text, unknown

# The NVIDIA-built PyTorch wheels for Jetson carry a local version segment —
# the part after "+" — that names the NVIDIA container release. A wheel from
# plain PyPI has no such segment. This is a hint, not a proof, which is why the
# probe reports the tag itself alongside the interpretation.
_NV_LOCAL_TAG = re.compile(r"(?:^|\.)nv\d", re.IGNORECASE)

# `# R36 (release), REVISION: 5.0, GCID: ...`
_L4T_RELEASE = re.compile(r"R(\d+)\s*\(release\)", re.IGNORECASE)
_L4T_REVISION = re.compile(
    r"REVISION:\s*(\d+(?:\.\d+)*)(?![\d.])", re.IGNORECASE
)

# Helper function.
def _split_local_version(raw: str) -> dict[str, Any]:
    if not raw:
        return {"raw": raw, "public": None, "local": None, "nvidia_build": False}
    public, sep, local = raw.partition("+")
    local = local if sep else None
    return {
        "raw": raw,
        "public": public or None,
        "local": local,
        "nvidia_build": bool(local and _NV_LOCAL_TAG.search(local)),
    }


# ---------------------------------------------------------------------------
# The probes.
# ---------------------------------------------------------------------------

def probe_torch(env: Env) -> dict[str, Any]:
    source = "import torch"
    try:
        torch = env.importer("torch")
    except ModuleNotAvailable as exc:
        return unknown(source, f"torch is not importable: {exc}")

    raw_version = getattr(torch, "__version__", None)
    if not raw_version:
        return unknown(source, "torch.__version__ is unavailable")

    version = _split_local_version(str(raw_version))
    cuda_version = getattr_path(torch, "version.cuda")
    is_available = getattr_path(torch, "cuda.is_available")
    if not callable(is_available):
        return unknown(source, "torch.cuda.is_available is unavailable")

    try:
        cuda_available = bool(is_available())
    except Exception as exc:
        return unknown(source, f"torch.cuda.is_available failed: {exc}")

    device_name = None
    if cuda_available:
        get_device_name = getattr_path(torch, "cuda.get_device_name")
        if callable(get_device_name):
            try:
                device_name = str(get_device_name(0))
            except Exception:
                # CUDA availability is still a useful, confirmed result even if
                # querying the device's friendly name fails.
                device_name = None

    if cuda_available:
        diagnosis = "torch is installed and sees the GPU"
    elif version["nvidia_build"]:
        diagnosis = (
            "torch is an NVIDIA build but cannot see the GPU — check the "
            "driver, CUDA runtime, and device access"
        )
    else:
        diagnosis = (
            "torch cannot see the GPU and has no NVIDIA local version tag — "
            "this may be a stock PyPI wheel"
        )

    return {
        "value": str(raw_version),
        "source": source,
        "status": "ok",
        "version": version,
        "cuda_available": cuda_available,
        "cuda_version": str(cuda_version) if cuda_version is not None else None,
        "device_name": device_name,
        "diagnosis": diagnosis,
    }


def probe_cuda(env: Env) -> dict[str, Any]:
    source = "/usr/local/cuda/version.json"
    raw = read_text(env.root, source)
    if raw is None:
        return unknown(source, "CUDA version manifest is missing or unreadable")

    try:
        manifest = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        return unknown(source, f"CUDA version manifest is not valid JSON: {exc}")

    cuda_manifest = manifest.get("cuda") if isinstance(manifest, dict) else None
    version = cuda_manifest.get("version") if isinstance(cuda_manifest, dict) else None
    if not isinstance(version, (str, int, float)) or not str(version).strip():
        return unknown(source, "CUDA version manifest has no cuda.version value")

    version = str(version).strip()
    line = major_minor(version)
    if line is None:
        return unknown(source, f"CUDA version is not a major-minor release: {version!r}")

    return {"value": version, "source": source, "status": "ok", "line": line}


def probe_opencv(env: Env) -> dict[str, Any]:
    source = "import cv2"
    try:
        cv2 = env.importer("cv2")
    except ModuleNotAvailable as exc:
        return unknown(source, f"OpenCV is not importable: {exc}")

    raw_version = getattr(cv2, "__version__", None)
    if not raw_version:
        return unknown(source, "cv2.__version__ is unavailable")

    get_device_count = getattr_path(cv2, "cuda.getCudaEnabledDeviceCount")
    if not callable(get_device_count):
        return {
            "value": str(raw_version),
            "source": source,
            "status": "ok",
            "cuda_devices": 0,
            "cuda_enabled": False,
            "detail": "the cv2.cuda device-count API is absent — this is a non-CUDA build",
        }

    try:
        cuda_devices = int(get_device_count())
    except Exception as exc:
        return unknown(source, f"OpenCV CUDA device query failed: {exc}")

    cuda_enabled = cuda_devices > 0
    if cuda_enabled:
        detail = f"OpenCV reports {cuda_devices} CUDA-enabled device(s)"
    else:
        detail = (
            "the cv2.cuda namespace exists but reports no devices — "
            "this is a non-CUDA build"
        )

    return {
        "value": str(raw_version),
        "source": source,
        "status": "ok",
        "cuda_devices": cuda_devices,
        "cuda_enabled": cuda_enabled,
        "detail": detail,
    }


def probe_tensorrt(env: Env) -> dict[str, Any]:
    source = "import tensorrt"
    try:
        tensorrt = env.importer("tensorrt")
    except ModuleNotAvailable as exc:
        if env.python.prefix != env.python.base_prefix:
            detail = (
                "TensorRT is not importable from this virtual environment; "
                "it may have been created without --system-site-packages"
            )
        else:
            detail = f"TensorRT is not importable: {exc}"
        return unknown(source, detail)

    raw_version = getattr(tensorrt, "__version__", None)
    if not raw_version:
        return unknown(source, "tensorrt.__version__ is unavailable")

    version = str(raw_version)
    line = major_minor(version)
    if line is None:
        return unknown(source, f"TensorRT version is not a major-minor release: {version!r}")

    return {"value": version, "source": source, "status": "ok", "line": line}


def probe_l4t(env: Env) -> dict[str, Any]:
    source = "/etc/nv_tegra_release"
    raw = read_text(env.root, source)
    if raw is None:
        return unknown(source, "L4T release file is missing or unreadable")

    release_match = _L4T_RELEASE.search(raw)
    revision_match = _L4T_REVISION.search(raw)
    if release_match is None or revision_match is None:
        return unknown(source, "L4T release file does not contain a release and revision")

    value = f"{release_match.group(1)}.{revision_match.group(1)}"
    line = major_minor(value)
    if line is None:
        return unknown(source, f"L4T version is not a major-minor release: {value!r}")

    return {
        "value": value,
        "source": source,
        "status": "ok",
        "line": line,
        "raw": raw,
    }

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Env.real()
    # out = probe_l4t(env)
#     print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Env.real()

    # testing probes
    report = {
        "probe_torch": probe_torch(env),
        "probe_cuda": probe_cuda(env),
        "probe_opencv": probe_opencv(env),
        "probe_tensorrt": probe_tensorrt(env),
        "probe_l4t": probe_l4t(env),
    }
    
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
