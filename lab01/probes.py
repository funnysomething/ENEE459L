"""Probes — read what the machine says about itself.

INSTRUCTOR SOLUTION. Do not distribute. The student copy of this file has the
body of every function below replaced by `raise NotImplementedError`.

Every probe takes a `root` argument and reads nothing outside it. That is not
decoration: it is what makes this lab gradeable without twenty boards on a
desk, and it is the reason the test suite can present a fake SD-booted machine
and check that the student's code notices. Code that hardcodes "/" cannot be
tested, and a measurement you cannot test is a measurement you cannot trust —
which is the whole argument of Lecture 01, applied to the student's own code.

Each probe returns a dict with, at minimum, a `value` and a `source` key. The
`source` is the path or command the value came from. A number without its
provenance is not evidence, so the report format refuses to carry one.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any
import pdb
import json

# ---------------------------------------------------------------------------
# Small helpers. These are given to students; the exercise is the probes.
# ---------------------------------------------------------------------------


def read_text(root: Path, rel: str) -> str | None:
    """Read `root/rel`, returning None if it is missing or unreadable.

    Missing is a normal outcome here, not an error: a devkit with no NVMe
    genuinely has no /sys/block/nvme0n1, and the report needs to say so rather
    than crash.
    """
    p = Path(root) / rel.lstrip("/")
    try:
        return p.read_text(errors="replace").strip("\x00").strip()
    except (OSError, UnicodeDecodeError):
        return None


def run(cmd: list[str]) -> str | None:
    """Run a command, returning stdout, or None if it is absent or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def unknown(source: str, why: str) -> dict[str, Any]:
    """The value this lab returns when it cannot determine something.

    Note what this is not: it is not None threaded through the report, and it
    is not a plausible default. It is an explicit record that the probe ran and
    failed, carrying the reason. Assignment 1's rubric gives credit for these.
    """
    return {"value": None, "source": source, "status": "unknown", "detail": why}

# LnkSta/LnkCap lines look like:
#   LnkSta: Speed 8GT/s, Width x4, TrErr- Train- SlotClk+ DLActive- ...
#   LnkCap: Port #0, Speed 16GT/s, Width x4, ASPM L1, Exit Latency L1 <64us
_SPEED_RE = re.compile(r"Speed\s+([\d.]+)GT/s")
_WIDTH_RE = re.compile(r"Width\s+x(\d+)")

# PCIe generation by per-lane transfer rate. Gen3 is 8 GT/s; the Orin Nano
# devkit's M.2 Key-M slot is wired Gen3 x4, so a Gen4 drive reporting 16 GT/s
# capability and 8 GT/s status is behaving correctly, not underperforming.
#
# Keyed by float, not by the string lspci printed. Keying by string means
# deciding whether "8", "8.0" and "08" are the same rate, and the obvious
# normalisation — stripping trailing zeros and dots — silently turns 20 into 2.
_GEN_BY_GTS = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}


def _parse_link_line(line: str) -> dict[str, Any]:
    speed = _SPEED_RE.search(line)
    width = _WIDTH_RE.search(line)
    gts = float(speed.group(1)) if speed else None
    return {
        "raw": line.strip(),
        "gts": gts,
        "width": int(width.group(1)) if width else None,
        "gen": _GEN_BY_GTS.get(gts) if gts is not None else None,
    }

def generate_interpretation_string(negotiated, capability):
    neg_speed = negotiated['gts']
    cap_speed = capability['gts']
    if cap_speed > neg_speed:
        interpretation = (
            f"drive capable of Gen{capability['gen']}, link running at "
            f"Gen{negotiated['gen']} — expected on this carrier board, "
            "whose M.2 Key-M slot is wired Gen3 x4"
        )
    else:
        interpretation = (
            f"link running at its full capability, Gen{negotiated['gen']} "
            f"x{negotiated['width']}"
        )
    return interpretation
    


# ---------------------------------------------------------------------------
# The probes.
# ---------------------------------------------------------------------------

## An example code.
def probe_module_model(root: Path = Path("/")) -> dict[str, Any]:
    """Which board is this?

    The device tree model string is the most trustworthy identity on a Jetson —
    it comes from the hardware description the bootloader handed the kernel,
    not from anything installed afterwards.
    """

    # step 1: Read the raw null-terminated text from /proc/device-tree/model
    src = "/proc/device-tree/model"
    raw = read_text(root, src)

    # if unable to read, return an empty dictionary by calling unknown().
    if not raw:
        return unknown(src, "device tree model node absent — not a Jetson, or /proc not mounted")
    
    # step 2: strip null bytes and whitespace from raw string
    raw = raw.rstrip("\x00").strip()
    return {"value": raw, "source": src, "status": "ok"}

# todo by students
def probe_memory_total_kb(root: Path = Path("/")) -> dict[str, Any]:
    src = "/proc/meminfo"
    raw = read_text(root, src)
    if not raw:
        return unknown(src, "meminfo missing")
    m = re.search(r"MemTotal:\s+(\d+)", raw)
    if not m:
        return unknown(src, "MemTotal line not found")
    return {"value": int(m.group(1)), "source": src, "status": "ok"}


def probe_root_source(root: Path = Path("/")) -> dict[str, Any]:
    src = "/proc/mounts"
    raw = read_text(root, src)
    if not raw:
        return unknown(src, "no root mount entry found in mount table")
    for line in raw.splitlines():
        if line.startswith("/"):
            parts = line.split()
            device = parts[0]
            kind = "nvme" if "nvme" in device else ("mmcblk" if "mmcblk" in device else "other")
            return {"value": device, "kind": kind, "source": src, "status": "ok"}
    return unknown(src, "no root mount entry found in mount table")


def probe_nvme_present(root: Path = Path("/")) -> dict[str, Any]:
    src = "/sys/block/nvme0n1"
    nvme_path = Path(root) / "sys/block/nvme0n1"
    model_path = nvme_path / "model"
    model = None
    try:
        if model_path.exists():
            model = model_path.read_text(errors="replace").strip()
    except (OSError, UnicodeDecodeError):
        pass
    if nvme_path.exists() or nvme_path.is_symlink():
        return {"value": True, "model": model, "source": src, "status": "ok"}
    return {"value": False, "model": None, "source": src, "status": "ok"}


def probe_pcie_link(root: Path = Path("/"), lspci_output: str | None = None) -> dict[str, Any]:
    src = "lspci -vv"
    if lspci_output is not None:
        output = lspci_output
    else:
        output = run(["lspci", "-vv"])
    if output is None:
        return unknown(src, "lspci not available or failed")
    lnksta_line = None
    lnkcap_line = None
    for line in output.splitlines():
        if "LnkSta:" in line:
            lnksta_line = line
        elif "LnkCap:" in line:
            lnkcap_line = line
    if lnksta_line is None or lnkcap_line is None:
        return unknown(src, "LnkSta or LnkCap line missing")
    negotiated = _parse_link_line(lnksta_line)
    capability = _parse_link_line(lnkcap_line)
    interpretation = generate_interpretation_string(negotiated, capability)
    return {
        "value": lnksta_line.strip(),
        "negotiated": negotiated,
        "capability": capability,
        "interpretation": interpretation,
        "source": src,
        "status": "ok",
    }


def probe_thermal_zones(root: Path = Path("/")) -> dict[str, Any]:
    src = "/sys/class/thermal/thermal_zone*/temp"
    zones = []
    zone_dir = Path(root) / "sys/class/thermal"
    if not zone_dir.exists():
        return unknown(src, "thermal zone directory missing")
    max_temp = 0.0
    for zone_path in zone_dir.glob("thermal_zone*"):
        temp_path = zone_path / "temp"
        type_path = zone_path / "type"
        if not temp_path.exists():
            continue
        try:
            with open(str(temp_path), "rb") as f:
                raw_bytes = f.read()
            if raw_bytes is None:
                continue
            content = raw_bytes.decode("utf-8", errors="replace")
            temp_raw = content.strip()
            temp_c = float(temp_raw) / 1000.0
        except (ValueError, OSError):
            continue
        zone_name = zone_path.name
        zone_type = None
        if type_path.exists():
            try:
                zone_type = type_path.read_text(errors="replace").strip()
            except (OSError, UnicodeDecodeError):
                pass
        zones.append({"zone": zone_name, "type": zone_type, "temp_c": temp_c})
        if temp_c > max_temp:
            max_temp = temp_c
    if not zones:
        return unknown(src, "no thermal zones found")
    return {
        "value": max_temp,
        "zones": zones,
        "source": src,
        "status": "ok",
    }


def probe_power_mode(root: Path = Path("/"), nvpmodel_output: str | None = None) -> dict[str, Any]:
    src = "nvpmodel -q"
    if nvpmodel_output is not None:
        output = nvpmodel_output
    else:
        output = run(["nvpmodel", "-q"])
    if output is None:
        return unknown(src, "nvpmodel not available or failed")
    # Parse output like "NV Power Mode: MAXN"
    mode_name = None
    mode_id = None
    for line in output.splitlines():
        if "NV Power Mode:" in line:
            mode_name = line.split(":")[-1].strip()
            # Map common names to IDs
            if "MAXN" in mode_name:
                mode_id = 0
            elif "25W" in mode_name or "15W" in mode_name:
                mode_id = 1 if "25W" in mode_name else 2
            else:
                mode_id = None
    if mode_name is None:
        return unknown(src, "power mode not found in output")
    return {
        "value": mode_name,
        "mode_id": mode_id,
        "source": src,
        "status": "ok",
    }

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
#     out = probe_power_mode()
#     print(out)

# for generating system_report.json
if __name__ == "__main__":
    report = {
        "module_model": probe_module_model(),
        "memory_total_kb": probe_memory_total_kb(),
        "root_source": probe_root_source(),
        "nvme_present": probe_nvme_present(),
        "pcie_link": probe_pcie_link(),
        "thermal_zones": probe_thermal_zones(),
        "power_mode": probe_power_mode(),
    }
    
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

