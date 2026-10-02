"""Check 4: is this Mac suitable for a local UI-TARS grounding model?

Reads chip, memory, free memory, disk and installed local-model runtimes.
Makes no model calls and downloads nothing. Model-size numbers below are
rough estimates (parameters x bits / 8, plus a fixed allowance) and stay
unverified until check 5 runs the model on this Mac.
"""

import json
import platform
import re
import shutil
import subprocess
from pathlib import Path

from common import FAIL, INFO, PASS, WARN, Report, require_macos

GIB = 1024**3
OS_AND_APPS_GIB = 8  # assumed headroom for macOS + your normal apps (estimate)

# UI-TARS-1.5-7B (Qwen2.5-VL based, ~8.3B params incl. vision encoder).
# est_gib = weights + ~1.5 GiB runtime/KV allowance. All estimates.
CANDIDATES = [
    ("UI-TARS-1.5-7B, 4-bit (e.g. MLX 4bit or GGUF Q4_K_M)", 6.5),
    ("UI-TARS-1.5-7B, 8-bit (MLX 8bit or GGUF Q8_0)", 10.5),
    ("UI-TARS-1.5-7B, 16-bit (BF16/FP16)", 18.0),
]


def sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def available_gib() -> float | None:
    """free + inactive + speculative pages, from vm_stat."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    except OSError:
        return None
    page = int(re.search(r"page size of (\d+)", out).group(1))
    pages = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative"):
        m = re.search(rf"{key}:\s+(\d+)", out)
        if m:
            pages += int(m.group(1))
    return pages * page / GIB


def gpu_cores() -> str | None:
    try:
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"], capture_output=True, text=True, timeout=30
        ).stdout
        for gpu in json.loads(out).get("SPDisplaysDataType", []):
            if gpu.get("sppci_cores"):
                return str(gpu["sppci_cores"])
    except Exception:
        pass
    return None


def main() -> int:
    report = Report("check_4_hardware", "Chip and memory for local UI-TARS")
    if not require_macos(report):
        return report.finish()

    chip = sysctl("machdep.cpu.brand_string")
    arm = sysctl("hw.optional.arm64") == "1"
    mem_gib = int(sysctl("hw.memsize") or 0) / GIB
    avail = available_gib()
    disk_free = shutil.disk_usage(Path.home()).free / GIB
    cores = gpu_cores()
    report.data.update(
        {
            "chip": chip,
            "apple_silicon": arm,
            "memory_gib": round(mem_gib, 1),
            "available_now_gib": round(avail, 1) if avail is not None else None,
            "disk_free_gib": round(disk_free, 1),
            "gpu_cores": cores,
            "macos": platform.mac_ver()[0],
        }
    )
    report.add(INFO, "chip", f"{chip} (GPU cores: {cores or 'unknown'}), macOS {platform.mac_ver()[0]}")
    report.add(INFO, "memory", f"{mem_gib:.0f} GiB total, ~{avail:.1f} GiB available right now" if avail else f"{mem_gib:.0f} GiB total")
    report.add(INFO, "disk free in home folder", f"{disk_free:.0f} GiB")

    if not arm:
        report.add(FAIL, "Apple Silicon", "Intel Mac: local vision models will be very slow; use Claude for grounding (option B)")
        return report.finish()
    report.add(PASS, "Apple Silicon", "unified memory can run a local vision model on the GPU")

    budget = mem_gib - OS_AND_APPS_GIB
    fitting = [(name, need) for name, need in CANDIDATES if need <= budget]
    report.data["candidates"] = [
        {"model": name, "est_gib": need, "fits_estimate": need <= budget} for name, need in CANDIDATES
    ]
    if not fitting:
        report.add(
            WARN,
            "recommendation (UNVERIFIED estimate)",
            f"{mem_gib:.0f} GiB leaves ~{budget:.0f} GiB after macOS/apps; even 4-bit (~6.5 GiB est.) is tight. "
            "Prefer Claude for grounding (option B), or try 4-bit with other apps closed.",
        )
    else:
        # Prefer 8-bit when it fits: 4-bit can cost coordinate accuracy (unmeasured here).
        best = next((c for c in fitting if "8-bit" in c[0]), fitting[0])
        report.add(
            PASS,
            "recommendation (UNVERIFIED estimate)",
            f"start with {best[0]} (~{best[1]} GiB est.); fits within ~{budget:.0f} GiB after macOS/apps. "
            "Confirm speed and accuracy with check 5.",
        )
    if disk_free < 20:
        report.add(WARN, "disk space", "less than 20 GiB free; model downloads are roughly 5–17 GB")

    runtimes = {
        "ollama": shutil.which("ollama"),
        "LM Studio": "/Applications/LM Studio.app" if Path("/Applications/LM Studio.app").exists() else None,
        "mlx_vlm (python)": shutil.which("mlx_vlm.server") or shutil.which("mlx_vlm.generate"),
        "llama.cpp server": shutil.which("llama-server"),
    }
    report.data["runtimes"] = runtimes
    found = [k for k, v in runtimes.items() if v]
    report.add(
        INFO,
        "local model runtimes",
        ("found: " + ", ".join(found)) if found else "none found; LM Studio or mlx-vlm are free options (see README)",
    )
    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
