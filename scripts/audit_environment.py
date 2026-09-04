#!/usr/bin/env python3
"""Phase 0 - Environment audit.

Records everything needed to reproduce a run: interpreter, key library
versions, CUDA/driver, GPU inventory, git commit hashes of every repository
this project depends on, and the full pip freeze.

Writes:
    <outdir>/environment.txt    human-readable + pip freeze
    <outdir>/environment.yaml   machine-readable (conda-style env spec)
    <outdir>/environment.json   machine-readable full record

Nothing here installs or upgrades anything.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Packages whose exact version materially changes results.
KEY_PACKAGES = [
    "torch", "torchaudio", "transformers", "qwen-asr", "peft", "accelerate",
    "datasets", "tokenizers", "numpy", "scipy", "librosa", "soundfile",
    "flash-attn", "deepspeed", "safetensors", "jiwer", "pandas", "matplotlib",
]

# Repositories whose commit hash must be pinned in the paper.
DEFAULT_REPOS = [
    "F:/Qwen_codes/Qwen3-ASR-main",
    "F:/qwen3_asr_hospital",
]


def run(cmd, shell=False):
    """Run a command, returning stripped stdout or an ERROR: marker."""
    try:
        out = subprocess.run(cmd, shell=shell, capture_output=True,
                             text=True, timeout=120)
        if out.returncode != 0:
            return "ERROR: rc=%d %s" % (out.returncode, out.stderr.strip()[:400])
        return out.stdout.strip()
    except Exception as exc:  # the audit must never crash the pipeline
        return "ERROR: %s" % exc


def package_versions():
    """Resolve installed versions via importlib.metadata (avoids heavy imports)."""
    from importlib import metadata

    versions = {}
    for name in KEY_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "NOT INSTALLED"
    return versions


def torch_info():
    """CUDA/GPU facts as seen by torch. Guarded: torch may be absent."""
    info = {}
    try:
        import torch
    except Exception as exc:
        return {"error": "torch import failed: %s" % exc}

    info["torch_version"] = torch.__version__
    info["cuda_compiled_version"] = torch.version.cuda
    info["cudnn_version"] = torch.backends.cudnn.version()
    info["cuda_available"] = torch.cuda.is_available()
    devices = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "index": i,
                "name": props.name,
                "total_memory_GiB": round(props.total_memory / 1024 ** 3, 2),
                "capability": "%d.%d" % (props.major, props.minor),
                "multi_processor_count": props.multi_processor_count,
            })
    info["devices"] = devices
    try:
        info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception:
        info["bf16_supported"] = None
    return info


def git_info(repos):
    """Commit hash + dirty flag for every dependency repo that exists."""
    result = {}
    for repo in repos:
        path = Path(repo)
        if not (path / ".git").exists():
            result[repo] = {"status": "not a git repo or missing"}
            continue
        result[repo] = {
            "commit": run(["git", "-C", str(path), "rev-parse", "HEAD"]),
            "branch": run(["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"]),
            "remote": run(["git", "-C", str(path), "config", "--get", "remote.origin.url"]),
            "dirty": "yes" if run(["git", "-C", str(path), "status", "--porcelain"]) else "no",
            "last_commit_date": run(["git", "-C", str(path), "log", "-1", "--format=%cI"]),
        }
    return result


def collect(repos):
    tracked_env = ("CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_ENDPOINT",
                   "TRANSFORMERS_CACHE", "PYTHONHASHSEED", "OMP_NUM_THREADS")
    return {
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "env_vars": {k: os.environ[k] for k in tracked_env if k in os.environ},
        "packages": package_versions(),
        "torch": torch_info(),
        "nvidia_smi": (
            run(["nvidia-smi",
                 "--query-gpu=name,driver_version,memory.total,compute_cap",
                 "--format=csv,noheader"])
            if shutil.which("nvidia-smi") else "nvidia-smi not found"),
        "nvcc": run(["nvcc", "--version"]) if shutil.which("nvcc") else "nvcc not found",
        "git": git_info(repos),
        "pip_freeze": run([sys.executable, "-m", "pip", "freeze"]),
    }


def write_txt(rec, path):
    lines = [
        "=" * 78,
        "QWEN3-ASR HOSPITAL DOMAIN ADAPTATION - ENVIRONMENT RECORD",
        "=" * 78,
        "recorded_at       : %s" % rec["recorded_at"],
        "hostname          : %s" % rec["hostname"],
        "platform          : %s (%s)" % (rec["platform"], rec["machine"]),
        "python            : %s" % rec["python_version"],
        "python_executable : %s" % rec["python_executable"],
        "cpu_count         : %s" % rec["cpu_count"],
        "",
        "-- environment variables ------------------------------------------------",
    ]
    lines += (["%-20s: %s" % (k, v) for k, v in rec["env_vars"].items()]
              or ["(none of the tracked variables are set)"])
    lines += ["", "-- key package versions -------------------------------------------------"]
    lines += ["%-20s: %s" % (k, v) for k, v in rec["packages"].items()]
    lines += ["", "-- torch / cuda ---------------------------------------------------------"]
    for key, val in rec["torch"].items():
        if key == "devices":
            for dev in val:
                lines.append("  GPU[%d]            : %s %sGiB sm_%s" % (
                    dev["index"], dev["name"], dev["total_memory_GiB"],
                    dev["capability"]))
        else:
            lines.append("%-20s: %s" % (key, val))
    lines += ["", "-- nvidia-smi -----------------------------------------------------------",
              str(rec["nvidia_smi"]),
              "", "-- nvcc -----------------------------------------------------------------",
              str(rec["nvcc"]),
              "", "-- git commits ----------------------------------------------------------"]
    for repo, meta in rec["git"].items():
        lines.append(repo)
        lines += ["    %-16s: %s" % (k, v) for k, v in meta.items()]
    lines += ["", "-- pip freeze -----------------------------------------------------------",
              str(rec["pip_freeze"]), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_yaml(rec, path):
    """Hand-rolled YAML so this script keeps zero third-party dependencies."""
    pip_lines = [ln for ln in str(rec["pip_freeze"]).splitlines() if ln.strip()]
    out = [
        "# Auto-generated by scripts/audit_environment.py - do not edit by hand.",
        "name: qwen3_asr_hospital",
        "# recorded_at: %s" % rec["recorded_at"],
        "# hostname: %s" % rec["hostname"],
        "# platform: %s" % rec["platform"],
        "channels:",
        "  - defaults",
        "dependencies:",
        "  - python=%s" % sys.version.split()[0],
        "  - pip",
        "  - pip:",
    ]
    out += ["      - %s" % ln for ln in pip_lines]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Record the execution environment.")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1]),
                    help="Directory to write environment.{txt,yaml,json} into.")
    ap.add_argument("--repos", nargs="*", default=DEFAULT_REPOS,
                    help="Repositories whose commit hashes should be pinned.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rec = collect(list(args.repos))
    write_txt(rec, outdir / "environment.txt")
    write_yaml(rec, outdir / "environment.yaml")
    (outdir / "environment.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")

    print("wrote %s" % (outdir / "environment.txt"))
    print("wrote %s" % (outdir / "environment.yaml"))
    print("wrote %s" % (outdir / "environment.json"))
    missing = [k for k, v in rec["packages"].items() if v == "NOT INSTALLED"]
    if missing:
        print("NOT INSTALLED: %s" % ", ".join(missing))


if __name__ == "__main__":
    main()
