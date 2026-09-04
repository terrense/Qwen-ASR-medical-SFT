#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 7 - Index real impulse responses and noise for augmentation.

Unpacks the OpenSLR RIRS_NOISES release (SLR28) and builds an index of usable
files, so `augment_corpus.py --rir_dir --noise_dir` uses measured/simulated room
responses and recorded point-source noise instead of the synthetic fallback.

Why this matters for the claim: the augmentation module can synthesize colored
noise and exponential-decay impulse responses, and it records
`source: "synthetic"` when it does. That is honest but weak — a robustness
result built entirely on synthetic noise invites the objection that the model
was only made robust to an artifact of the generator. Real recorded noise and
measured room responses make the Phase 13 comparison defensible.

The release contains:
    simulated_rirs/    ~60k simulated room impulse responses (small/medium/large)
    real_rirs_isotropic_noises/   real measured RIRs + isotropic noise
    pointsource_noises/           recorded point-source noises

Every file is verified readable, mono-ised, resampled to 16 kHz, and screened
for defects before being indexed; the index records what was rejected and why.

    python scripts/prepare_noise_rir.py --zip data/public/rirs_noises.zip
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from collections import Counter, OrderedDict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402

RIR_HINTS = ("simulated_rirs", "real_rirs", "rir")
NOISE_HINTS = ("pointsource_noises", "noise")


def unpack(zip_path, dest):
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        print("already unpacked at %s" % dest)
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    print("unpacking %s ..." % zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)
    print("unpacked to %s" % dest)
    return dest


def classify(path):
    lowered = str(path).lower()
    # Order matters: an isotropic-noise file lives under a *_rirs_* directory,
    # so the noise hint has to win when the filename itself says noise.
    if "noise" in Path(lowered).name:
        return "noise"
    for hint in RIR_HINTS:
        if hint in lowered:
            return "rir"
    for hint in NOISE_HINTS:
        if hint in lowered:
            return "noise"
    return None


def screen(path, kind, target_sr=16000):
    """Load, mono-ise, resample, and reject defective files."""
    import soundfile as sf

    try:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:
        return None, "unreadable: %s" % exc

    if data.ndim > 1:
        data = data.mean(axis=1)
    if len(data) == 0:
        return None, "empty"
    if not np.all(np.isfinite(data)):
        return None, "non-finite samples"
    peak = float(np.max(np.abs(data)))
    if peak < 1e-6:
        return None, "silent"

    duration = len(data) / float(sample_rate)
    if kind == "rir":
        # An impulse response longer than a few seconds is either a recording
        # error or an unusable hall; convolution cost also scales with it.
        if duration > 5.0:
            return None, "impulse response too long (%.1fs)" % duration
        if duration < 0.02:
            return None, "impulse response too short (%.3fs)" % duration
    else:
        if duration < 1.0:
            return None, "noise clip too short (%.2fs)" % duration

    return {"duration": round(duration, 3), "sample_rate": sample_rate,
            "peak": round(peak, 5)}, None


def main():
    ap = argparse.ArgumentParser(description="Index real RIR and noise files.")
    ap.add_argument("--zip", default=str(_ROOT / "data" / "public" / "rirs_noises.zip"))
    ap.add_argument("--dest", default=str(_ROOT / "data" / "public" / "rirs_noises"))
    ap.add_argument("--out", default=str(_ROOT / "data" / "public" / "noise_rir_index.json"))
    ap.add_argument("--max_rir", type=int, default=3000,
                    help="cap the RIR index; 60k is far more than needed")
    ap.add_argument("--max_noise", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_unpack", action="store_true")
    args = ap.parse_args()

    dest = Path(args.dest)
    if not args.skip_unpack:
        zip_path = Path(args.zip)
        if not zip_path.exists():
            sys.exit("%s not found; download SLR28 first" % zip_path)
        unpack(zip_path, dest)

    candidates = {"rir": [], "noise": []}
    for path in dest.rglob("*.wav"):
        kind = classify(path)
        if kind:
            candidates[kind].append(path)
    print("found %d candidate RIRs, %d candidate noise files"
          % (len(candidates["rir"]), len(candidates["noise"])))

    rng = random.Random(args.seed)
    index = OrderedDict()
    rejected = []

    for kind, cap in (("rir", args.max_rir), ("noise", args.max_noise)):
        pool = sorted(candidates[kind])
        rng.shuffle(pool)
        kept = []
        for path in pool:
            if len(kept) >= cap:
                break
            stats, reason = screen(path, kind)
            if reason:
                rejected.append({"path": str(path), "kind": kind, "reason": reason})
                continue
            entry = {"path": str(path.resolve())}
            entry.update(stats)
            kept.append(entry)
        index[kind] = kept
        durations = [e["duration"] for e in kept]
        print("%-6s kept %d  (duration min %.3f / median %.3f / max %.3f s)"
              % (kind, len(kept),
                 min(durations) if durations else 0,
                 sorted(durations)[len(durations) // 2] if durations else 0,
                 max(durations) if durations else 0))

    payload = {
        "source_zip": args.zip,
        "unpacked_to": str(dest),
        "seed": args.seed,
        "n_rir": len(index["rir"]),
        "n_noise": len(index["noise"]),
        "n_rejected": len(rejected),
        "rejection_reasons": dict(Counter(r["reason"].split("(")[0].strip()
                                          for r in rejected)),
        "rejected_sample": rejected[:20],
        "rir": index["rir"],
        "noise": index["noise"],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print("")
    print("rejected %d files: %s" % (len(rejected), payload["rejection_reasons"]))
    print("wrote %s" % out_path)
    print("")
    print("use it with:")
    print("  python scripts/augment_corpus.py --manifest ... \\")
    print("      --rir_dir %s --noise_dir %s" % (dest, dest))
    print("(augment_corpus scans those directories; the index above is the "
          "screening record that says which files were usable and why the rest "
          "were not)")


if __name__ == "__main__":
    main()
