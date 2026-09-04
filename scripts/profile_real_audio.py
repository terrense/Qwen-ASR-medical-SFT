#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Profile the real hospital recordings before annotation.

These were captured on site, so their acoustic conditions are mixed rather than
designed. Phase 13 wants to report CER separately for quiet / far-field / noisy
speech, which requires knowing what is actually in the set.

Every measure here is computed from the waveform - nothing is inferred from a
model, and no transcript is required:

  speech//noise segmentation   A simple energy-based VAD splits each file into
                               speech and non-speech frames using a threshold
                               relative to that file's own energy distribution.
  estimated SNR                Ratio of mean speech-frame energy to mean
                               non-speech-frame energy, in dB. This is an
                               *estimate*: with no clean reference it cannot be
                               a true SNR, and it is labelled as such everywhere.
  reverberation proxy          Decay slope of the energy envelope after speech
                               offsets. Longer decay suggests a more reverberant
                               or more distant capture.
  spectral centroid / rolloff  Distance and channel effects move energy down in
                               frequency; far-field capture typically lowers the
                               centroid.
  clipping / silence ratio     Straight QC quantities.

The output is a per-file CSV plus a summary that buckets files into tentative
condition tiers. **The tiers are a starting point for the annotators, not a
label**: the annotation form asks a human to confirm the condition, and the
human answer is what Phase 13 uses.

    python scripts/profile_real_audio.py --audio_dir F:/asr_final_20260807plus
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402

FRAME_MS = 25
HOP_MS = 10


def frame_energies(signal, sample_rate):
    frame = int(sample_rate * FRAME_MS / 1000)
    hop = int(sample_rate * HOP_MS / 1000)
    if len(signal) < frame:
        return np.array([]), frame, hop
    n_frames = 1 + (len(signal) - frame) // hop
    strides = np.lib.stride_tricks.as_strided(
        signal, shape=(n_frames, frame),
        strides=(signal.strides[0] * hop, signal.strides[0]))
    return (strides ** 2).mean(axis=1), frame, hop


def analyse(path):
    import soundfile as sf

    signal, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    duration = len(signal) / float(sample_rate)

    energies, _, _ = frame_energies(signal, sample_rate)
    if energies.size == 0:
        return None

    db = 10 * np.log10(energies + 1e-12)
    # Threshold relative to this file's own distribution: robust to overall gain.
    floor = np.percentile(db, 10)
    peak = np.percentile(db, 95)
    threshold = floor + 0.45 * (peak - floor)
    speech = db > threshold

    speech_energy = energies[speech].mean() if speech.any() else 0.0
    noise_energy = energies[~speech].mean() if (~speech).any() else 1e-12
    snr_estimate = 10 * np.log10((speech_energy + 1e-12) / (noise_energy + 1e-12))

    # Reverberation proxy: median decay slope (dB/s) over the 100 ms following
    # each speech offset.
    slopes = []
    tail_frames = int(100 / HOP_MS)
    offsets = np.flatnonzero((speech[:-1].astype(int) - speech[1:].astype(int)) == 1)
    for offset in offsets:
        tail = db[offset + 1: offset + 1 + tail_frames]
        if len(tail) >= 4:
            x = np.arange(len(tail)) * (HOP_MS / 1000.0)
            slope = np.polyfit(x, tail, 1)[0]
            if np.isfinite(slope):
                slopes.append(slope)
    decay_slope = float(np.median(slopes)) if slopes else float("nan")

    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sample_rate)
    power = spectrum ** 2
    total = power.sum() + 1e-12
    centroid = float((freqs * power).sum() / total)
    cumulative = np.cumsum(power) / total
    rolloff = float(freqs[np.searchsorted(cumulative, 0.85)])

    return OrderedDict([
        ("file", path.name),
        ("duration", round(duration, 3)),
        ("sample_rate", sample_rate),
        ("rms", round(float(np.sqrt((signal ** 2).mean())), 6)),
        ("peak", round(float(np.abs(signal).max()), 4)),
        ("clipping_ratio", round(float((np.abs(signal) >= 0.999).mean()), 6)),
        ("speech_ratio", round(float(speech.mean()), 4)),
        ("silence_ratio", round(float(1 - speech.mean()), 4)),
        ("snr_estimate_db", round(float(snr_estimate), 2)),
        ("decay_slope_db_per_s", round(decay_slope, 1) if decay_slope == decay_slope else None),
        ("spectral_centroid_hz", round(centroid, 1)),
        ("rolloff85_hz", round(rolloff, 1)),
    ])


def tier(record):
    """Tentative condition bucket. A human confirms or overrides this."""
    snr = record["snr_estimate_db"]
    centroid = record["spectral_centroid_hz"]
    if snr >= 20:
        base = "likely_quiet"
    elif snr >= 12:
        base = "likely_moderate"
    else:
        base = "likely_noisy"
    # A low centroid with a mediocre SNR is the signature of distant capture.
    if base != "likely_quiet" and centroid < 900:
        base = "likely_farfield_or_muffled"
    return base


def main():
    ap = argparse.ArgumentParser(description="Acoustic profile of real recordings.")
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out", default=str(_ROOT / "results" / "metrics" / "real_audio_profile.csv"))
    ap.add_argument("--summary", default=str(_ROOT / "results" / "metrics" / "real_audio_profile.json"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    paths = sorted(audio_dir.glob("*.wav"))
    if args.limit:
        paths = paths[:args.limit]
    print("profiling %d files from %s" % (len(paths), audio_dir))

    records, failed = [], []
    for index, path in enumerate(paths, 1):
        try:
            record = analyse(path)
        except Exception as exc:
            failed.append({"file": path.name, "error": str(exc)})
            continue
        if record is None:
            failed.append({"file": path.name, "error": "too short to frame"})
            continue
        record["tier"] = tier(record)
        records.append(record)
        if index % 400 == 0:
            print("  %d/%d" % (index, len(paths)))

    if not records:
        sys.exit("no files could be profiled")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    def stats(field):
        values = [r[field] for r in records if r[field] is not None]
        if not values:
            return None
        values = sorted(values)
        n = len(values)
        return {"min": round(values[0], 2), "p10": round(values[int(0.1 * n)], 2),
                "median": round(values[n // 2], 2),
                "p90": round(values[int(0.9 * n)], 2),
                "max": round(values[-1], 2),
                "mean": round(float(np.mean(values)), 2)}

    tiers = Counter(r["tier"] for r in records)
    summary = OrderedDict([
        ("audio_dir", str(audio_dir)),
        ("n_files", len(records)),
        ("n_failed", len(failed)),
        ("total_hours", round(sum(r["duration"] for r in records) / 3600.0, 3)),
        ("duration", stats("duration")),
        ("snr_estimate_db", stats("snr_estimate_db")),
        ("spectral_centroid_hz", stats("spectral_centroid_hz")),
        ("rolloff85_hz", stats("rolloff85_hz")),
        ("speech_ratio", stats("speech_ratio")),
        ("clipping_ratio", stats("clipping_ratio")),
        ("decay_slope_db_per_s", stats("decay_slope_db_per_s")),
        ("tiers", dict(tiers)),
        ("tier_pct", {k: round(100.0 * v / len(records), 1) for k, v in tiers.items()}),
        ("failed", failed[:20]),
        ("note", "SNR is an energy-ratio estimate from a single channel, not a "
                 "true SNR; tiers are a starting point for annotators, and the "
                 "human-confirmed condition is what Phase 13 reports."),
    ])
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    print("")
    print("files      : %d  (%.2f hours)" % (len(records), summary["total_hours"]))
    for field in ("duration", "snr_estimate_db", "spectral_centroid_hz",
                  "speech_ratio", "clipping_ratio"):
        s = summary[field]
        print("%-22s min %8.2f  p10 %8.2f  median %8.2f  p90 %8.2f  max %8.2f"
              % (field, s["min"], s["p10"], s["median"], s["p90"], s["max"]))
    print("")
    print("tentative condition tiers (human confirms during annotation):")
    for name, count in tiers.most_common():
        print("  %-28s %5d  %5.1f%%" % (name, count, 100.0 * count / len(records)))
    if failed:
        print("")
        print("%d files failed to profile" % len(failed))
    print("")
    print("wrote %s" % out_path)
    print("wrote %s" % args.summary)


if __name__ == "__main__":
    main()
