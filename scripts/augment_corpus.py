#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 7 - Apply acoustic augmentation to a manifest.

Reads a clean manifest, assigns each utterance a condition, renders the
augmented waveform, and writes a new manifest whose ``condition``, ``snr`` and
``sir`` fields describe what was actually done.

Two invariants the implementation enforces rather than assumes:

  the transcript always belongs to the foreground speaker
      A competing-speech interferer is drawn from a *different speaker* and a
      *different script*. Both constraints are checked per utterance, and the
      interferer's identity is written into the output manifest so any
      contamination can be audited later.

  regeneration is exact
      Every random choice derives from ``(seed, utt_id)``, so re-running
      reproduces each waveform bit-for-bit and a single utterance can be rebuilt
      without replaying the corpus.

Noise and impulse responses come from ``--noise_dir`` / ``--rir_dir`` when given.
Without them the module synthesizes colored noise and exponential-decay impulse
responses, and records ``source: synthetic`` in every affected row, so a corpus
built on synthetic noise can never be mistaken for one built on recordings.

    python scripts/augment_corpus.py \
        --manifest data/manifests/train_20h.jsonl \
        --outdir data/synthetic/audio/train_aug \
        --out_manifest data/manifests/train_20h_aug.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402

from augmentation import acoustic as A  # noqa: E402
from data import audio_qc  # noqa: E402
from data.manifest import (format_report, read_manifest,  # noqa: E402
                           validate_manifest, write_manifest)


def build_competing_provider(rows, row, cache, max_attempts=25):
    """Return a provider yielding an interferer from another speaker+script.

    The provider is a closure over the current row so the two disjointness
    constraints are checked at draw time; if no candidate satisfies them the
    provider returns ``None`` and ``augment`` records the substitution instead
    of silently mixing in the same voice.
    """

    def provider(rng):
        for _ in range(max_attempts):
            candidate = rows[int(rng.integers(0, len(rows)))]
            if candidate["utt_id"] == row["utt_id"]:
                continue
            if candidate.get("speaker_id") == row.get("speaker_id"):
                continue
            if candidate.get("script_id") and \
                    candidate.get("script_id") == row.get("script_id"):
                continue
            path = candidate["audio"]
            if path not in cache:
                try:
                    signal, _ = audio_qc.load_audio(path)
                except Exception:
                    cache[path] = None
                else:
                    cache[path] = signal
            if cache[path] is None or len(cache[path]) == 0:
                continue
            return cache[path], {"utt_id": candidate["utt_id"],
                                 "speaker_id": candidate.get("speaker_id")}
        raise LookupError("no valid interferer")

    def guarded(rng):
        try:
            return provider(rng)
        except LookupError:
            raise

    return guarded


def collect_files(source, kind=None, extensions=(".wav", ".flac")):
    """Resolve an audio source to a file list.

    Accepts either a directory to scan, or the screening index written by
    ``scripts/prepare_noise_rir.py``. Preferring the index matters: it has
    already rejected unreadable, silent, over-long and truncated files, and
    re-scanning the directory would quietly pull those rejects back in.
    """
    if not source:
        return []
    path = Path(source)

    if path.is_file() and path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get(kind or "noise", [])
        files = [e["path"] for e in entries if Path(e["path"]).exists()]
        print("using screened index %s: %d %s files"
              % (path.name, len(files), kind or "noise"))
        return files

    if not path.is_dir():
        print("WARNING: %s is neither a directory nor an index JSON; ignoring"
              % source)
        return []
    return sorted(str(p) for p in path.rglob("*")
                  if p.suffix.lower() in extensions)


def main():
    ap = argparse.ArgumentParser(description="Apply acoustic augmentation.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True, help="where augmented wavs go")
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise_dir", default=None,
                    help="directory of noise audio, or the screening index JSON")
    ap.add_argument("--rir_dir", default=None,
                    help="directory of impulse responses, or the screening index JSON")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--keep_clean_audio", action="store_true",
                    help="reuse the source file for clean rows instead of "
                         "writing an identical copy")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    print("source manifest: %s (%d utterances)" % (args.manifest, len(rows)))

    noise_paths = collect_files(args.noise_dir, kind="noise")
    rir_paths = collect_files(args.rir_dir, kind="rir")
    print("noise files    : %d%s" % (len(noise_paths),
                                     "" if noise_paths else " (synthetic fallback)"))
    print("impulse responses: %d%s" % (len(rir_paths),
                                       "" if rir_paths else " (synthetic fallback)"))
    bank = A.NoiseBank(noise_paths, rir_paths)

    plan = A.plan_conditions([r["utt_id"] for r in rows], args.seed)
    planned_counts = Counter(plan.values())
    print("")
    print("planned condition mix:")
    for name, target in A.CONDITION_WEIGHTS.items():
        share = planned_counts[name] / max(1, len(rows))
        print("  %-18s %6.2f%%  target %5.2f%%" % (name, 100 * share, 100 * target))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cache = {}
    out_rows = []
    failures = []
    realized = Counter()
    substitutions = 0
    t0 = time.time()

    for index, row in enumerate(rows, 1):
        condition = plan[row["utt_id"]]
        try:
            signal, _ = audio_qc.load_audio(row["audio"])
        except Exception as exc:
            failures.append({"utt_id": row["utt_id"], "audio": row["audio"],
                             "reason": "unreadable source: %s" % exc})
            continue

        provider = None
        if condition == "competing_speech":
            provider = build_competing_provider(rows, row, cache)

        try:
            augmented, meta = A.augment(
                signal, row["utt_id"], args.seed, bank, condition=condition,
                competing_provider=provider)
        except LookupError:
            # No admissible interferer: fall back to additive noise rather than
            # mixing a same-speaker utterance, and record the substitution.
            augmented, meta = A.augment(
                signal, row["utt_id"], args.seed, bank, condition="noise")
            meta["competing_speech_unavailable"] = True
            substitutions += 1
        except Exception as exc:
            failures.append({"utt_id": row["utt_id"], "audio": row["audio"],
                             "reason": "augmentation failed: %s" % exc})
            continue

        if meta.get("competing_speech_unavailable"):
            substitutions += 1

        realized[meta["condition"]] += 1

        if meta["condition"] == "clean" and args.keep_clean_audio:
            out_audio = row["audio"]
        else:
            out_audio = str(outdir / ("%s.wav" % row["utt_id"]))
            audio_qc.save_audio(out_audio, augmented)

        stats = audio_qc.measure(augmented)
        passed, reasons = audio_qc.evaluate(stats)
        if not passed:
            failures.append({"utt_id": row["utt_id"], "audio": out_audio,
                             "reason": "post-augmentation QC: %s" % reasons,
                             "stats": stats})
            continue

        new_row = OrderedDict(row)
        new_row["audio"] = out_audio
        new_row["duration"] = stats["duration"]
        new_row["condition"] = meta["condition"]
        new_row["snr"] = meta["snr"]
        new_row["sir"] = meta["sir"]
        new_row["augmentation"] = {k: v for k, v in meta.items()
                                   if k not in ("condition", "snr", "sir")}
        new_row["source_audio"] = row["audio"]
        out_rows.append(new_row)

        if index % args.log_every == 0 or index == len(rows):
            elapsed = time.time() - t0
            print("  %d/%d  (%.1fs, %.1f utt/s)"
                  % (index, len(rows), elapsed, index / max(elapsed, 1e-6)))

    write_manifest(out_rows, args.out_manifest)
    report = validate_manifest(out_rows)

    failure_log = Path(args.out_manifest).with_suffix(".failures.jsonl")
    with failure_log.open("w", encoding="utf-8") as handle:
        for item in failures:
            handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    total = max(1, sum(realized.values()))
    summary = {
        "source_manifest": args.manifest,
        "out_manifest": args.out_manifest,
        "seed": args.seed,
        "n_input": len(rows),
        "n_output": len(out_rows),
        "n_failed": len(failures),
        "failure_log": str(failure_log),
        "noise_source": "corpus" if noise_paths else "synthetic",
        "rir_source": "corpus" if rir_paths else "synthetic",
        "competing_speech_substitutions": substitutions,
        "planned_condition_counts": dict(planned_counts),
        "realized_condition_counts": dict(realized),
        "realized_condition_pct": {k: round(100.0 * v / total, 2)
                                   for k, v in realized.items()},
        "target_condition_pct": {k: round(100.0 * v, 2)
                                 for k, v in A.CONDITION_WEIGHTS.items()},
        "snr_range_db": list(A.SNR_RANGE),
        "sir_range_db": list(A.SIR_RANGE),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    summary_path = Path(args.out_manifest).with_suffix(".augmentation.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    print("")
    print(format_report(report, "augmented manifest"))
    print("")
    print("realized condition mix:")
    for name, target in A.CONDITION_WEIGHTS.items():
        print("  %-18s %6.2f%%  target %5.2f%%"
              % (name, 100.0 * realized[name] / total, 100 * target))
    if substitutions:
        print("")
        print("%d competing-speech utterances fell back to additive noise "
              "(no admissible interferer); each is flagged in its manifest row"
              % substitutions)
    if failures:
        print("%d utterances were dropped; see %s" % (len(failures), failure_log))
    print("")
    print("wrote %s" % args.out_manifest)
    print("wrote %s" % summary_path)


if __name__ == "__main__":
    main()
