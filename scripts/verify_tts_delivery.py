#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acceptance check for a delivered TTS corpus.

Run this the moment the synthesis team hands over manifests. It answers, in one
command, whether the delivery is usable — before any GPU time is spent training
on it.

The checks are ordered by how badly a failure would corrupt the study:

  FATAL   speaker pools overlap between train / dev / test
          This silently destroys speaker-disjointness. Every "held-out" test
          number would be inflated by voices the model already heard, and the
          error is invisible in the CER itself. Nothing else matters if this
          fails.
  FATAL   text does not match the source script, character for character
          The manifest text is the training target; if it drifted from the
          script the whole corpus is mislabelled.
  FATAL   audio missing, unreadable, or not 16 kHz mono
  ERROR   train scripts not rendered exactly twice with two different voices
          (the corpus needs ~2x coverage to reach the 20-hour budget)
  ERROR   total training hours below the largest budget
  ERROR   condition is not "clean" / snr or sir set at synthesis time
          Augmentation is applied downstream; pre-baked noise cannot be undone.
  WARN    QC removal rate above 2%
  WARN    one voice dominates its split

Exit code is non-zero if any FATAL or ERROR fires, so this can gate a pipeline.

    python scripts/verify_tts_delivery.py
    python scripts/verify_tts_delivery.py --require_hours 20 --strict
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from data.manifest import read_manifest, validate_manifest  # noqa: E402

FATAL, ERROR, WARN, OK = "FATAL", "ERROR", "WARN", "OK"


class Report:
    def __init__(self):
        self.findings = []

    def add(self, level, check, detail):
        self.findings.append({"level": level, "check": check, "detail": detail})

    def count(self, level):
        return sum(1 for f in self.findings if f["level"] == level)

    def failed(self):
        return self.count(FATAL) + self.count(ERROR) > 0


def load_scripts(split):
    path = _ROOT / "data" / "manifests" / "splits" / ("%s_scripts.jsonl" % split)
    if not path.exists():
        return None
    return {r["script_id"]: r for r in read_manifest(path)}


def check_audio(rows, report, sample_limit):
    """Verify format on a sample; a full pass would read every file twice."""
    try:
        import soundfile as sf
    except ImportError:
        report.add(WARN, "audio format", "soundfile unavailable, skipped")
        return

    bad_format, missing, duration_mismatch = [], [], []
    step = max(1, len(rows) // sample_limit) if sample_limit else 1
    checked = 0
    for row in rows[::step]:
        path = Path(row["audio"])
        if not path.exists():
            missing.append(row["utt_id"])
            continue
        try:
            info = sf.info(str(path))
        except Exception as exc:
            bad_format.append("%s: unreadable (%s)" % (row["utt_id"], exc))
            continue
        checked += 1
        if info.samplerate != 16000 or info.channels != 1:
            bad_format.append("%s: %d Hz, %d ch"
                              % (row["utt_id"], info.samplerate, info.channels))
        actual = info.frames / float(info.samplerate)
        declared = row.get("duration") or 0
        if declared and abs(actual - declared) > 0.05:
            duration_mismatch.append("%s: manifest %.2fs vs file %.2fs"
                                     % (row["utt_id"], declared, actual))

    if missing:
        report.add(FATAL, "audio present",
                   "%d referenced files do not exist (e.g. %s)"
                   % (len(missing), ", ".join(missing[:3])))
    if bad_format:
        report.add(FATAL, "audio format 16kHz mono",
                   "%d files wrong (e.g. %s)" % (len(bad_format), bad_format[0]))
    if duration_mismatch:
        report.add(ERROR, "declared duration matches audio",
                   "%d mismatch by >50ms (e.g. %s). Budgets are cut by audio "
                   "hours, so a wrong duration silently corrupts D1/D5/D10/D20"
                   % (len(duration_mismatch), duration_mismatch[0]))
    if not (missing or bad_format or duration_mismatch):
        report.add(OK, "audio", "%d sampled files: exist, 16 kHz mono, "
                                "durations match" % checked)


def main():
    ap = argparse.ArgumentParser(description="Accept or reject a TTS delivery.")
    ap.add_argument("--manifests_dir", default=str(_ROOT / "data" / "manifests"))
    ap.add_argument("--splits", nargs="*", default=["train", "dev", "test"])
    ap.add_argument("--pattern", default="%s_synthetic.jsonl")
    ap.add_argument("--require_hours", type=float, default=20.0,
                    help="largest training budget the train split must cover")
    ap.add_argument("--expect_train_renderings", type=int, default=2)
    ap.add_argument("--max_speaker_share", type=float, default=0.10)
    ap.add_argument("--max_removal_rate", type=float, default=0.02)
    ap.add_argument("--audio_sample", type=int, default=400,
                    help="how many files to open for format checking (0 = all)")
    ap.add_argument("--out", default=str(_ROOT / "results" / "metrics" / "tts_delivery_check.json"))
    args = ap.parse_args()

    report = Report()
    manifests = OrderedDict()
    speakers = OrderedDict()

    print("=" * 74)
    print("TTS DELIVERY ACCEPTANCE CHECK")
    print("=" * 74)

    for split in args.splits:
        path = Path(args.manifests_dir) / (args.pattern % split)
        if not path.exists():
            report.add(FATAL, "manifest present", "%s is missing" % path)
            print("  %-8s MISSING (%s)" % (split, path))
            continue
        rows = read_manifest(path)
        manifests[split] = rows
        speakers[split] = {r.get("speaker_id") for r in rows}
        hours = sum(r.get("duration", 0) or 0 for r in rows) / 3600.0
        print("  %-8s %6d utterances  %6.2f h  %2d voices"
              % (split, len(rows), hours, len(speakers[split])))

    if not manifests:
        print("\nnothing to check")
        sys.exit(2)

    # ---- FATAL: speaker pools must be disjoint --------------------------
    print("")
    print("-- speaker disjointness (the check that matters most) --")
    names = list(speakers)
    clean = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = speakers[names[i]] & speakers[names[j]]
            if overlap:
                clean = False
                report.add(FATAL, "speaker pools disjoint",
                           "%s and %s share %d voices: %s. Held-out results "
                           "would be inflated by voices the model trained on"
                           % (names[i], names[j], len(overlap),
                              sorted(overlap)[:5]))
                print("  %s / %s  OVERLAP: %s" % (names[i], names[j],
                                                  sorted(overlap)[:5]))
            else:
                print("  %s / %s  disjoint" % (names[i], names[j]))
    if clean:
        report.add(OK, "speaker pools disjoint", "no voice appears in two splits")

    # ---- FATAL: text must match the source scripts ----------------------
    print("")
    print("-- text fidelity against source scripts --")
    for split, rows in manifests.items():
        scripts = load_scripts(split)
        if scripts is None:
            report.add(WARN, "script source", "no split file for %s" % split)
            continue
        mismatched, unknown = [], []
        for row in rows:
            script_id = row.get("script_id")
            if script_id not in scripts:
                unknown.append(row["utt_id"])
                continue
            if row["text"] != scripts[script_id]["text"]:
                mismatched.append("%s: %r vs %r"
                                  % (row["utt_id"], row["text"],
                                     scripts[script_id]["text"]))
        if mismatched:
            report.add(FATAL, "text matches script",
                       "%s: %d texts differ from the source script (e.g. %s)"
                       % (split, len(mismatched), mismatched[0]))
            print("  %-8s %d MISMATCHED" % (split, len(mismatched)))
        elif unknown:
            report.add(ERROR, "script_id resolvable",
                       "%s: %d rows have a script_id not in the split file"
                       % (split, len(unknown)))
            print("  %-8s %d unknown script_id" % (split, len(unknown)))
        else:
            print("  %-8s all %d texts match character for character"
                  % (split, len(rows)))
            report.add(OK, "text matches script", "%s clean" % split)

    # ---- train coverage and budget --------------------------------------
    if "train" in manifests:
        rows = manifests["train"]
        hours = sum(r.get("duration", 0) or 0 for r in rows) / 3600.0
        print("")
        print("-- training budget --")
        print("  audio hours: %.2f (need >= %.1f)" % (hours, args.require_hours))
        if hours < args.require_hours:
            report.add(ERROR, "training hours",
                       "%.2f h available but the largest budget needs %.1f h; "
                       "D20 would be short" % (hours, args.require_hours))
        else:
            report.add(OK, "training hours", "%.2f h" % hours)

        per_script = Counter(r.get("script_id") for r in rows)
        counts = Counter(per_script.values())
        print("  renderings per script: %s" % dict(counts))
        wrong = {k: v for k, v in counts.items() if k != args.expect_train_renderings}
        if wrong:
            report.add(ERROR, "renderings per script",
                       "expected exactly %d per script, found %s"
                       % (args.expect_train_renderings, dict(counts)))

        # The two renderings of one script must use two different voices.
        by_script = defaultdict(set)
        for row in rows:
            by_script[row.get("script_id")].add(row.get("speaker_id"))
        same_voice = [s for s, v in by_script.items()
                      if len(v) < min(args.expect_train_renderings,
                                      per_script[s])]
        if same_voice:
            report.add(ERROR, "renderings use distinct voices",
                       "%d scripts were rendered more than once with the same "
                       "voice (e.g. %s)" % (len(same_voice), same_voice[:3]))
        else:
            report.add(OK, "renderings use distinct voices", "all scripts")

    # ---- condition must be clean ----------------------------------------
    print("")
    print("-- condition purity (augmentation happens downstream) --")
    for split, rows in manifests.items():
        conditions = Counter(r.get("condition") for r in rows)
        prebaked = [r["utt_id"] for r in rows
                    if r.get("condition") != "clean"
                    or r.get("snr") is not None or r.get("sir") is not None]
        print("  %-8s %s" % (split, dict(conditions)))
        if prebaked:
            report.add(ERROR, "condition is clean",
                       "%s: %d rows are not clean or carry snr/sir. Noise baked "
                       "in at synthesis cannot be removed, and the robustness "
                       "experiment needs a clean baseline"
                       % (split, len(prebaked)))
        else:
            report.add(OK, "condition is clean", "%s all clean" % split)

    # ---- speaker balance -------------------------------------------------
    print("")
    print("-- speaker balance --")
    for split, rows in manifests.items():
        counts = Counter(r.get("speaker_id") for r in rows)
        if not counts:
            continue
        top, n = counts.most_common(1)[0]
        share = n / len(rows)
        print("  %-8s busiest voice %s at %.1f%%" % (split, top, 100 * share))
        if share > args.max_speaker_share:
            report.add(WARN, "speaker balance",
                       "%s: %s covers %.1f%% (cap %.0f%%)"
                       % (split, top, 100 * share, 100 * args.max_speaker_share))

    # ---- schema + audio --------------------------------------------------
    print("")
    print("-- manifest schema --")
    for split, rows in manifests.items():
        result = validate_manifest(rows)
        print("  %-8s %d errors, %d warnings"
              % (split, result["n_errors"], result["n_warnings"]))
        if result["n_errors"]:
            report.add(FATAL, "manifest schema",
                       "%s: %d errors (e.g. %s)"
                       % (split, result["n_errors"], result["errors"][0]))

    print("")
    print("-- audio files --")
    for split, rows in manifests.items():
        check_audio(rows, report, args.audio_sample)

    # ---- QC removal logs -------------------------------------------------
    for split in manifests:
        removal = _ROOT / "data" / "synthetic" / ("removed_%s.jsonl" % split)
        if removal.exists():
            removed = sum(1 for line in open(removal, encoding="utf-8") if line.strip())
            total = len(manifests[split]) + removed
            rate = removed / total if total else 0.0
            if rate > args.max_removal_rate:
                report.add(WARN, "QC removal rate",
                           "%s: %.2f%% removed (cap %.0f%%), usually a bad anchor"
                           % (split, 100 * rate, 100 * args.max_removal_rate))

    # ---- verdict ---------------------------------------------------------
    print("")
    print("=" * 74)
    for level in (FATAL, ERROR, WARN):
        for finding in report.findings:
            if finding["level"] == level:
                print("%-6s %-34s %s" % (level, finding["check"], finding["detail"]))
    print("-" * 74)
    print("%d FATAL, %d ERROR, %d WARN, %d OK"
          % (report.count(FATAL), report.count(ERROR), report.count(WARN),
             report.count(OK)))
    verdict = "REJECT" if report.failed() else "ACCEPT"
    print("VERDICT: %s" % verdict)
    print("=" * 74)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"verdict": verdict, "findings": report.findings,
         "splits": {k: {"n": len(v),
                        "hours": round(sum(r.get("duration", 0) or 0
                                           for r in v) / 3600.0, 3),
                        "voices": sorted(speakers[k])}
                    for k, v in manifests.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote %s" % out_path)
    sys.exit(1 if report.failed() else 0)


if __name__ == "__main__":
    main()
