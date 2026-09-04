#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 18 - Paired bootstrap significance test between two systems.

    python scripts/paired_test.py \
        --a experiments/qwen06_zero/test_aishell1/predictions.jsonl \
        --b experiments/qwen17_zero/test_aishell1/predictions.jsonl \
        --name_a "Qwen3-ASR-0.6B" --name_b "Qwen3-ASR-1.7B"

Why paired and not a plain difference of two CERs: the two systems are scored on
the *same* utterances, so most of the variance is shared. Resampling utterance
indices once and recomputing both systems' CER from that same index set removes
the shared variance and gives a far tighter, honest interval than treating the
two numbers as independent.

The pairing is asserted, never assumed: both files must cover the same utt_ids
in the same order with identical reference lengths. A mismatch means the two
runs used different normalization or different manifests, and comparing them
would be meaningless - so the script raises instead of reporting a number.

Reports the CER difference, a bootstrap confidence interval, and a two-sided
p-value. Also reports the utterances where the systems disagree most, which is
usually where the interesting behaviour is.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from evaluation.metrics import paired_bootstrap  # noqa: E402


def load(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Paired bootstrap between two runs.")
    ap.add_argument("--a", required=True, help="predictions.jsonl of system A")
    ap.add_argument("--b", required=True, help="predictions.jsonl of system B")
    ap.add_argument("--name_a", default="system A")
    ap.add_argument("--name_b", default="system B")
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--out", default=None)
    ap.add_argument("--top_disagreements", type=int, default=8)
    args = ap.parse_args()

    rows_a = load(args.a)
    rows_b = load(args.b)
    print("%s : %d utterances" % (args.name_a, len(rows_a)))
    print("%s : %d utterances" % (args.name_b, len(rows_b)))

    scorable_a = [r for r in rows_a if r.get("reference_length")]
    scorable_b = [r for r in rows_b if r.get("reference_length")]

    result = paired_bootstrap(scorable_a, scorable_b,
                              n_samples=args.n_samples, seed=args.seed,
                              confidence=args.confidence)

    print("")
    print("=" * 68)
    print("%-28s CER %.4f  (%.2f%%)" % (args.name_a, result["cer_a"],
                                        100 * result["cer_a"]))
    print("%-28s CER %.4f  (%.2f%%)" % (args.name_b, result["cer_b"],
                                        100 * result["cer_b"]))
    print("-" * 68)
    print("difference (A - B)         %+.4f  (%+.2f pp)"
          % (result["cer_difference"], 100 * result["cer_difference"]))
    print("%.0f%% CI                     [%+.4f, %+.4f]  ([%+.2f, %+.2f] pp)"
          % (100 * args.confidence, result["ci_lower"], result["ci_upper"],
             100 * result["ci_lower"], 100 * result["ci_upper"]))
    print("p-value (two-sided)        %.4f" % result["p_value"])
    print("bootstrap samples          %d  (seed %d)"
          % (result["n_bootstrap_samples"], result["seed"]))
    print("paired utterances          %d" % result["n_utterances"])
    print("-" * 68)
    if result["significant_at_confidence"]:
        better = args.name_b if result["cer_difference"] > 0 else args.name_a
        print("SIGNIFICANT: the interval excludes zero; %s is better" % better)
    else:
        print("NOT SIGNIFICANT: the interval includes zero, the two systems "
              "cannot be separated on this test set")
    print("=" * 68)

    # Where do they disagree most? Useful for error analysis, not for the claim.
    by_id_b = {r["utt_id"]: r for r in scorable_b}
    deltas = []
    for row in scorable_a:
        other = by_id_b.get(row["utt_id"])
        if not other:
            continue
        deltas.append((row["edit_distance"] - other["edit_distance"], row, other))
    deltas.sort(key=lambda item: -abs(item[0]))

    if args.top_disagreements:
        print("")
        print("largest per-utterance disagreements:")
        for delta, row, other in deltas[:args.top_disagreements]:
            print("  %s  (edit distance %d vs %d, ref %d chars)"
                  % (row["utt_id"], row["edit_distance"], other["edit_distance"],
                     row["reference_length"]))
            print("    ref : %s" % row["reference_normalized"])
            print("    %-4s: %s" % (args.name_a[:4], row["hypothesis_normalized"]))
            print("    %-4s: %s" % (args.name_b[:4], other["hypothesis_normalized"]))

    payload = dict(result)
    payload["name_a"] = args.name_a
    payload["name_b"] = args.name_b
    payload["predictions_a"] = args.a
    payload["predictions_b"] = args.b
    out_path = Path(args.out) if args.out else Path(args.a).parent / "paired_test.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print("")
    print("wrote %s" % out_path)


if __name__ == "__main__":
    main()
