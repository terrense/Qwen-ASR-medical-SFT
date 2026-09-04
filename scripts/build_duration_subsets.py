#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 4 (stage B) - Nested training subsets by audio duration.

Builds D1 (1 h), D5 (5 h), D10 (10 h) and D20 (20 h) with strict nesting:

    D1 subset of D5 subset of D10 subset of D20

Nesting matters because Phase 12 compares arms across data budgets. If the 1 h
and 5 h sets were sampled independently, a difference between budgets would
confound "more data" with "different data". Building each larger set by *adding*
to the smaller one removes that confound: the only thing that changes is how
much audio there is.

Budgets are measured in **audio hours**, not utterance counts, because that is
the quantity the research question is posed in.

Selection is stratified by domain category and balanced across speakers, so a
small budget does not accidentally become a single-category or single-voice
corpus. The exact utt_id list of every subset is written to disk.

    python scripts/build_duration_subsets.py \
        --manifest data/manifests/train_synthetic.jsonl --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from data.manifest import read_manifest, write_manifest, to_qwen_sft_format  # noqa: E402

DEFAULT_BUDGETS = [("D1", 1.0), ("D5", 5.0), ("D10", 10.0), ("D20", 20.0)]


def stratified_round_robin(rows, seed):
    """Order utterances so that any prefix is balanced and text-diverse.

    Two properties every prefix must have:

    1. balanced by domain category and speaker - utterances are grouped by
       (domain_category, speaker_id) and dealt round-robin over those groups,
       so taking the first N yields roughly proportional categories and roughly
       uniform speakers for *every* N.

    2. unique scripts first - when the corpus renders one script with several
       voices (needed to reach the larger hour budgets, since the utterances are
       short), the orderingis split into passes: pass 0 contains the first
       rendering of every script, pass 1 the second, and so on. Small budgets
       therefore consume distinct sentences rather than the same sentence in two
       voices, which keeps D1 and D5 as textually diverse as the corpus allows.
       Without this, a 1-hour subset could spend half its budget re-hearing the
       same text.
    """
    rng = random.Random(seed)

    # Assign each row its occurrence index within its script.
    occurrence = defaultdict(int)
    passes = defaultdict(list)
    for row in rows:
        key = row.get("script_id") or row.get("text") or row["utt_id"]
        index = occurrence[key]
        occurrence[key] += 1
        passes[index].append(row)

    ordered = []
    for pass_index in sorted(passes):
        ordered.extend(_balanced_order(passes[pass_index], rng))
    return ordered


def _balanced_order(rows, rng):
    """Deal one pass round-robin over (category, speaker) groups."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("domain_category"), row.get("speaker_id"))].append(row)

    category_totals = Counter()
    for row in rows:
        category_totals[row.get("domain_category")] += 1
    total = sum(category_totals.values())
    if not total:
        return []

    for bucket in groups.values():
        rng.shuffle(bucket)

    by_category = defaultdict(list)
    for (category, _speaker), bucket in groups.items():
        by_category[category].append(bucket)
    for buckets in by_category.values():
        rng.shuffle(buckets)

    cursors = {category: 0 for category in by_category}
    credit = {category: 0.0 for category in by_category}
    remaining = {category: sum(len(b) for b in buckets)
                 for category, buckets in by_category.items()}

    ordered = []
    while sum(remaining.values()) > 0:
        for category in credit:
            if remaining[category] > 0:
                credit[category] += category_totals[category] / total
        category = max((c for c in credit if remaining[c] > 0),
                       key=lambda c: credit[c])
        credit[category] -= 1.0

        buckets = by_category[category]
        for _ in range(len(buckets)):
            index = cursors[category] % len(buckets)
            cursors[category] += 1
            if buckets[index]:
                ordered.append(buckets[index].pop())
                remaining[category] -= 1
                break
    return ordered


def build_subsets(rows, budgets, seed):
    """Return nested subsets, largest budget last."""
    ordered = stratified_round_robin(rows, seed)

    subsets = OrderedDict()
    for name, hours in sorted(budgets, key=lambda b: b[1]):
        target_seconds = hours * 3600.0
        taken, accumulated = [], 0.0
        for row in ordered:
            if accumulated >= target_seconds:
                break
            taken.append(row)
            accumulated += row.get("duration", 0) or 0
        subsets[name] = {"rows": taken, "hours": accumulated / 3600.0,
                         "target_hours": hours}
    return subsets


def verify_nesting(subsets):
    """Assert each subset is contained in the next larger one."""
    names = list(subsets)
    problems = []
    for smaller, larger in zip(names, names[1:]):
        small_ids = {r["utt_id"] for r in subsets[smaller]["rows"]}
        large_ids = {r["utt_id"] for r in subsets[larger]["rows"]}
        missing = small_ids - large_ids
        if missing:
            problems.append("%s is not contained in %s (%d utterances missing)"
                            % (smaller, larger, len(missing)))
    if problems:
        raise AssertionError("; ".join(problems))
    return True


def main():
    ap = argparse.ArgumentParser(description="Nested duration-budget subsets.")
    ap.add_argument("--manifest", default=str(_ROOT / "data" / "manifests" / "train_synthetic.jsonl"))
    ap.add_argument("--outdir", default=str(_ROOT / "data" / "manifests"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budgets", default="D1:1,D5:5,D10:10,D20:20")
    ap.add_argument("--write_sft", action="store_true",
                    help="also emit the official Qwen3-ASR fine-tuning JSONL")
    args = ap.parse_args()

    budgets = []
    for item in args.budgets.split(","):
        name, hours = item.split(":")
        budgets.append((name.strip(), float(hours)))

    rows = read_manifest(args.manifest)
    total_hours = sum(r.get("duration", 0) or 0 for r in rows) / 3600.0
    print("source manifest: %s" % args.manifest)
    print("  %d utterances, %.3f audio hours" % (len(rows), total_hours))

    largest = max(hours for _, hours in budgets)
    if total_hours < largest:
        print("\nWARNING: the pool holds %.2f h but the largest budget asks for "
              "%.2f h. That subset will be short and is reported as such."
              % (total_hours, largest))

    subsets = build_subsets(rows, budgets, args.seed)
    verify_nesting(subsets)

    outdir = Path(args.outdir)
    report = OrderedDict([("source_manifest", args.manifest),
                          ("seed", args.seed),
                          ("pool_utterances", len(rows)),
                          ("pool_hours", round(total_hours, 4)),
                          ("nesting_verified", True),
                          ("subsets", OrderedDict())])

    print("")
    print("%-6s %10s %10s %8s %9s %9s" % ("SUBSET", "TARGET_H", "ACTUAL_H",
                                          "UTTS", "SPEAKERS", "CATS"))
    print("-" * 60)
    for name, info in subsets.items():
        subset_rows = info["rows"]
        budget_tag = name.lower().replace("d", "") + "h"
        manifest_path = outdir / ("train_%s.jsonl" % budget_tag)
        write_manifest(subset_rows, manifest_path)

        ids_path = outdir / "subset_ids" / ("%s_utt_ids.txt" % name)
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.write_text("\n".join(r["utt_id"] for r in subset_rows) + "\n",
                            encoding="utf-8")

        if args.write_sft:
            to_qwen_sft_format(subset_rows,
                               str(outdir / ("train_%s_sft.jsonl" % budget_tag)))

        categories = Counter(r.get("domain_category") for r in subset_rows)
        speakers = Counter(r.get("speaker_id") for r in subset_rows)
        report["subsets"][name] = {
            "target_hours": info["target_hours"],
            "actual_hours": round(info["hours"], 4),
            "n_utterances": len(subset_rows),
            "n_speakers": len(speakers),
            "manifest": str(manifest_path),
            "utt_id_list": str(ids_path),
            "by_category": dict(categories),
            "by_category_pct": {k: round(100.0 * v / max(1, len(subset_rows)), 2)
                                for k, v in categories.items()},
            "speaker_min": min(speakers.values()) if speakers else 0,
            "speaker_max": max(speakers.values()) if speakers else 0,
        }
        print("%-6s %10.2f %10.3f %8d %9d %9d"
              % (name, info["target_hours"], info["hours"], len(subset_rows),
                 len(speakers), len(categories)))

    (outdir / "duration_subsets_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print("nesting verified: D1 subset of D5 subset of D10 subset of D20")
    print("wrote %s" % (outdir / "duration_subsets_report.json"))


if __name__ == "__main__":
    main()
