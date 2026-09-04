#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3 - Produce the hospital-domain text corpus.

Writes:
    data/scripts/all_scripts.jsonl        one record per unique utterance
    data/scripts/generation_report.json   distribution, rejections, capacities
    data/scripts/rejected_samples.jsonl   every removed candidate and its reason

Run:
    python scripts/generate_scripts.py --total 18000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from data.corpus_generator import generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Generate hospital-domain scripts.")
    ap.add_argument("--total", type=int, default=18000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--near_dup_threshold", type=float, default=0.9)
    ap.add_argument("--outdir", default=str(_ROOT / "data" / "scripts"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("generating %d utterances (seed=%d, near-dup threshold=%.2f)"
          % (args.total, args.seed, args.near_dup_threshold))
    records, report = generate(total=args.total, seed=args.seed,
                               near_dup_threshold=args.near_dup_threshold)

    scripts_path = outdir / "all_scripts.jsonl"
    with scripts_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    (outdir / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    rejected_path = outdir / "rejected_samples.jsonl"
    with rejected_path.open("w", encoding="utf-8") as handle:
        for item in report["rejection_examples"]:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("")
    print("generated %d / %d requested" % (report["generated_total"], args.total))
    print("template families : %d" % report["n_template_families"])
    print("rejections        : %s" % report["rejections"])
    print("length (chars)    : %s" % report["length_stats"])
    print("")
    print("%-18s %8s %8s %8s" % ("CATEGORY", "N", "ACTUAL%", "TARGET%"))
    print("-" * 46)
    for category, target in report["target_category_pct"].items():
        print("%-18s %8d %7.2f%% %7.2f%%"
              % (category, report["by_category"].get(category, 0),
                 report["by_category_pct"].get(category, 0.0), target))

    if report["family_shortfalls"]:
        print("")
        print("families that could not fill their quota:")
        for fid, info in report["family_shortfalls"].items():
            print("  %-24s %d/%d (capacity %d)"
                  % (fid, info["produced"], info["quota"], info["capacity"]))

    print("")
    print("wrote %s" % scripts_path)
    print("wrote %s" % (outdir / "generation_report.json"))
    print("wrote %s" % rejected_path)


if __name__ == "__main__":
    main()
