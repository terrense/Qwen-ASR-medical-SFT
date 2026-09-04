#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 9 - Build the medical entity lexicon.

The lexicon is derived exclusively from the generation vocabulary
(``src/data/hospital_vocab.py``). It never reads predictions, references from a
test set, or model output of any kind, which is what makes the terminology
metrics independent of the systems being compared.

Terms are normalized with the same rules used at scoring time, so a lexicon
entry and a normalized transcript are directly comparable.

Writes:
    data/medical_lexicon.json
    results/metrics/lexicon_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from data import hospital_vocab as V  # noqa: E402
from evaluation import normalization as norm  # noqa: E402
from evaluation.metrics import MedicalLexicon  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Build the medical entity lexicon.")
    ap.add_argument("--out", default=str(_ROOT / "data" / "medical_lexicon.json"))
    ap.add_argument("--report", default=str(_ROOT / "results" / "metrics" / "lexicon_report.json"))
    ap.add_argument("--scripts", default=str(_ROOT / "data" / "scripts" / "all_scripts.jsonl"),
                    help="Optional: report term coverage over the generated corpus")
    ap.add_argument("--min_length", type=int, default=1,
                    help="Drop terms shorter than this after normalization")
    args = ap.parse_args()

    config = norm.DEFAULT_CONFIG
    raw = V.build_entity_lexicon()

    # Normalize every term with the scoring-time rules so lexicon lookups and
    # normalized hypotheses live in the same string space.
    normalized = {}
    dropped = []
    for category, terms in raw.items():
        bucket = []
        for term in terms:
            clean = norm.normalize(term, config)
            if len(clean) < args.min_length:
                dropped.append({"term": term, "reason": "too short after normalization"})
                continue
            bucket.append(clean)
        normalized[category] = sorted(set(bucket))

    lexicon = MedicalLexicon(normalized)
    collisions = lexicon.validate()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lexicon.to_json(out_path, extra={
        "normalization": config.to_dict(),
        "normalization_fingerprint": config.fingerprint(),
        "source": "src/data/hospital_vocab.py (generation vocabulary only)",
        "independent_of_predictions": True,
    })

    report = {
        "n_terms": len(lexicon.all_terms),
        "n_categories": len(normalized),
        "terms_per_category": {k: len(v) for k, v in normalized.items()},
        "category_collisions": collisions,
        "dropped": dropped,
        "normalization_fingerprint": config.fingerprint(),
        "longest_terms": sorted(lexicon.all_terms, key=len, reverse=True)[:15],
        "shortest_terms": sorted(lexicon.all_terms, key=len)[:15],
    }

    scripts_path = Path(args.scripts)
    if scripts_path.exists():
        hits = Counter()
        n_with_terms = 0
        n_scripts = 0
        for line in scripts_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            n_scripts += 1
            text = norm.normalize(json.loads(line)["text"], config)
            found = lexicon.find(text)
            if found:
                n_with_terms += 1
            hits.update(found)
        report["corpus_coverage"] = {
            "scripts_file": str(scripts_path),
            "n_scripts": n_scripts,
            "n_scripts_with_at_least_one_term": n_with_terms,
            "pct_scripts_with_terms": round(100.0 * n_with_terms / max(1, n_scripts), 2),
            "n_distinct_terms_seen": len(hits),
            "n_terms_never_seen": len(lexicon.all_terms) - len(hits),
            "most_frequent": hits.most_common(20),
            "never_seen_sample": sorted(set(lexicon.all_terms) - set(hits))[:30],
        }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    print("lexicon terms      : %d across %d categories"
          % (report["n_terms"], report["n_categories"]))
    for category, count in report["terms_per_category"].items():
        print("  %-16s %d" % (category, count))
    if collisions:
        print("category collisions: %d (%s)" % (len(collisions), list(collisions)[:5]))
    else:
        print("category collisions: none")
    if "corpus_coverage" in report:
        cov = report["corpus_coverage"]
        print("corpus coverage    : %.2f%% of %d scripts contain >=1 term"
              % (cov["pct_scripts_with_terms"], cov["n_scripts"]))
        print("                     %d distinct terms seen, %d never used"
              % (cov["n_distinct_terms_seen"], cov["n_terms_never_seen"]))
    print("")
    print("wrote %s" % out_path)
    print("wrote %s" % report_path)


if __name__ == "__main__":
    main()
