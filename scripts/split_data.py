#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 4 (stage A) - Script-disjoint train / dev / test split.

Splitting is done by ``template_family``, never by individual utterance. A
template family is a semantic sentence pattern, so holding a family out
guarantees that a test utterance is not a re-voicing of a pattern the model was
trained on. Splitting only by waveform would leave the same sentence pattern on
both sides and inflate the apparent gain from adaptation.

Families are held out *within* each domain category so that every split still
covers all nine Phase 3 categories.

Speaker disjointness is enforced later, at TTS time (stage B): the synthetic
voice inventory is partitioned into train / dev / test voice pools, and the
cross-TTS test uses a different engine entirely.

Cross-TTS test scripts: the specification asks for "entirely held-out scripts
and voices". This script marks a subset of the *test* families for cross-TTS
rendering. Those scripts are held out from training exactly as required, and
pairing them with the same scripts in the in-domain synthetic test additionally
isolates the TTS engine as the only changing variable. Both sets are recorded
separately so either analysis is possible.

Writes into data/manifests/splits/:
    train_scripts.jsonl  dev_scripts.jsonl  test_scripts.jsonl
    cross_tts_scripts.jsonl
    family_assignment.json    which family went to which split
    split_report.json         counts and verification
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


def load_scripts(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def assign_families(scripts, test_frac, dev_frac, seed):
    """Assign whole families to test / dev / train inside each domain category.

    Template families differ enormously in size (one family can hold a third of
    a category), so filling one split at a time overshoots badly. Instead each
    category is partitioned with a largest-deficit heuristic: families are
    considered largest-first and each goes to whichever split is currently
    furthest below its target share. That is the standard greedy approximation
    for balanced partitioning and keeps every split close to its target while
    preserving whole-family disjointness.

    Every split is guaranteed at least one family per category, so the
    per-category CER breakdown is defined everywhere.
    """
    rng = random.Random(seed)

    by_category = defaultdict(list)
    family_sizes = Counter()
    family_category = {}
    for row in scripts:
        family = row["template_family"]
        family_sizes[family] += 1
        family_category[family] = row["domain_category"]
    for family, category in family_category.items():
        by_category[category].append(family)

    train_frac = 1.0 - test_frac - dev_frac
    if train_frac <= 0:
        raise ValueError("test_frac + dev_frac must be < 1.0")
    targets = {"train": train_frac, "dev": dev_frac, "test": test_frac}

    assignment = {}
    for category, families in sorted(by_category.items()):
        if len(families) < 3:
            raise ValueError(
                "category '%s' has only %d template families; at least 3 are "
                "needed for a family-disjoint train/dev/test split"
                % (category, len(families)))

        # Shuffle first so that equal-sized families do not always break the
        # same way, then sort by size so the big ones are placed first.
        families = sorted(families)
        rng.shuffle(families)
        families.sort(key=lambda f: -family_sizes[f])

        total = sum(family_sizes[f] for f in families)
        current = {"train": 0, "dev": 0, "test": 0}
        holding = {"train": [], "dev": [], "test": []}

        for family in families:
            size = family_sizes[family]
            # Splits that still have no family come first, so none ends empty.
            empty = [s for s in ("test", "dev", "train") if not holding[s]]
            remaining = len(families) - sum(len(v) for v in holding.values())
            if empty and remaining <= len(empty):
                choice = empty[0]
            else:
                choice = max(("train", "dev", "test"),
                             key=lambda s: targets[s] * total - current[s])
            holding[choice].append(family)
            current[choice] += size

        for split, members in holding.items():
            for family in members:
                assignment[family] = split

    return assignment, family_sizes, family_category


def main():
    ap = argparse.ArgumentParser(description="Script-disjoint corpus split.")
    ap.add_argument("--scripts", default=str(_ROOT / "data" / "scripts" / "all_scripts.jsonl"))
    ap.add_argument("--outdir", default=str(_ROOT / "data" / "manifests" / "splits"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--dev_frac", type=float, default=0.10)
    ap.add_argument("--cross_tts_per_category", type=int, default=40,
                    help="test-family scripts per category also rendered by the "
                         "second TTS engine")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scripts = load_scripts(args.scripts)
    print("loaded %d scripts from %s" % (len(scripts), args.scripts))

    assignment, family_sizes, family_category = assign_families(
        scripts, args.test_frac, args.dev_frac, args.seed)

    buckets = {"train": [], "dev": [], "test": []}
    for row in scripts:
        row = OrderedDict(row)
        split = assignment[row["template_family"]]
        row["split"] = split
        buckets[split].append(row)

    # Cross-TTS scripts are drawn from the test pool, stratified by category.
    rng = random.Random(args.seed + 1)
    by_category = defaultdict(list)
    for row in buckets["test"]:
        by_category[row["domain_category"]].append(row)
    cross = []
    for category in sorted(by_category):
        pool = sorted(by_category[category], key=lambda r: r["script_id"])
        rng.shuffle(pool)
        cross.extend(pool[:args.cross_tts_per_category])
    cross_ids = {row["script_id"] for row in cross}
    for row in buckets["test"]:
        row["cross_tts_selected"] = row["script_id"] in cross_ids

    # --- verification: no family may appear in two splits -------------------
    family_splits = defaultdict(set)
    for split, rows in buckets.items():
        for row in rows:
            family_splits[row["template_family"]].add(split)
    leaked = {f: sorted(s) for f, s in family_splits.items() if len(s) > 1}
    if leaked:
        raise AssertionError("template families appear in multiple splits: %s" % leaked)

    text_splits = defaultdict(set)
    for split, rows in buckets.items():
        for row in rows:
            text_splits[row["text"]].add(split)
    text_leaks = {t: sorted(s) for t, s in text_splits.items() if len(s) > 1}
    if text_leaks:
        raise AssertionError("%d identical texts appear in multiple splits, e.g. %s"
                             % (len(text_leaks), list(text_leaks)[:3]))

    for name, rows in list(buckets.items()) + [("cross_tts", cross)]:
        path = outdir / ("%s_scripts.jsonl" % name)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    families_by_split = defaultdict(list)
    for family, split in sorted(assignment.items()):
        families_by_split[split].append(family)

    report = OrderedDict()
    report["seed"] = args.seed
    report["source"] = args.scripts
    report["n_scripts"] = len(scripts)
    report["test_frac_target"] = args.test_frac
    report["dev_frac_target"] = args.dev_frac
    report["family_disjoint_verified"] = True
    report["text_disjoint_verified"] = True
    for split in ("train", "dev", "test"):
        rows = buckets[split]
        report[split] = {
            "n_scripts": len(rows),
            "pct": round(100.0 * len(rows) / len(scripts), 2),
            "n_families": len(families_by_split[split]),
            "by_category": dict(Counter(r["domain_category"] for r in rows)),
        }
    report["cross_tts"] = {
        "n_scripts": len(cross),
        "drawn_from": "test families (held out from training)",
        "by_category": dict(Counter(r["domain_category"] for r in cross)),
    }
    report["families_by_split"] = {k: v for k, v in families_by_split.items()}

    (outdir / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "family_assignment.json").write_text(
        json.dumps(OrderedDict(sorted(assignment.items())), ensure_ascii=False,
                   indent=2), encoding="utf-8")

    print("")
    print("%-10s %8s %7s %9s" % ("SPLIT", "SCRIPTS", "PCT", "FAMILIES"))
    print("-" * 38)
    for split in ("train", "dev", "test"):
        info = report[split]
        print("%-10s %8d %6.2f%% %9d" % (split, info["n_scripts"], info["pct"],
                                         info["n_families"]))
    print("%-10s %8d %6s  %9s" % ("cross_tts", len(cross), "-", "(from test)"))

    print("")
    print("per-category script counts")
    print("%-18s %8s %8s %8s %10s" % ("CATEGORY", "TRAIN", "DEV", "TEST", "CROSS_TTS"))
    print("-" * 58)
    categories = sorted({r["domain_category"] for r in scripts})
    for category in categories:
        print("%-18s %8d %8d %8d %10d" % (
            category,
            report["train"]["by_category"].get(category, 0),
            report["dev"]["by_category"].get(category, 0),
            report["test"]["by_category"].get(category, 0),
            report["cross_tts"]["by_category"].get(category, 0)))

    print("")
    print("family disjointness verified: no family spans two splits")
    print("text disjointness verified  : no identical text spans two splits")
    print("wrote %s" % outdir)


if __name__ == "__main__":
    main()
