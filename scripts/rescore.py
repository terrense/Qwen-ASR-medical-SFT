#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-score an existing predictions.jsonl under a different normalization.

Decoding is expensive; normalization is not. This script recomputes CER and
terminology metrics from the *raw* reference and hypothesis strings already
stored in a predictions file, under a different `NormalizationConfig`. No model
is loaded and no audio is touched, so a normalization question can be answered
in seconds instead of re-running inference.

The motivating case: Qwen3-ASR-1.7B tends to emit Arabic digits ("500米") where
AISHELL-1 references use Chinese numerals ("五百米"). Under the default rules
that difference is charged as several substitutions even though nothing was
misheard. Turning on `normalize_numbers` measures how much of a reported gap is
orthographic rather than acoustic.

This is a *diagnostic*, not a replacement for the primary metric. The headline
CER stays on the default rules; a number-normalized CER is reported alongside
and labelled, exactly as the Phase 8 specification requires.

    python scripts/rescore.py --predictions experiments/.../predictions.jsonl \
        --normalize_numbers
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from evaluation import metrics as M  # noqa: E402
from evaluation import normalization as norm  # noqa: E402


def rescore(rows, config, lexicon=None):
    records = []
    for row in rows:
        ref = norm.normalize(row.get("reference_raw", ""), config)
        hyp = norm.normalize(row.get("hypothesis_raw", ""), config)
        cer = M.utterance_cer(ref, hyp)
        if cer is None:
            continue
        record = dict(cer)
        record["utt_id"] = row.get("utt_id")
        record["reference_normalized"] = ref
        record["hypothesis_normalized"] = hyp
        for key in ("condition", "domain_category", "source", "speaker_id"):
            record[key] = row.get(key)
        if lexicon:
            record.update(M.term_metrics(ref, hyp, lexicon))
        records.append(record)
    return records


def main():
    ap = argparse.ArgumentParser(description="Re-score predictions offline.")
    ap.add_argument("--predictions", required=True, nargs="+",
                    help="one or more predictions.jsonl files")
    ap.add_argument("--lexicon", default=str(_ROOT / "data" / "medical_lexicon.json"))
    ap.add_argument("--normalize_numbers", action="store_true")
    ap.add_argument("--traditional_to_simplified", action="store_true")
    ap.add_argument("--keep_punctuation", action="store_true")
    ap.add_argument("--out", default=None,
                    help="write the comparison JSON here")
    args = ap.parse_args()

    lexicon = None
    lex_path = Path(args.lexicon)
    if lex_path.exists():
        lexicon = M.MedicalLexicon.from_json(lex_path)

    default = norm.DEFAULT_CONFIG
    variant = norm.NormalizationConfig(
        normalize_numbers=args.normalize_numbers,
        traditional_to_simplified=args.traditional_to_simplified,
        strip_punctuation=not args.keep_punctuation)

    print("baseline rules : %s" % default.fingerprint())
    print("variant  rules : %s" % variant.fingerprint())
    print(norm.describe(variant))
    print("")

    summary = OrderedDict()
    for path in args.predictions:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        base = M.corpus_cer(rescore(rows, default, lexicon))
        alt = M.corpus_cer(rescore(rows, variant, lexicon))
        delta = (base["cer"] - alt["cer"]) if (base["cer"] is not None
                                               and alt["cer"] is not None) else None
        summary[path] = {
            "n_utterances": base["n_utterances"],
            "cer_default": base["cer"],
            "cer_variant": alt["cer"],
            "absolute_reduction": delta,
            "relative_reduction": (delta / base["cer"]) if (delta and base["cer"]) else None,
        }
        print("%-58s" % path)
        print("   default CER  %.4f  (%.2f%%)" % (base["cer"], 100 * base["cer"]))
        print("   variant CER  %.4f  (%.2f%%)" % (alt["cer"], 100 * alt["cer"]))
        if delta is not None:
            print("   difference   %+.4f  (%+.2f pp, %.1f%% of the default CER)"
                  % (-delta, -100 * delta,
                     100 * delta / base["cer"] if base["cer"] else 0.0))
        print("")

    payload = {"baseline_rules": default.to_dict(),
               "variant_rules": variant.to_dict(),
               "baseline_fingerprint": default.fingerprint(),
               "variant_fingerprint": variant.fingerprint(),
               "results": summary}
    out_path = Path(args.out) if args.out else (
        Path(args.predictions[0]).parent / "rescore_comparison.json")
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print("wrote %s" % out_path)


if __name__ == "__main__":
    main()
